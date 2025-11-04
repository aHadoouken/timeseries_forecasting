"""
Trainer class for vibration prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import os
import time
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from training.losses import CombinedLoss, TrajectoryLoss
from training.metrics import VibrationMetrics, MetricsTracker
from models.base_model import BaseVibrationModel
from data.augmentation import BatchAugmenter, VibrationAugmenter

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for vibration prediction models.
    """

    def __init__(
        self,
        model: BaseVibrationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training parameters
        # Convert config values to appropriate types
        self.epochs = int(config.get("epochs", 100))
        self.learning_rate = float(config.get("learning_rate", 0.001))
        self.weight_decay = float(config.get("weight_decay", 1e-5))
        self.gradient_clipping = float(config.get("gradient_clipping", 1.0))
        self.early_stopping_patience = config.get("early_stopping_patience", 15)

        # Logging parameters
        self.log_interval = config.get("log_interval", 10)
        self.save_interval = config.get("save_interval", 50)
        self.plot_predictions = config.get("plot_predictions", True)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize loss function
        # loss_weights = config.get('loss', {}).get('weights', {
        #     'trajectory': 1.0, 'amplitude': 2.0, 'stability': 0.5, 'physics': 0.1
        # })
        # self.criterion = CombinedLoss(loss_weights)
        self.criterion = TrajectoryLoss(loss_type="mse")

        # Initialize metrics
        self.train_metrics = VibrationMetrics()
        self.val_metrics = VibrationMetrics()
        self.metrics_tracker = MetricsTracker()

        # Initialize augmenter
        if config.get("augmentation", {}).get("enabled", True):
            augmenter = VibrationAugmenter(config.get("augmentation", {}))
            self.batch_augmenter = BatchAugmenter(augmenter)
        else:
            self.batch_augmenter = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {},
            "val_metrics": {},
        }

        # Create output directory
        self.output_dir = config.get("output_dir", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer based on configuration.
        """
        print(self.config.get("optimizer", "adam"))
        optimizer_type = self.config.get("optimizer", "adam").lower()

        if optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler.
        """
        scheduler_type = self.config.get("scheduler", "reduce_lr_on_plateau").lower()

        if scheduler_type == "reduce_lr_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.get("scheduler_factor", 0.5),
                patience=self.config.get("scheduler_patience", 10),
            )
        elif scheduler_type == "cosine_annealing":
            return CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        total_samples = 0
        loss_components = {}

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            features = batch["features"]
            targets = batch["targets"]
            batch_size, seq_len, _ = features.shape
            batch_size, horizon_len, _ = targets.shape

            # Apply augmentation
            if self.batch_augmenter is not None:
                batch = self.batch_augmenter.augment_batch(batch, augment_prob=0.5)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch)["trajectory"]

            predictions = outputs[:, seq_len - 1:-1, :]
            targets = batch["targets"]

            # Compute loss
            loss = self.criterion(
                predictions,
                targets,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            # batch_size = batch['features'].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Accumulate loss components
            # for key, value in individual_losses.items():
            #     if key not in loss_components:
            #         loss_components[key] = 0.0
            #     loss_components[key] += value.item() * batch_size

            # Update training metrics
            # if "trajectory" in outputs:
            self.train_metrics.update(
                predictions=predictions,
                targets=targets,
                # amplitudes_pred=outputs.get("amplitude"),
                # amplitudes_true=batch.get("max_amplitude"),
                # parameters=batch["parameters"],
                # trajectory_ids=batch.get("trajectory_id"),
            )

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/total_samples:.4f}",
                }
            )

            # Log batch metrics
            if batch_idx % self.log_interval == 0:
                logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}")

        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        # avg_loss_components = {k: v / total_samples for k, v in loss_components.items()}

        # Compute detailed metrics
        detailed_metrics = self.train_metrics.compute_all_metrics()

        epoch_metrics = {"loss": avg_loss, **detailed_metrics}

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        """
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        total_samples = 0
        loss_components = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                features = batch["features"]
                targets = batch["targets"]
                batch_size, seq_len, _ = features.shape
                batch_size, horizon_len, _ = targets.shape

                # Forward pass
                outputs = self.model(batch)

                predictions = outputs["trajectory"][:, seq_len - 1:-1, :]
                targets = batch["targets"]

                # Compute loss
                loss = self.criterion(
                    predictions,
                    targets,
                )

                # Update metrics
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Accumulate loss components
                # for key, value in individual_losses.items():
                #     if key not in loss_components:
                #         loss_components[key] = 0.0
                #     loss_components[key] += value.item() * batch_size

                # Update validation metrics
                self.val_metrics.update(
                    predictions=predictions,
                    targets=targets,
                    # amplitudes_pred=outputs.get("amplitude"),
                    # amplitudes_true=batch.get("max_amplitude"),
                    # parameters=batch["parameters"],
                    # trajectory_ids=batch.get("trajectory_id"),
                )

        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        # avg_loss_components = {k: v / total_samples for k, v in loss_components.items()}

        # Compute detailed metrics
        detailed_metrics = self.val_metrics.compute_all_metrics()

        epoch_metrics = {"loss": avg_loss, **detailed_metrics}

        return epoch_metrics

    def train(self) -> Dict:
        """
        Main training loop.
        """
        logger.info(f"Starting training for {self.epochs} epochs")

        start_time = time.time()
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate epoch
            val_metrics = self.validate_epoch()

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Update metrics tracker
            self.metrics_tracker.update(epoch, train_metrics, val_metrics)

            # Update training history
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["val_loss"].append(val_metrics["loss"])

            # Check for best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0

                # Save best model
                self.save_checkpoint("best_model.pth", epoch, is_best=True)
            else:
                epochs_without_improvement += 1

            # Log epoch results
            logger.info(
                f"Epoch {epoch:3d}: "
                f"Train Loss: {train_metrics['loss']:.6f}, "
                f"Val Loss: {val_metrics['loss']:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Print detailed metrics periodically
            if epoch % (self.log_interval * 5) == 0:
                print(f"\nEpoch {epoch} Detailed Metrics:")
                print("-" * 40)

                # Print key metrics
                key_metrics = ["rmse_overall", "amplitude_rmse", "bifurcation_f1"]
                for metric in key_metrics:
                    if metric in val_metrics:
                        print(f"{metric:20s}: {val_metrics[metric]:.6f}")

            # Save checkpoint periodically
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth", epoch)

            # Plot predictions periodically
            # if self.plot_predictions and epoch % (self.save_interval // 2) == 0:
            #     self.plot_sample_predictions(epoch)

            # Early stopping
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Training completed
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(
                f"Loaded best model with validation loss: {self.best_val_loss:.6f}"
            )

        # Final evaluation
        final_val_metrics = self.validate_epoch()

        # Save final results
        self.save_training_results()

        return {
            "best_val_loss": self.best_val_loss,
            "final_metrics": final_val_metrics,
            "training_time": training_time,
            "total_epochs": self.current_epoch + 1,
        }

    def save_checkpoint(self, filename: str, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        """
        filepath = os.path.join(self.output_dir, filename)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "training_history": self.training_history,
        }

        torch.save(checkpoint, filepath)

        if is_best:
            logger.info(f"Best model saved to {filepath}")
        else:
            logger.debug(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> int:
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get(
            "training_history", self.training_history
        )

        epoch = checkpoint["epoch"]
        logger.info(f"Checkpoint loaded from {filepath}, epoch {epoch}")

        return epoch

    def plot_sample_predictions(self, epoch: int, n_samples: int = 3):
        """
        Plot sample predictions for visualization.
        """
        self.model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            outputs = self.model(batch["features"], batch["parameters"])

        # Convert to numpy
        features = batch["features"][:n_samples].cpu().numpy()
        targets = batch["targets"][:n_samples].cpu().numpy()
        predictions = outputs["trajectory"][:n_samples].cpu().numpy()

        # Create plots
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Position plot
            axes[i, 0].plot(features[i, :, 0], label="Input", alpha=0.7)
            axes[i, 0].plot(
                range(len(features[i]), len(features[i]) + len(targets[i])),
                targets[i, :, 0],
                label="Target",
                color="green",
            )
            axes[i, 0].plot(
                range(len(features[i]), len(features[i]) + len(predictions[i])),
                predictions[i, :, 0],
                label="Prediction",
                color="red",
                linestyle="--",
            )
            axes[i, 0].set_title(f"Sample {i+1} - Position")
            axes[i, 0].legend()
            axes[i, 0].grid(True)

            # Velocity plot
            axes[i, 1].plot(features[i, :, 1], label="Input", alpha=0.7)
            axes[i, 1].plot(
                range(len(features[i]), len(features[i]) + len(targets[i])),
                targets[i, :, 1],
                label="Target",
                color="green",
            )
            axes[i, 1].plot(
                range(len(features[i]), len(features[i]) + len(predictions[i])),
                predictions[i, :, 1],
                label="Prediction",
                color="red",
                linestyle="--",
            )
            axes[i, 1].set_title(f"Sample {i+1} - Velocity")
            axes[i, 1].legend()
            axes[i, 1].grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"predictions_epoch_{epoch}.png"), dpi=150
        )
        plt.close()

    def plot_training_history(self):
        """
        Plot training history.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        axes[0, 0].plot(self.training_history["train_loss"], label="Train")
        axes[0, 0].plot(self.training_history["val_loss"], label="Validation")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Learning rate
        if hasattr(self.scheduler, "get_last_lr"):
            lr_history = [group["lr"] for group in self.optimizer.param_groups]
            axes[0, 1].plot(lr_history)
            axes[0, 1].set_title("Learning Rate")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].set_yscale("log")
            axes[0, 1].grid(True)

        # Metrics history (if available)
        train_rmse, val_rmse = self.metrics_tracker.get_metric_history("rmse_overall")
        if train_rmse and val_rmse:
            axes[1, 0].plot(train_rmse, label="Train RMSE")
            axes[1, 0].plot(val_rmse, label="Val RMSE")
            axes[1, 0].set_title("RMSE History")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("RMSE")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Amplitude metrics
        train_amp, val_amp = self.metrics_tracker.get_metric_history("amplitude_rmse")
        if train_amp and val_amp:
            axes[1, 1].plot(train_amp, label="Train Amp RMSE")
            axes[1, 1].plot(val_amp, label="Val Amp RMSE")
            axes[1, 1].set_title("Amplitude RMSE History")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Amplitude RMSE")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_history.png"), dpi=150)
        plt.close()

    def save_training_results(self):
        """
        Save comprehensive training results.
        """
        results = {
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "final_train_metrics": self.train_metrics.compute_all_metrics(),
            "final_val_metrics": self.val_metrics.compute_all_metrics(),
            "model_info": self.model.get_model_info(),
        }

        # Save as torch file
        torch.save(results, os.path.join(self.output_dir, "training_results.pth"))

        # Plot training history
        self.plot_training_history()

        # Print final summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best Validation Loss: {self.best_val_loss:.6f}")
        print(f"Total Epochs: {self.current_epoch + 1}")

        # Print final metrics summary
        self.val_metrics.print_summary()

        logger.info("Training results saved to " + self.output_dir)
