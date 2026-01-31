"""
Trainer class for vibration prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
from collections import defaultdict
import os
import time
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import matplotlib

# Try to use interactive backend if available
try:
    import IPython

    if IPython.get_ipython() is not None:
        # We're in Jupyter/IPython environment
        matplotlib.use("module://ipykernel.pylab.backend_inline")
except:
    # Use default backend
    pass

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
        const_parameters: Dict,
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
        self.const_parameters = const_parameters

        # Training parameters
        # Convert config values to appropriate types
        self.epochs = int(config.get("epochs", 100))
        self.learning_rate = float(config.get("learning_rate", 0.001))
        self.weight_decay = float(config.get("weight_decay", 1e-5))
        self.gradient_clipping = float(config.get("gradient_clipping", 1.0))
        self.early_stopping_patience = config.get("early_stopping_patience", 15)
        self.pretrain_val = config.get("pretrain_val", True)

        # Scheduled Sampling parameters
        self.use_scheduled_sampling = config.get("scheduled_sampling", {}).get(
            "enabled", True
        )
        self.initial_teacher_forcing_ratio = config.get("scheduled_sampling", {}).get(
            "initial_ratio", 1.0
        )
        self.teacher_forcing_decay = config.get("scheduled_sampling", {}).get(
            "decay_rate", 0.99
        )
        self.min_teacher_forcing_ratio = config.get("scheduled_sampling", {}).get(
            "min_ratio", 0.1
        )
        self.current_teacher_forcing_ratio = self.initial_teacher_forcing_ratio

        # Logging parameters
        self.log_interval = config.get("log_interval", 10)
        self.save_interval = config.get("save_interval", 50)
        self.plot_predictions = config.get("plot_predictions", True)
        self.plot_realtime = config.get("plot_realtime", True)
        self.plot_update_interval = config.get(
            "plot_update_interval", 1
        )  # Update plot every N epochs

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize loss function
        loss_weights = config.get('loss', {}).get('weights', {
            'trajectory': 0.5, 'physics': 0.5
        })
        self.criterion = CombinedLoss(loss_weights)
        # self.criterion = TrajectoryLoss(loss_type="mse")

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
            "learning_rates": [],
            "teacher_forcing_ratios": [],
            "step_losses": [],  # Initialize step_losses list
            "loss_components": defaultdict(list),
            "step_loss_components": defaultdict(list),
        }

        # Real-time plotting setup
        self.fig = None
        self.axes = None
        self.loss_lines = None
        self.is_notebook = self._check_notebook_environment()

        # Create output directory
        self.output_dir = config.get("output_dir", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Initialize real-time plot if enabled
        if self.plot_realtime:
            self._init_realtime_plot()

    def _check_notebook_environment(self) -> bool:
        """
        Check if we're running in a notebook environment.
        """
        try:
            import IPython

            return IPython.get_ipython() is not None
        except:
            return False

    def _init_realtime_plot(self):
        """
        Initialize the real-time loss plotting figure.
        """
        if self.is_notebook:
            # Use inline plotting for notebooks
            plt.ioff()  # Turn off interactive mode for notebooks
        else:
            # Use interactive mode for regular Python scripts
            plt.ion()

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

        # Initialize empty plots
        self.loss_lines = {}

        # Loss plot
        (self.loss_lines["train_loss"],) = self.axes[0, 0].plot(
            [], [], "b-", label="Train Loss", linewidth=0.5
        )
        (self.loss_lines["train_loss_traj"],) = self.axes[0, 0].plot(
            [], [], "c-", label="Train Loss Traj", linewidth=0.5
        )
        (self.loss_lines["train_loss_physics"],) = self.axes[0, 0].plot(
            [], [], "g-", label="Train Loss Physics", linewidth=0.5
        )

        (self.loss_lines["val_loss"],) = self.axes[0, 0].plot(
            [], [], "r-", label="Val Loss", linewidth=0.5
        )
        self.axes[0, 0].set_title("Loss Curves")
        self.axes[0, 0].set_xlabel("Epoch")
        self.axes[0, 0].set_ylabel("Loss")
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)

        # Loss difference plot (train - val)
        (self.loss_lines["log_train_loss"],) = self.axes[0, 1].plot(
            [], [], "b-", label="Log Train Loss", linewidth=0.5
        )
        (self.loss_lines["log_train_loss_traj"],) = self.axes[0, 1].plot(
            [], [], "c-", label="Train Loss Traj", linewidth=0.5
        )
        (self.loss_lines["log_train_loss_physics"],) = self.axes[0, 1].plot(
            [], [], "g-", label="Train Loss Physics", linewidth=0.5
        )
        (self.loss_lines["log_val_loss"],) = self.axes[0, 1].plot(
            [], [], "r-", label="LogVal Loss", linewidth=0.5
        )
        self.axes[0, 1].set_title("Log Loss Curves")
        self.axes[0, 1].set_xlabel("Epoch")
        self.axes[0, 1].set_ylabel("Loss")
        self.axes[0, 1].set_yscale("log")
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)

        # Learning rate plot
        (self.loss_lines["lr"],) = self.axes[1, 1].plot([], [], "g-", linewidth=0.5)
        self.axes[1, 1].set_title("Learning Rate")
        self.axes[1, 1].set_xlabel("Epoch")
        self.axes[1, 1].set_ylabel("Learning Rate")
        self.axes[1, 1].set_yscale("log")
        self.axes[1, 1].grid(True, alpha=0.3)

        # Teacher forcing ratio plot (if using scheduled sampling)
        (self.loss_lines["step_losses"],) = self.axes[1, 0].plot(
            [], [], "b-", linewidth=0.5
        )
        (self.loss_lines["step_losses_traj"],) = self.axes[1, 0].plot(
            [], [], "c-", linewidth=0.5
        )
        (self.loss_lines["step_losses_physics"],) = self.axes[1, 0].plot(
            [], [], "g-", linewidth=0.5
        )
        self.axes[1, 0].set_title("Train step losses")
        self.axes[1, 0].set_xlabel("Step")
        self.axes[1, 0].set_ylabel("Loss")
        self.axes[1, 0].grid(True, alpha=0.3)

        plt.tight_layout()

        if not self.is_notebook:
            plt.show(block=False)
            plt.pause(0.1)

    def _update_realtime_plot(self, epoch: int):
        """
        Update the real-time loss plot with new data.
        """
        if not self.plot_realtime or self.fig is None:
            return

        # Only update every N epochs to reduce overhead
        if epoch % self.plot_update_interval != 0 and epoch != self.epochs - 1:
            return

        epochs = list(range(len(self.training_history["train_loss"])))
        steps = list(range(len(self.training_history["step_losses"])))

        # Update loss curves
        if self.training_history["train_loss"]:
            self.loss_lines["train_loss"].set_data(
                epochs, self.training_history["train_loss"]
            )
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()

        if self.training_history["loss_components"]:
            if "trajectory" in self.training_history["loss_components"]:
                self.loss_lines["train_loss_traj"].set_data(
                    epochs, self.training_history["loss_components"]["trajectory"]
                )
                self.axes[0, 0].relim()
                self.axes[0, 0].autoscale_view()
            if "physics" in self.training_history["loss_components"]:
                self.loss_lines["train_loss_physics"].set_data(
                    epochs, self.training_history["loss_components"]["physics"]
                )
                self.axes[0, 0].relim()
                self.axes[0, 0].autoscale_view()

        if self.training_history["val_loss"]:
            self.loss_lines["val_loss"].set_data(
                epochs, self.training_history["val_loss"]
            )
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()

        # Update log loss curves
        if self.training_history["train_loss"]:
            self.loss_lines["log_train_loss"].set_data(
                epochs, (self.training_history["train_loss"])
            )
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()

        if self.training_history["loss_components"]:
            if "trajectory" in self.training_history["loss_components"]:
                self.loss_lines["log_train_loss_traj"].set_data(
                    epochs, self.training_history["loss_components"]["trajectory"]
                )
                self.axes[0, 1].relim()
                self.axes[0, 1].autoscale_view()
            if "physics" in self.training_history["loss_components"]:
                self.loss_lines["log_train_loss_physics"].set_data(
                    epochs, self.training_history["loss_components"]["physics"]
                )
                self.axes[0, 1].relim()
                self.axes[0, 1].autoscale_view()

        if self.training_history["val_loss"]:
            self.loss_lines["log_val_loss"].set_data(
                epochs, (self.training_history["val_loss"])
            )
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()

        # Update learning rate
        if self.training_history["learning_rates"]:
            self.loss_lines["lr"].set_data(
                epochs, self.training_history["learning_rates"]
            )
            self.axes[1, 1].relim()
            self.axes[1, 1].autoscale_view()

        # Update step losses plot
        if self.training_history["step_losses"]:
            self.loss_lines["step_losses"].set_data(
                steps, self.training_history["step_losses"]
            )
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()

        if self.training_history["step_loss_components"]:
            if "trajectory" in self.training_history["step_loss_components"]:
                self.loss_lines["step_losses_traj"].set_data(
                    steps, self.training_history["step_loss_components"]["trajectory"]
                )
                self.axes[1, 0].relim()
                self.axes[1, 0].autoscale_view()
            if "physics" in self.training_history["step_loss_components"]:
                self.loss_lines["step_losses_physics"].set_data(
                    steps, self.training_history["step_loss_components"]["physics"]
                )
                self.axes[0, 0].relim()
                self.axes[0, 0].autoscale_view()

        # Update loss difference
        # if len(self.training_history['train_loss']) > 0 and len(self.training_history['val_loss']) > 0:
        #     loss_diff = [t - v for t, v in zip(self.training_history['train_loss'],
        #                                       self.training_history['val_loss'])]
        #     self.loss_lines['loss_diff'].set_data(epochs, loss_diff)
        #     self.axes[1, 1].relim()
        #     self.axes[1, 1].autoscale_view()

        # Add best model marker
        if self.best_val_loss < float("inf"):
            best_epoch = self.training_history["val_loss"].index(
                min(self.training_history["val_loss"])
            )
            # Remove old markers
            for ax in self.axes.flat:
                for artist in ax.collections[:]:
                    if hasattr(artist, "_is_best_marker"):
                        artist.remove()
            # Add new marker on loss plot
            marker = self.axes[0, 0].scatter(
                best_epoch,
                self.best_val_loss,
                color="green",
                s=100,
                marker="*",
                zorder=5,
                label=f"Best (epoch {best_epoch})",
            )
            marker._is_best_marker = True

            # Update legend to show best loss
            handles, labels = self.axes[0, 0].get_legend_handles_labels()
            # Filter out old best markers from legend
            new_handles = []
            new_labels = []
            for h, l in zip(handles, labels):
                if not l.startswith("Best"):
                    new_handles.append(h)
                    new_labels.append(l)
            new_handles.append(marker)
            new_labels.append(f"Best (epoch {best_epoch})")
            self.axes[0, 0].legend(new_handles, new_labels)

        # Update title with current epoch info
        self.fig.suptitle(
            f"Training Progress - Epoch {epoch+1}/{self.epochs}",
            fontsize=14,
            fontweight="bold",
        )

        # Refresh plot
        if self.is_notebook:
            # For Jupyter notebooks
            clear_output(wait=True)
            display(self.fig)
        else:
            # For regular Python scripts
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

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
        elif scheduler_type == "steplr":
            return StepLR(
                self.optimizer,
                step_size=self.config.get("scheduler_patience", 10),
                gamma=self.config.get("scheduler_factor", 0.5),
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
        total_batches = 0
        total_loss_components = defaultdict(float)
        step_losses = []
        step_loss_components = defaultdict(list)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} (TF: {self.current_teacher_forcing_ratio:.3f}, LR: {self.optimizer.param_groups[0]['lr']:.6f})",
        )

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

            # Forward pass with scheduled sampling if enabled
            if self.use_scheduled_sampling and hasattr(
                self.model, "forward_with_scheduled_sampling"
            ):
                outputs = self.model.forward_with_scheduled_sampling(
                    batch, teacher_forcing_ratio=self.current_teacher_forcing_ratio
                )
                # forward_with_scheduled_sampling also has shift: each prediction is for the next step
                # So we need to exclude the last prediction which goes beyond targets
            else:
                # Standard forward pass
                outputs = self.model(batch)
            predictions = outputs["trajectory"][:, seq_len - 1 : -1, :]

            targets = batch["targets"]

            # Compute loss
            loss, losses = self.criterion(
                predictions,
                targets,
                self.const_parameters
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
            total_loss += loss.item()
            step_losses.append(loss.item())
            total_batches += 1
            total_samples += batch_size

            for name, val in losses.items():
                total_loss_components[name] += val.item()
                step_loss_components[name].append(val.item())

            # Update training metrics
            # if "trajectory" in outputs:
            self.train_metrics.update(
                predictions=predictions,
                targets=targets,
            )

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/total_batches:.4f}",
                    "TF_ratio": f"{self.current_teacher_forcing_ratio:.3f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # Log batch metrics
            # if batch_idx % self.log_interval == 0:
            #     logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}, TF_ratio={self.current_teacher_forcing_ratio:.3f}")

        # Compute epoch metrics
        avg_loss = total_loss / total_batches
        avg_loss_components = {k: v / total_samples for k, v in total_loss_components.items()}

        # Compute detailed metrics
        detailed_metrics = self.train_metrics.compute_all_metrics()

        epoch_metrics = {
            "step_losses": step_losses,
            "loss": avg_loss,
            "teacher_forcing_ratio": self.current_teacher_forcing_ratio,
            "step_loss_components": step_loss_components,
            "loss_components": avg_loss_components,
            **detailed_metrics,
        }

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        """
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        total_samples = 0
        total_batches = 0
        total_loss_components = defaultdict(float)
        step_losses = []
        step_loss_components = defaultdict(list)

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

                predictions = outputs["trajectory"][:, seq_len - 1 : -1, :]
                targets = batch["targets"]

                # Compute loss
                loss, losses = self.criterion(
                    predictions,
                    targets,
                    self.const_parameters
                )

                # Update metrics
                total_loss += loss.item()
                total_batches += 1
                total_samples += batch_size

                for name, val in losses.items():
                    total_loss_components[name] += val.item()
                    step_loss_components[name].append(val.item())

                # Update validation metrics
                self.val_metrics.update(
                    predictions=predictions,
                    targets=targets,
                )

        # Compute epoch metrics
        avg_loss = total_loss / total_batches
        avg_loss_components = {k: v / total_samples for k, v in total_loss_components.items()}
        # avg_loss_components = {k: v / total_samples for k, v in loss_components.items()}

        # Compute detailed metrics
        detailed_metrics = self.val_metrics.compute_all_metrics()

        epoch_metrics = {"loss": avg_loss, "loss_components": avg_loss_components, **detailed_metrics}

        return epoch_metrics

    def train(self) -> Dict:
        """
        Main training loop.
        """
        logger.info(f"Starting training for {self.epochs} epochs")

        start_time = time.time()
        epochs_without_improvement = 0
        if self.pretrain_val:
            val_metrics = self.validate_epoch()
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, float):
                    print(f"{metric_name}: {metric_value:.4f}")
                elif isinstance(metric_value, dict):
                    for submetric_name, submetric_value in metric_value.items():
                        print(f"{metric_name}.{submetric_name}: {submetric_value:.4f}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate epoch
            val_metrics = self.validate_epoch()

            # Update teacher forcing ratio for scheduled sampling
            if self.use_scheduled_sampling:
                self.current_teacher_forcing_ratio = max(
                    self.min_teacher_forcing_ratio,
                    self.current_teacher_forcing_ratio * self.teacher_forcing_decay,
                )

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Update metrics tracker
            self.metrics_tracker.update(epoch, train_metrics, val_metrics)

            # Update training history
            self.training_history["step_losses"].extend(train_metrics["step_losses"])
            self.training_history["train_loss"].append(train_metrics["loss"])
            for name, val in train_metrics["loss_components"].items():
                self.training_history[f"loss_components"][name].append(val)

            for name, val in train_metrics["step_loss_components"].items():
                self.training_history[f"step_loss_components"][name].extend(val)
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["learning_rates"].append(
                self.optimizer.param_groups[0]["lr"]
            )
            if self.use_scheduled_sampling:
                self.training_history["teacher_forcing_ratios"].append(
                    self.current_teacher_forcing_ratio
                )

            # Update real-time plot
            self._update_realtime_plot(epoch)

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
            log_msg = (
                f"Epoch {epoch:3d}: "
                f"Train Loss: {train_metrics['loss']:.6f}, "
                f"Val Loss: {val_metrics['loss']:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            if self.use_scheduled_sampling:
                log_msg += f", TF Ratio: {self.current_teacher_forcing_ratio:.3f}"
            print(log_msg)

            # Print detailed metrics periodically
            if epoch % (self.log_interval) == 0:
                print(f"\nEpoch {epoch} Detailed Metrics:")
                print("-" * 40)

                # Print key metrics
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, float):
                        print(f"{metric_name:20s}: Train: {train_metrics[metric_name]:.6f}, Val: {metric_value:.6f}")
                    elif isinstance(metric_value, dict):
                        for submetric_name, submetric_value in metric_value.items():
                            print(f"{metric_name}.{submetric_name:20s}: Train: {train_metrics[metric_name][submetric_name]:.6f}, Val: {submetric_value:.6f}")

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

        # Close real-time plot if it was open
        if self.plot_realtime and self.fig is not None and not self.is_notebook:
            plt.close(self.fig)

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
