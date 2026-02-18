"""
Loss functions for vibration prediction models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TrajectoryLoss(nn.Module):
    """
    Loss for trajectory prediction accuracy.
    """

    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute trajectory loss.

        Args:
            predictions: Predicted trajectories [batch_size, seq_len, features]
            targets: Target trajectories [batch_size, seq_len, features]
            mask: Optional mask for valid timesteps

        Returns:
            Loss value
        """
        loss = self.criterion(predictions, targets)

        if mask is not None:
            # Apply mask to loss
            if self.reduction == 'none':
                loss = loss * mask.unsqueeze(-1)
                loss = loss.sum() / mask.sum()
            else:
                # For mean/sum reduction, we need to recompute
                diff = predictions - targets
                if self.loss_type == 'mse':
                    loss = (diff ** 2) * mask.unsqueeze(-1)
                elif self.loss_type == 'mae':
                    loss = torch.abs(diff) * mask.unsqueeze(-1)

                loss = loss.sum() / mask.sum()

        return loss


class AmplitudeLoss(nn.Module):
    """
    Loss for amplitude prediction with emphasis on growth detection.
    """

    def __init__(self):
        super().__init__()
        # self.growth_penalty = growth_penalty

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        std_pred = torch.std(predictions, dim=1)
        std_target = torch.std(targets, dim=1)

        # Штрафуем разницу в энергиях
        return torch.mean((std_pred - std_target) ** 2)

    # def forward(
    #     self,
    #     pred_amplitude: torch.Tensor,
    #     target_amplitude: torch.Tensor,
    #     pred_trajectory: Optional[torch.Tensor] = None,
    #     target_trajectory: Optional[torch.Tensor] = None
    # ) -> torch.Tensor:
    #     """
    #     Compute amplitude loss with growth penalty.

    #     Args:
    #         pred_amplitude: Predicted max amplitude [batch_size, 1]
    #         target_amplitude: Target max amplitude [batch_size, 1]
    #         pred_trajectory: Predicted trajectory (optional)
    #         target_trajectory: Target trajectory (optional)

    #     Returns:
    #         Loss value
    #     """
    #     # Basic amplitude MSE
    #     amplitude_loss = F.mse_loss(pred_amplitude, target_amplitude)

    #     # Penalty for underestimating amplitude growth
    #     underestimation_mask = pred_amplitude < target_amplitude
    #     underestimation_penalty = F.mse_loss(
    #         pred_amplitude[underestimation_mask],
    #         target_amplitude[underestimation_mask]
    #     ) * self.growth_penalty if underestimation_mask.any() else 0

    #     total_loss = amplitude_loss + underestimation_penalty

    #     # If trajectories are provided, compute trajectory-based amplitude
    #     if pred_trajectory is not None and target_trajectory is not None:
    #         pred_traj_amplitude = torch.max(torch.abs(pred_trajectory), dim=1)[0].max(dim=1)[0]
    #         target_traj_amplitude = torch.max(torch.abs(target_trajectory), dim=1)[0].max(dim=1)[0]

    #         traj_amplitude_loss = F.mse_loss(pred_traj_amplitude, target_traj_amplitude)
    #         total_loss += traj_amplitude_loss

    #     return total_loss


class StabilityLoss(nn.Module):
    """
    Loss to encourage stable predictions.
    """

    def __init__(self, stability_weight: float = 1.0):
        super().__init__()
        self.stability_weight = stability_weight

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute stability loss based on prediction smoothness.

        Args:
            predictions: Predicted trajectories [batch_size, seq_len, features]

        Returns:
            Stability loss
        """
        # Compute second derivatives (acceleration)
        if predictions.shape[1] < 3:
            return torch.tensor(0.0, device=predictions.device)

        # First derivative (velocity)
        first_diff = predictions[:, 1:, :] - predictions[:, :-1, :]

        # Second derivative (acceleration)
        second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]

        # Penalize large accelerations
        stability_loss = torch.mean(second_diff ** 2) * self.stability_weight

        return stability_loss


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss based on system dynamics.
    """

    def __init__(self, physics_weight: float = 1.0):
        super().__init__()
        self.physics_weight = physics_weight

    def forward(
        self,
        predictions: torch.Tensor,
        parameters: torch.Tensor,
        dt: float = 0.01
    ) -> torch.Tensor:
        """
        Compute physics-based loss.

        Args:
            predictions: Predicted trajectories [batch_size, seq_len, 2] (x, x_dot)
            parameters: System parameters [batch_size, n_params]
            dt: Time step

        Returns:
            Physics loss
        """
        if predictions.shape[1] < 2 or predictions.shape[2] < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Extract position and velocity
        x = predictions[:, :, 0]  # [batch_size, seq_len]
        x_dot = predictions[:, 1:-1, 1]  # [batch_size, seq_len]

        # # Compute numerical derivatives
        # x_dot_numerical = torch.zeros_like(x_dot)
        # x_dot_numerical[:, 1:] = (x[:, 1:] - x[:, :-1]) / dt
        # x_dot_numerical[:, 0] = x_dot_numerical[:, 1]  # Forward fill

        # Central difference calculation (fixed indexing)
        x_dot_numerical = (x[:, 2:] - x[:, :-2]) / (2 * dt)

        # Consistency loss: predicted velocity should match numerical derivative
        velocity_consistency = F.mse_loss(x_dot, x_dot_numerical)

        physics_loss = velocity_consistency

        return physics_loss

        # Energy conservation (approximate)
        # if parameters.shape[1] >= 3:  # Assuming k, d, e are first 3 parameters
        #     k = parameters[:, 0].unsqueeze(1)  # Damping
        #     d = parameters[:, 1].unsqueeze(1)  # Stiffness

        #     # Kinetic energy
        #     kinetic_energy = 0.5 * x_dot ** 2

        #     # Potential energy (harmonic oscillator approximation)
        #     potential_energy = 0.5 * d * x ** 2

        #     # Total energy
        #     total_energy = kinetic_energy + potential_energy

        #     # Energy should be relatively conserved (penalize large changes)
        #     energy_changes = torch.abs(total_energy[:, 1:] - total_energy[:, :-1])
        #     energy_conservation = torch.mean(energy_changes)
        # else:
        #     energy_conservation = torch.tensor(0.0, device=predictions.device)

        # physics_loss = (velocity_consistency + energy_conservation) * self.physics_weight

        # return physics_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    """

    def __init__(self, weights: Dict[str, float]):
        """
        Initialize combined loss.

        Args:
            weights: Dictionary of loss component weights
        """
        super().__init__()

        self.weights = weights

        # Initialize loss components
        self.trajectory_loss = TrajectoryLoss(loss_type='mse')
        self.amplitude_loss = AmplitudeLoss()
        self.stability_loss = StabilityLoss()
        self.physics_loss = PhysicsLoss()

        logger.info(f"Combined loss initialized with weights: {weights}")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        parameters: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
            parameters: System parameters (optional)

        Returns:
            Total loss and individual loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions.device)

        # Trajectory loss
        if 'trajectory' in self.weights and self.weights['trajectory'] > 0:
            traj_loss = self.trajectory_loss(predictions, targets)
            losses['trajectory'] = self.weights['trajectory'] * traj_loss
            total_loss += self.weights['trajectory'] * traj_loss

        if 'amplitude' in self.weights and self.weights['amplitude'] > 0:
            traj_loss = self.amplitude_loss(predictions, targets)
            losses['amplitude'] = self.weights['amplitude'] * traj_loss
            total_loss += self.weights['amplitude'] * traj_loss

        # Physics loss
        if 'physics' in self.weights and self.weights['physics'] > 0:
            phys_loss = self.physics_loss(predictions, targets, parameters["dt"])
            losses['physics'] = self.weights['physics'] * phys_loss
            total_loss += self.weights['physics'] * phys_loss

        # Amplitude loss
        # if 'amplitude' in self.weights and self.weights['amplitude'] > 0:
        #     if 'amplitude' in outputs and 'max_amplitude' in targets:
        #         amp_loss = self.amplitude_loss(
        #             outputs['amplitude'],
        #             targets['max_amplitude'],
        #             outputs.get('trajectory'),
        #             targets.get('targets')
        #         )
        #         losses['amplitude'] = amp_loss
        #         total_loss += self.weights['amplitude'] * amp_loss

        # Stability loss
        # if 'stability' in self.weights and self.weights['stability'] > 0:
        #     if 'trajectory' in outputs:
        #         stab_loss = self.stability_loss(outputs['trajectory'])
        #         losses['stability'] = stab_loss
        #         total_loss += self.weights['stability'] * stab_loss



        return total_loss, losses

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update loss weights during training.
        """
        self.weights.update(new_weights)
        logger.info(f"Loss weights updated: {self.weights}")


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in bifurcation detection.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Predicted probabilities [batch_size]
            targets: Binary targets [batch_size]

        Returns:
            Focal loss
        """
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning similar/dissimilar trajectory representations.
    """

    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            labels: Similarity labels [batch_size, batch_size]

        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive and negative masks
        positive_mask = labels.bool()
        negative_mask = ~positive_mask

        # Positive pairs
        positive_loss = -torch.log(torch.exp(similarity_matrix[positive_mask]).sum() + 1e-8)

        # Negative pairs
        negative_loss = torch.log(torch.exp(similarity_matrix[negative_mask]).sum() + 1e-8)

        return positive_loss + negative_loss


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress.
    """

    def __init__(self, base_loss: nn.Module, adaptation_rate: float = 0.01):
        super().__init__()
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        self.loss_history = []
        self.adaptive_weights = {}

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute adaptive loss.
        """
        loss, individual_losses = self.base_loss(*args, **kwargs)

        # Store loss history
        self.loss_history.append({k: v.item() for k, v in individual_losses.items()})

        # Adapt weights based on recent performance
        if len(self.loss_history) > 10:
            self._adapt_weights()

        return loss, individual_losses

    def _adapt_weights(self):
        """
        Adapt loss weights based on recent performance.
        """
        recent_losses = self.loss_history[-10:]

        for loss_name in recent_losses[0].keys():
            recent_values = [loss[loss_name] for loss in recent_losses]

            # If loss is not decreasing, increase its weight
            if len(recent_values) >= 5:
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                if trend > 0:  # Increasing trend
                    if loss_name in self.base_loss.weights:
                        self.base_loss.weights[loss_name] *= (1 + self.adaptation_rate)
                else:  # Decreasing trend
                    if loss_name in self.base_loss.weights:
                        self.base_loss.weights[loss_name] *= (1 - self.adaptation_rate)
                        self.base_loss.weights[loss_name] = max(0.01, self.base_loss.weights[loss_name])


class UncertaintyLoss(nn.Module):
    """
    Loss for uncertainty estimation in predictions.
    """

    def __init__(self, uncertainty_weight: float = 1.0):
        super().__init__()
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            uncertainties: Predicted uncertainties (log variance)

        Returns:
            Uncertainty loss
        """
        # Convert log variance to variance
        variances = torch.exp(uncertainties)

        # Negative log likelihood assuming Gaussian distribution
        nll = 0.5 * (torch.log(2 * np.pi * variances) + (predictions - targets) ** 2 / variances)

        return nll.mean() * self.uncertainty_weight
