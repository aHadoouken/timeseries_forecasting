"""
Unit tests for loss functions.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.losses import (
    TrajectoryLoss, AmplitudeLoss, StabilityLoss,
    PhysicsLoss, CombinedLoss
)


class TestTrajectoryLoss(unittest.TestCase):
    """Test trajectory loss function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 50
        self.features = 2

        self.predictions = torch.randn(self.batch_size, self.seq_len, self.features)
        self.targets = torch.randn(self.batch_size, self.seq_len, self.features)

    def test_mse_loss(self):
        """Test MSE trajectory loss."""
        loss_fn = TrajectoryLoss(loss_type='mse')
        loss = loss_fn(self.predictions, self.targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(loss.item(), 0)  # Non-negative

    def test_mae_loss(self):
        """Test MAE trajectory loss."""
        loss_fn = TrajectoryLoss(loss_type='mae')
        loss = loss_fn(self.predictions, self.targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

    def test_huber_loss(self):
        """Test Huber trajectory loss."""
        loss_fn = TrajectoryLoss(loss_type='huber')
        loss = loss_fn(self.predictions, self.targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

    def test_masked_loss(self):
        """Test loss with mask."""
        loss_fn = TrajectoryLoss(loss_type='mse')
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[:, -10:] = 0  # Mask last 10 timesteps

        loss = loss_fn(self.predictions, self.targets, mask)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)


class TestAmplitudeLoss(unittest.TestCase):
    """Test amplitude loss function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.pred_amplitude = torch.rand(self.batch_size, 1) * 5
        self.target_amplitude = torch.rand(self.batch_size, 1) * 5

    def test_basic_amplitude_loss(self):
        """Test basic amplitude loss."""
        loss_fn = AmplitudeLoss()
        loss = loss_fn(self.pred_amplitude, self.target_amplitude)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

    def test_underestimation_penalty(self):
        """Test underestimation penalty."""
        loss_fn = AmplitudeLoss(growth_penalty=3.0)

        # Create case where prediction underestimates target
        pred_low = torch.ones(2, 1) * 1.0
        target_high = torch.ones(2, 1) * 5.0

        loss_with_penalty = loss_fn(pred_low, target_high)

        # Compare with case where prediction matches target
        pred_match = target_high.clone()
        loss_without_penalty = loss_fn(pred_match, target_high)

        self.assertGreater(loss_with_penalty.item(), loss_without_penalty.item())

    def test_trajectory_based_amplitude(self):
        """Test trajectory-based amplitude loss."""
        loss_fn = AmplitudeLoss()

        seq_len = 50
        pred_traj = torch.randn(self.batch_size, seq_len, 2)
        target_traj = torch.randn(self.batch_size, seq_len, 2)

        loss = loss_fn(
            self.pred_amplitude,
            self.target_amplitude,
            pred_traj,
            target_traj
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)


class TestStabilityLoss(unittest.TestCase):
    """Test stability loss function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 50
        self.features = 2

    def test_stability_loss(self):
        """Test stability loss computation."""
        loss_fn = StabilityLoss()

        # Create smooth predictions
        t = torch.linspace(0, 10, self.seq_len).unsqueeze(0).unsqueeze(-1)
        smooth_predictions = torch.sin(t).expand(self.batch_size, -1, self.features)

        # Create noisy predictions
        noisy_predictions = smooth_predictions + torch.randn_like(smooth_predictions) * 0.5

        smooth_loss = loss_fn(smooth_predictions)
        noisy_loss = loss_fn(noisy_predictions)

        # Noisy predictions should have higher stability loss
        self.assertGreater(noisy_loss.item(), smooth_loss.item())

    def test_short_sequence(self):
        """Test stability loss with short sequence."""
        loss_fn = StabilityLoss()
        short_predictions = torch.randn(self.batch_size, 2, self.features)

        loss = loss_fn(short_predictions)
        self.assertEqual(loss.item(), 0.0)  # Should return 0 for short sequences


class TestPhysicsLoss(unittest.TestCase):
    """Test physics-informed loss function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 50

        # Create physically consistent trajectory
        t = torch.linspace(0, 5, self.seq_len)
        x = torch.sin(t).unsqueeze(0).unsqueeze(-1).expand(self.batch_size, -1, 1)
        x_dot = torch.cos(t).unsqueeze(0).unsqueeze(-1).expand(self.batch_size, -1, 1)

        self.predictions = torch.cat([x, x_dot], dim=-1)
        self.parameters = torch.randn(self.batch_size, 15)

    def test_physics_loss(self):
        """Test physics loss computation."""
        loss_fn = PhysicsLoss()
        loss = loss_fn(self.predictions, self.parameters)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

    def test_short_sequence(self):
        """Test physics loss with short sequence."""
        loss_fn = PhysicsLoss()
        short_predictions = torch.randn(self.batch_size, 1, 2)

        loss = loss_fn(short_predictions, self.parameters)
        self.assertEqual(loss.item(), 0.0)


class TestCombinedLoss(unittest.TestCase):
    """Test combined loss function."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = {
            'trajectory': 1.0,
            'amplitude': 2.0,
            'stability': 0.5,
            'physics': 0.1
        }
        self.loss_fn = CombinedLoss(self.weights)

        self.batch_size = 4
        self.seq_len = 50

        # Mock outputs
        self.outputs = {
            'next_step': torch.randn(self.batch_size, 2),
            'trajectory': torch.randn(self.batch_size, self.seq_len, 2),
            'amplitude': torch.rand(self.batch_size, 1)
        }

        # Mock targets
        self.targets = {
            'targets': torch.randn(self.batch_size, self.seq_len, 2),
            'max_amplitude': torch.rand(self.batch_size, 1)
        }

        self.parameters = torch.randn(self.batch_size, 15)

    def test_combined_loss_computation(self):
        """Test combined loss computation."""
        total_loss, individual_losses = self.loss_fn(
            self.outputs, self.targets, self.parameters
        )

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.dim(), 0)
        self.assertGreaterEqual(total_loss.item(), 0)

        self.assertIsInstance(individual_losses, dict)
        self.assertGreater(len(individual_losses), 0)

    def test_weight_update(self):
        """Test weight update functionality."""
        new_weights = {'trajectory': 2.0, 'amplitude': 1.0}
        self.loss_fn.update_weights(new_weights)

        self.assertEqual(self.loss_fn.weights['trajectory'], 2.0)
        self.assertEqual(self.loss_fn.weights['amplitude'], 1.0)
        self.assertEqual(self.loss_fn.weights['stability'], 0.5)  # Unchanged

    def test_missing_outputs(self):
        """Test behavior with missing outputs."""
        incomplete_outputs = {'next_step': torch.randn(self.batch_size, 2)}

        total_loss, individual_losses = self.loss_fn(
            incomplete_outputs, self.targets, self.parameters
        )

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertGreaterEqual(total_loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
