"""
Unit tests for model components.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.lstm_predictor import LSTMPredictor
from models.base_model import ModelFactory


class TestLSTMPredictor(unittest.TestCase):
    """Test LSTM predictor model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'type': 'lstm',
            'input_size': 2,
            'output_size': 2,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'recurrent_dropout': 0.1,
            'n_params': 15,
            'parameter_embedding_dim': 16
        }
        self.model = LSTMPredictor(self.config)

        # Test data
        self.batch_size = 4
        self.seq_len = 100
        self.features = torch.randn(self.batch_size, self.seq_len, 2)
        self.parameters = torch.randn(self.batch_size, 15)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, LSTMPredictor)
        self.assertEqual(self.model.input_size, 2)
        self.assertEqual(self.model.output_size, 2)
        self.assertEqual(self.model.hidden_size, 64)
        self.assertEqual(self.model.num_layers, 2)

    def test_forward_pass(self):
        """Test forward pass."""
        outputs = self.model(self.features, self.parameters)

        # Check output keys
        expected_keys = ['next_step', 'trajectory', 'amplitude', 'attention_weights']
        for key in expected_keys:
            self.assertIn(key, outputs)

        # Check output shapes
        self.assertEqual(outputs['next_step'].shape, (self.batch_size, 2))
        self.assertEqual(outputs['trajectory'].shape, (self.batch_size, 2))
        self.assertEqual(outputs['amplitude'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['attention_weights'].shape, (self.batch_size, self.seq_len))

    def test_trajectory_prediction(self):
        """Test trajectory prediction."""
        horizon = 50
        trajectory = self.model.predict_trajectory(
            self.features, self.parameters, horizon
        )

        expected_shape = (self.batch_size, horizon, 2)
        self.assertEqual(trajectory.shape, expected_shape)

    def test_uncertainty_prediction(self):
        """Test uncertainty prediction."""
        horizon = 20
        n_samples = 5

        mean_pred, std_pred = self.model.predict_with_uncertainty(
            self.features, self.parameters, horizon, n_samples
        )

        expected_shape = (self.batch_size, horizon, 2)
        self.assertEqual(mean_pred.shape, expected_shape)
        self.assertEqual(std_pred.shape, expected_shape)

        # Standard deviation should be non-negative
        self.assertTrue(torch.all(std_pred >= 0))

    def test_attention_weights(self):
        """Test attention weights extraction."""
        attention_weights = self.model.get_attention_weights(
            self.features, self.parameters
        )

        expected_shape = (self.batch_size, self.seq_len)
        self.assertEqual(attention_weights.shape, expected_shape)

        # Attention weights should sum to 1 (approximately)
        attention_sums = torch.sum(attention_weights, dim=1)
        torch.testing.assert_close(attention_sums, torch.ones(self.batch_size), atol=1e-5)

    def test_model_info(self):
        """Test model info extraction."""
        info = self.model.get_model_info()

        self.assertIn('model_type', info)
        self.assertIn('total_parameters', info)
        self.assertIn('trainable_parameters', info)
        self.assertIsInstance(info['total_parameters'], int)
        self.assertIsInstance(info['trainable_parameters'], int)
        self.assertGreater(info['total_parameters'], 0)


class TestModelFactory(unittest.TestCase):
    """Test model factory."""

    def test_model_registration(self):
        """Test model registration."""
        available_models = ModelFactory.list_models()
        self.assertIn('lstm', available_models)

    def test_model_creation(self):
        """Test model creation."""
        config = {
            'type': 'lstm',
            'input_size': 2,
            'output_size': 2,
            'hidden_size': 32,
            'num_layers': 1,
            'n_params': 10
        }

        model = ModelFactory.create_model(config)
        self.assertIsInstance(model, LSTMPredictor)

    def test_invalid_model_type(self):
        """Test invalid model type."""
        config = {'type': 'invalid_model'}

        with self.assertRaises(ValueError):
            ModelFactory.create_model(config)


if __name__ == '__main__':
    unittest.main()
