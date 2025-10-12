"""
Base model class for vibration prediction models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BaseVibrationModel(nn.Module, ABC):
    """
    Abstract base class for vibration prediction models.
    """

    def __init__(self, config: Dict):
        """
        Initialize base model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.model_type = config.get('type', 'base')

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        parameters: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            features: Input features [batch_size, seq_len, n_features]
            parameters: System parameters [batch_size, n_params]

        Returns:
            Dictionary containing model outputs
        """
        pass

    @abstractmethod
    def predict_trajectory(
        self,
        features: torch.Tensor,
        parameters: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """
        Predict future trajectory.

        Args:
            features: Input features
            parameters: System parameters
            horizon: Prediction horizon

        Returns:
            Predicted trajectory [batch_size, horizon, n_features]
        """
        pass

    def get_model_info(self) -> Dict:
        """
        Get model information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config
        }

    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_info': self.get_model_info()
        }

        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu'):
        """
        Load model from checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Create model instance
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded from {filepath}")
        return model, checkpoint

    def freeze_layers(self, layer_names: list):
        """
        Freeze specified layers.
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")

    def unfreeze_layers(self, layer_names: list):
        """
        Unfreeze specified layers.
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")


class ModelFactory:
    """
    Factory class for creating models.
    """

    _models = {}

    @classmethod
    def register_model(cls, model_type: str, model_class):
        """
        Register a model class.
        """
        cls._models[model_type] = model_class

    @classmethod
    def create_model(cls, config: Dict) -> BaseVibrationModel:
        """
        Create model instance based on configuration.
        """
        model_type = config.get('type', 'lstm')

        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls._models[model_type](config)

    @classmethod
    def list_models(cls) -> list:
        """
        List available model types.
        """
        return list(cls._models.keys())
