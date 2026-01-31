"""
LSTM-based predictor for vibration time series.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
import logging
import numpy as np

from .base_model import BaseVibrationModel, ModelFactory

logger = logging.getLogger(__name__)


class ParameterEmbedding(nn.Module):
    """
    Embedding layer for system parameters.
    """

    def __init__(self, n_params: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Linear(n_params, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parameter embedding.

        Args:
            parameters: [batch_size, n_params]

        Returns:
            Embedded parameters [batch_size, embedding_dim]
        """
        embedded = self.embedding(parameters)
        embedded = self.layer_norm(embedded)
        embedded = F.relu(embedded)
        embedded = self.dropout(embedded)
        return embedded


class AttentionMechanism(nn.Module):
    """
    Attention mechanism for focusing on important time steps.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM output.

        Args:
            lstm_output: [batch_size, seq_len, hidden_size]

        Returns:
            Attended output [batch_size, hidden_size] and attention weights
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(
            attention_scores, dim=1
        )  # [batch_size, seq_len, 1]

        # Apply attention
        attended_output = torch.sum(
            lstm_output * attention_weights, dim=1
        )  # [batch_size, hidden_size]

        return attended_output, attention_weights.squeeze(-1)


class LSTMPredictor(BaseVibrationModel):
    """
    LSTM-based vibration predictor with parameter conditioning.
    """

    def __init__(self, config: Dict):
        """
        Initialize LSTM predictor.

        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)

        # Model parameters
        self.input_size = config.get("input_size", 2)  # x, x_dot
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.recurrent_dropout = config.get("recurrent_dropout", 0.2)
        self.bidirectional = config.get("bidirectional", False)

        # Output configuration
        self.output_size = config.get("output_size", 2)  # x, x_dot
        self.output_layers = config.get("output_layers", [128, 64])
        self.output_activation = config.get("output_activation", "tanh")
        self.teacher_forcing_noise_std = config.get("teacher_forcing_noise_std", 0.0)

        # Build model components
        self._build_model()

        # Initialize weights
        # self._initialize_weights()

        logger.info(f"LSTM Predictor initialized: {self.get_model_info()}")

    def _build_model(self):
        """
        Build model architecture.
        """
        # Input projection (combine features with parameter embedding)
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.recurrent_dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        # Attention mechanism
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # Output layers
        output_layers = []
        prev_size = lstm_output_size

        for layer_size in self.output_layers:
            output_layers.extend(
                [
                    nn.Linear(prev_size, layer_size),
                    # nn.LayerNorm(layer_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            prev_size = layer_size

        # Final output layer
        output_layers.append(nn.Linear(prev_size, self.output_size))

        # Apply output activation
        if self.output_activation == "tanh":
            output_layers.append(nn.Tanh())
        elif self.output_activation == "sigmoid":
            output_layers.append(nn.Sigmoid())

        self.output_net = nn.Sequential(*output_layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the LSTM predictor.

        Args:
            features: Input features [batch_size, seq_len, input_size]
            parameters: System parameters [batch_size, n_params]

        Returns:
            Dictionary containing model outputs
        """
        features = batch["features"]
        targets = batch["targets"]
        batch_size, seq_len, _ = features.shape
        batch_size, horizon_len, _ = targets.shape

        x = torch.cat([features, targets], dim=1)

        if self.training:  # Только во время обучения
            std = torch.std(x, dim=1, keepdim=True)
            # print(std.shape, x.shape)
            noise = torch.randn_like(x) * self.teacher_forcing_noise_std * std
            x = x + noise

        # Project to hidden size
        projected_input = self.input_projection(x)  # [batch_size, seq_len, hidden_size]

        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(
            projected_input
        )  # [batch_size, seq_len, lstm_output_size]

        # Generate outputs
        trajectory_pred = self.output_net(
            lstm_output
        )  # [batch_size, seq_len, output_size]

        return {
            # 'next_step': next_step,
            "trajectory": trajectory_pred,
            # 'amplitude': amplitude_pred,
            # 'attention_weights': attention_weights,
            "hidden_state": hidden,
            "cell_state": cell,
            "lstm_output": lstm_output,
        }

    # def forward_with_scheduled_sampling(
    #     self, batch: Dict[str, torch.Tensor], teacher_forcing_ratio: float = 1.0
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Forward pass with scheduled sampling for training.

    #     Args:
    #         batch: Input batch containing features and targets
    #         teacher_forcing_ratio: Probability of using ground truth (1.0 = always use ground truth)

    #     Returns:
    #         Dictionary containing model outputs
    #     """
    #     features = batch["features"]
    #     targets = batch["targets"]
    #     batch_size, seq_len, _ = features.shape
    #     _, horizon_len, _ = targets.shape

    #     projected_input = self.input_projection(features)
    #     lstm_output, (hidden, cell) = self.lstm(projected_input)
    #     output = self.output_net(lstm_output)

    #     # Заранее генерим teacher forcing mask (1 = teacher, 0 = model)
    #     teacher_mask = (torch.rand(batch_size, horizon_len, 1, device=features.device) < teacher_forcing_ratio).float()

    #     preds = [output]
    #     current_input = targets[:, 0:1, :]  # начнем с teacher

    #     for t in range(horizon_len):
    #         projected = self.input_projection(current_input)
    #         lstm_out, (hidden, cell) = self.lstm(projected, (hidden, cell))
    #         pred = self.output_net(lstm_out)
    #         preds.append(pred)

    #         if t + 1 < horizon_len:
    #             # Смешиваем teacher и predicted значения
    #             current_input = teacher_mask[:, t:t+1, :] * targets[:, t:t+1, :] + \
    #                             (1 - teacher_mask[:, t:t+1, :]) * pred

    #     trajectory_pred = torch.cat(preds, dim=1)


    #     return {
    #         "trajectory": trajectory_pred,
    #         "hidden_state": hidden,
    #         "cell_state": cell,
    #         "teacher_forcing_ratio": teacher_forcing_ratio,
    #     }

    def forward_with_scheduled_sampling(
        self, batch: Dict[str, torch.Tensor], teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with scheduled sampling for training.

        Args:
            batch: Input batch containing features and targets
            teacher_forcing_ratio: Probability of using ground truth (1.0 = always use ground truth)

        Returns:
            Dictionary containing model outputs
        """
        features = batch["features"]
        targets = batch["targets"]
        batch_size, seq_len, _ = features.shape
        _, horizon_len, _ = targets.shape

        projected_input = self.input_projection(features)
        lstm_output, (hidden, cell) = self.lstm(projected_input)
        output = self.output_net(lstm_output)

        preds = [output]
        current_input = targets[:, 0:1, :]  # начнем с teacher

        for t in range(horizon_len):
            projected = self.input_projection(current_input)
            lstm_out, (hidden, cell) = self.lstm(projected, (hidden, cell))
            pred = self.output_net(lstm_out)
            preds.append(pred)

            if t + 1 < horizon_len:
                # Используем взвешенное среднее между target и prediction
                current_input = teacher_forcing_ratio * targets[:, t:t+1, :] + (1 - teacher_forcing_ratio) * pred

        trajectory_pred = torch.cat(preds, dim=1)


        return {
            "trajectory": trajectory_pred,
            "hidden_state": hidden,
            "cell_state": cell,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }

    def predict_trajectory(
        self, features: torch.Tensor, parameters: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        """
        Predict future trajectory using autoregressive generation.

        Args:
            features: Input features [batch_size, seq_len, input_size]
            parameters: System parameters [batch_size, n_params]
            horizon: Prediction horizon

        Returns:
            Predicted trajectory [batch_size, horizon, output_size]
        """
        batch_size = features.shape[0]
        device = features.device

        # Initialize prediction sequence
        predictions = []
        current_input = features

        # Embed parameters once
        param_embedded = self.param_embedding(parameters)

        for step in range(horizon):
            # Forward pass
            outputs = self.forward(current_input, parameters)
            next_step = outputs["next_step"]  # [batch_size, output_size]

            predictions.append(next_step.unsqueeze(1))  # [batch_size, 1, output_size]

            # Update input for next step (sliding window)
            if current_input.shape[1] > 1:
                # Remove first timestep and append prediction
                current_input = torch.cat(
                    [current_input[:, 1:, :], next_step.unsqueeze(1)], dim=1
                )
            else:
                # Replace single timestep
                current_input = next_step.unsqueeze(1)

        # Concatenate all predictions
        trajectory = torch.cat(predictions, dim=1)  # [batch_size, horizon, output_size]

        return trajectory

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        parameters: torch.Tensor,
        horizon: int,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory with uncertainty estimation using Monte Carlo dropout.

        Args:
            features: Input features
            parameters: System parameters
            horizon: Prediction horizon
            n_samples: Number of MC samples

        Returns:
            Mean predictions and standard deviations
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            pred = self.predict_trajectory(features, parameters, horizon)
            predictions.append(pred)

        predictions = torch.stack(
            predictions, dim=0
        )  # [n_samples, batch_size, horizon, output_size]

        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)

        self.eval()  # Disable dropout

        return mean_pred, std_pred

    def get_attention_weights(
        self, features: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        """
        outputs = self.forward(features, parameters)
        return outputs["attention_weights"]

    def compute_feature_importance(
        self, features: torch.Tensor, parameters: torch.Tensor, method: str = "gradient"
    ) -> torch.Tensor:
        """
        Compute feature importance using gradient-based methods.
        """
        features.requires_grad_(True)

        outputs = self.forward(features, parameters)
        loss = outputs["amplitude"].sum()  # Use amplitude as target

        loss.backward()

        if method == "gradient":
            importance = torch.abs(features.grad).mean(dim=0)  # [seq_len, input_size]
        elif method == "integrated_gradient":
            # Simplified integrated gradients
            baseline = torch.zeros_like(features)
            importance = torch.abs(features.grad * (features - baseline)).mean(dim=0)
        else:
            raise ValueError(f"Unknown importance method: {method}")

        return importance


# Register the model
ModelFactory.register_model("lstm", LSTMPredictor)


class MultiScaleLSTM(BaseVibrationModel):
    """
    Multi-scale LSTM for capturing different temporal patterns.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        self.scales = config.get("scales", [1, 2, 4])  # Different temporal scales
        self.base_config = config.copy()

        # Create LSTM for each scale
        self.scale_lstms = nn.ModuleList()
        for scale in self.scales:
            scale_config = self.base_config.copy()
            scale_config["hidden_size"] = config.get("hidden_size", 128) // len(
                self.scales
            )
            self.scale_lstms.append(LSTMPredictor(scale_config))

        # Fusion layer
        total_hidden = config.get("hidden_size", 128)
        self.fusion = nn.Sequential(
            nn.Linear(total_hidden, total_hidden),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.2)),
            nn.Linear(total_hidden, config.get("output_size", 2)),
        )

    def forward(
        self, features: torch.Tensor, parameters: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-scale processing.
        """
        scale_outputs = []

        for i, (scale, lstm) in enumerate(zip(self.scales, self.scale_lstms)):
            # Downsample features for this scale
            if scale > 1:
                downsampled = features[:, ::scale, :]
            else:
                downsampled = features

            # Process with scale-specific LSTM
            output = lstm(downsampled, parameters)
            scale_outputs.append(output["lstm_output"][:, -1, :])  # Take last timestep

        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=-1)
        final_output = self.fusion(fused)

        return {
            "next_step": final_output,
            "trajectory": final_output,
            "amplitude": torch.norm(final_output, dim=-1, keepdim=True),
            "scale_outputs": scale_outputs,
        }

    def predict_trajectory(
        self, features: torch.Tensor, parameters: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        """
        Multi-scale trajectory prediction.
        """
        # Use the first scale LSTM for trajectory prediction
        return self.scale_lstms[0].predict_trajectory(features, parameters, horizon)


# Register multi-scale model
ModelFactory.register_model("multiscale_lstm", MultiScaleLSTM)
