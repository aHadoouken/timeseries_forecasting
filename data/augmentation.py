"""
Data augmentation techniques for vibration time series data.
"""

import numpy as np
import torch
from scipy import signal
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VibrationAugmenter:
    """
    Data augmentation for vibration time series.
    """

    def __init__(self, config: Dict):
        """
        Initialize augmenter with configuration.

        Args:
            config: Augmentation configuration dictionary
        """
        self.noise_level = config.get('noise_level', 0.01)
        self.time_warping = config.get('time_warping', True)
        self.parameter_interpolation = config.get('parameter_interpolation', True)

    def augment_sequence(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        t: np.ndarray,
        parameters: Dict,
        augment_prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Apply random augmentations to a sequence.

        Args:
            x: Position time series
            x_dot: Velocity time series
            t: Time array
            parameters: System parameters
            augment_prob: Probability of applying each augmentation

        Returns:
            Augmented (x, x_dot, t, parameters)
        """
        x_aug, x_dot_aug, t_aug = x.copy(), x_dot.copy(), t.copy()
        params_aug = parameters.copy()

        # Apply augmentations randomly
        if np.random.random() < augment_prob:
            x_aug, x_dot_aug = self._add_noise(x_aug, x_dot_aug)

        if self.time_warping and np.random.random() < augment_prob:
            x_aug, x_dot_aug, t_aug = self._time_warp(x_aug, x_dot_aug, t_aug)

        if self.parameter_interpolation and np.random.random() < augment_prob:
            params_aug = self._interpolate_parameters(params_aug)

        return x_aug, x_dot_aug, t_aug, params_aug

    def _add_noise(self, x: np.ndarray, x_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add Gaussian noise to the signals.
        """
        x_std = np.std(x)
        x_dot_std = np.std(x_dot)

        x_noise = np.random.normal(0, self.noise_level * x_std, x.shape)
        x_dot_noise = np.random.normal(0, self.noise_level * x_dot_std, x_dot.shape)

        return x + x_noise, x_dot + x_dot_noise

    def _time_warp(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        t: np.ndarray,
        warp_strength: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply time warping to the signals.
        """
        n_points = len(t)

        # Create warping function
        warp_points = np.linspace(0, 1, max(5, n_points // 100))
        warp_values = np.random.normal(0, warp_strength, len(warp_points))
        warp_values[0] = warp_values[-1] = 0  # Keep endpoints fixed

        # Interpolate warping function
        warp_interp = interp1d(
            np.linspace(0, 1, len(warp_points)),
            warp_values,
            kind='cubic'
        )

        # Apply warping
        t_normalized = (t - t[0]) / (t[-1] - t[0])
        warp_factors = 1 + warp_interp(t_normalized)

        # Ensure monotonic time
        warp_factors = np.maximum(warp_factors, 0.1)

        # Create new time grid
        dt_original = np.mean(np.diff(t))
        t_warped = t[0] + np.cumsum(warp_factors * dt_original)
        t_warped = t_warped[:len(t)]  # Ensure same length

        # Interpolate signals to new time grid
        x_interp = interp1d(t, x, kind='linear', fill_value='extrapolate')
        x_dot_interp = interp1d(t, x_dot, kind='linear', fill_value='extrapolate')

        x_warped = x_interp(t_warped)
        x_dot_warped = x_dot_interp(t_warped)

        return x_warped, x_dot_warped, t_warped

    def _interpolate_parameters(self, parameters: Dict) -> Dict:
        """
        Create new parameter set by interpolating between existing ones.
        """
        params_aug = parameters.copy()

        # Parameters that can be safely interpolated
        interpolatable_params = ['k', 'd', 'e', 'b', 'b_e', 'nonlin_k', 'F_As', 'F_Ae']

        for param in interpolatable_params:
            if param in params_aug:
                # Add small random variation
                variation = np.random.normal(0, 0.05 * abs(params_aug[param]))
                params_aug[param] += variation

                # Ensure positive values for certain parameters
                if param in ['k', 'd', 'nonlin_k']:
                    params_aug[param] = max(params_aug[param], 1e-6)

        return params_aug

    def magnitude_scaling(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale the magnitude of the signals.
        """
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return x * scale_factor, x_dot * scale_factor

    def frequency_shift(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        t: np.ndarray,
        shift_range: Tuple[float, float] = (0.95, 1.05)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply frequency shifting by time scaling.
        """
        freq_factor = np.random.uniform(shift_range[0], shift_range[1])

        # Scale time
        t_scaled = t * freq_factor

        # Interpolate back to original time grid
        x_interp = interp1d(t_scaled, x, kind='linear', fill_value='extrapolate')
        x_dot_interp = interp1d(t_scaled, x_dot, kind='linear', fill_value='extrapolate')

        x_shifted = x_interp(t)
        x_dot_shifted = x_dot_interp(t) / freq_factor  # Adjust velocity

        return x_shifted, x_dot_shifted, t

    def phase_shift(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        max_shift_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random phase shift.
        """
        max_shift = int(len(x) * max_shift_ratio)
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift > 0:
            x_shifted = np.concatenate([x[shift:], x[:shift]])
            x_dot_shifted = np.concatenate([x_dot[shift:], x_dot[:shift]])
        elif shift < 0:
            x_shifted = np.concatenate([x[shift:], x[:shift]])
            x_dot_shifted = np.concatenate([x_dot[shift:], x_dot[:shift]])
        else:
            x_shifted, x_dot_shifted = x, x_dot

        return x_shifted, x_dot_shifted

    def add_trend(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        trend_strength: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add linear trend to the signals.
        """
        n_points = len(x)
        trend_slope = np.random.uniform(-trend_strength, trend_strength)
        trend = trend_slope * np.linspace(0, 1, n_points)

        return x + trend, x_dot

    def mixup(
        self,
        x1: np.ndarray,
        x_dot1: np.ndarray,
        x2: np.ndarray,
        x_dot2: np.ndarray,
        params1: Dict,
        params2: Dict,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply mixup augmentation between two sequences.
        """
        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)

        # Mix signals
        x_mixed = lam * x1 + (1 - lam) * x2
        x_dot_mixed = lam * x_dot1 + (1 - lam) * x_dot2

        # Mix parameters
        params_mixed = {}
        for key in params1.keys():
            if key in params2:
                if isinstance(params1[key], (int, float)):
                    params_mixed[key] = lam * params1[key] + (1 - lam) * params2[key]
                else:
                    params_mixed[key] = params1[key]  # Keep non-numeric as is
            else:
                params_mixed[key] = params1[key]

        return x_mixed, x_dot_mixed, params_mixed

    def cutmix(
        self,
        x1: np.ndarray,
        x_dot1: np.ndarray,
        x2: np.ndarray,
        x_dot2: np.ndarray,
        cut_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply cutmix augmentation.
        """
        n_points = len(x1)
        cut_length = int(n_points * cut_ratio)
        cut_start = np.random.randint(0, n_points - cut_length)
        cut_end = cut_start + cut_length

        x_mixed = x1.copy()
        x_dot_mixed = x_dot1.copy()

        x_mixed[cut_start:cut_end] = x2[cut_start:cut_end]
        x_dot_mixed[cut_start:cut_end] = x_dot2[cut_start:cut_end]

        return x_mixed, x_dot_mixed


class BatchAugmenter:
    """
    Batch-level augmentation for training.
    """

    def __init__(self, augmenter: VibrationAugmenter):
        self.augmenter = augmenter

    def augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
        augment_prob: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to a batch of data.

        Args:
            batch: Batch dictionary with 'features', 'parameters', 'targets'
            augment_prob: Probability of augmenting each sample

        Returns:
            Augmented batch
        """
        batch_size = batch['features'].shape[0]
        augmented_batch = {key: value.clone() for key, value in batch.items()}

        for i in range(batch_size):
            if np.random.random() < augment_prob:
                # Extract sequences
                features = batch['features'][i].numpy()  # [seq_len, n_features]
                x = features[:, 0]
                x_dot = features[:, 1]

                # Create dummy time array
                t = np.arange(len(x))

                # Extract parameters (convert back from normalized if needed)
                params = {f'param_{j}': batch['parameters'][i, j].item()
                         for j in range(batch['parameters'].shape[1])}

                # Apply augmentation
                x_aug, x_dot_aug, t_aug, params_aug = self.augmenter.augment_sequence(
                    x, x_dot, t, params, augment_prob=1.0
                )

                # Update batch
                augmented_batch['features'][i, :, 0] = torch.FloatTensor(x_aug)
                augmented_batch['features'][i, :, 1] = torch.FloatTensor(x_dot_aug)

                # Update parameters if they changed
                for j, (key, value) in enumerate(params_aug.items()):
                    if j < batch['parameters'].shape[1]:
                        augmented_batch['parameters'][i, j] = value

        return augmented_batch
