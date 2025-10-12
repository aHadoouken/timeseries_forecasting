"""
Feature extraction and preprocessing utilities for vibration data.
"""

import numpy as np
import torch
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract various features from vibration time series data.
    """

    def __init__(self, feature_config: Dict):
        """
        Initialize FeatureExtractor.

        Args:
            feature_config: Configuration dictionary specifying which features to extract
        """
        self.feature_config = feature_config
        self.use_features = feature_config.get('use_features', ['basic'])
        self.window_sizes = feature_config.get('window_sizes', [10, 50, 100])

    def extract_features(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        t: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract features from time series data.

        Args:
            x: Position time series
            x_dot: Velocity time series
            t: Time array (optional)

        Returns:
            Feature array of shape [seq_len, n_features]
        """
        features = []

        # Basic features (always included)
        if 'basic' in self.use_features:
            basic_features = self._extract_basic_features(x, x_dot)
            features.append(basic_features)

        # Statistical features
        if 'statistical' in self.use_features:
            stat_features = self._extract_statistical_features(x, x_dot)
            features.append(stat_features)

        # Frequency domain features
        if 'frequency' in self.use_features:
            freq_features = self._extract_frequency_features(x, x_dot, t)
            features.append(freq_features)

        # Nonlinear dynamics features
        if 'nonlinear' in self.use_features:
            nonlinear_features = self._extract_nonlinear_features(x, x_dot)
            features.append(nonlinear_features)

        # Phase space features
        if 'phase_space' in self.use_features:
            phase_features = self._extract_phase_space_features(x, x_dot)
            features.append(phase_features)

        # Concatenate all features
        if features:
            return np.concatenate(features, axis=1)
        else:
            # Fallback to basic features
            return self._extract_basic_features(x, x_dot)

    def _extract_basic_features(self, x: np.ndarray, x_dot: np.ndarray) -> np.ndarray:
        """
        Extract basic features: position and velocity.
        """
        return np.column_stack([x, x_dot])

    def _extract_statistical_features(self, x: np.ndarray, x_dot: np.ndarray) -> np.ndarray:
        """
        Extract statistical features using rolling windows.
        """
        features = []
        seq_len = len(x)

        for window_size in self.window_sizes:
            # Rolling statistics for position
            x_rolling_mean = self._rolling_statistic(x, window_size, np.mean)
            x_rolling_std = self._rolling_statistic(x, window_size, np.std)
            x_rolling_min = self._rolling_statistic(x, window_size, np.min)
            x_rolling_max = self._rolling_statistic(x, window_size, np.max)

            # Rolling statistics for velocity
            xdot_rolling_mean = self._rolling_statistic(x_dot, window_size, np.mean)
            xdot_rolling_std = self._rolling_statistic(x_dot, window_size, np.std)

            # Higher order moments
            x_rolling_skew = self._rolling_statistic(x, window_size, skew)
            x_rolling_kurt = self._rolling_statistic(x, window_size, kurtosis)

            window_features = np.column_stack([
                x_rolling_mean, x_rolling_std, x_rolling_min, x_rolling_max,
                xdot_rolling_mean, xdot_rolling_std,
                x_rolling_skew, x_rolling_kurt
            ])

            features.append(window_features)

        return np.concatenate(features, axis=1)

    def _extract_frequency_features(
        self,
        x: np.ndarray,
        x_dot: np.ndarray,
        t: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract frequency domain features.
        """
        seq_len = len(x)
        features = []

        # Sampling rate estimation
        if t is not None:
            fs = 1.0 / np.mean(np.diff(t))
        else:
            fs = 100.0  # Default sampling rate

        for window_size in self.window_sizes:
            # Rolling spectral features
            spectral_features = []

            for i in range(seq_len):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(seq_len, i + window_size // 2)

                if end_idx - start_idx < window_size // 2:
                    # Pad with previous values if not enough data
                    if i > 0:
                        spectral_features.append(spectral_features[-1])
                    else:
                        spectral_features.append(np.zeros(6))  # 6 spectral features
                    continue

                x_window = x[start_idx:end_idx]

                # Compute power spectral density
                freqs, psd = signal.welch(x_window, fs=fs, nperseg=min(len(x_window), 256))

                # Extract spectral features
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
                spectral_rolloff = self._spectral_rolloff(freqs, psd, 0.85)
                spectral_flux = np.sum(np.diff(psd) ** 2) if len(psd) > 1 else 0

                # Dominant frequency
                dominant_freq = freqs[np.argmax(psd)]

                # Total power
                total_power = np.sum(psd)

                window_spectral_features = [
                    spectral_centroid, spectral_spread, spectral_rolloff,
                    spectral_flux, dominant_freq, total_power
                ]

                spectral_features.append(window_spectral_features)

            features.append(np.array(spectral_features))

        return np.concatenate(features, axis=1)

    def _extract_nonlinear_features(self, x: np.ndarray, x_dot: np.ndarray) -> np.ndarray:
        """
        Extract nonlinear dynamics features.
        """
        seq_len = len(x)
        features = []

        for window_size in self.window_sizes:
            nonlinear_features = []

            for i in range(seq_len):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(seq_len, i + window_size // 2)

                if end_idx - start_idx < window_size // 2:
                    if i > 0:
                        nonlinear_features.append(nonlinear_features[-1])
                    else:
                        nonlinear_features.append(np.zeros(4))  # 4 nonlinear features
                    continue

                x_window = x[start_idx:end_idx]
                xdot_window = x_dot[start_idx:end_idx]

                # Lyapunov exponent approximation
                lyapunov = self._approximate_lyapunov(x_window)

                # Correlation dimension approximation
                corr_dim = self._approximate_correlation_dimension(x_window, xdot_window)

                # Hurst exponent
                hurst = self._hurst_exponent(x_window)

                # Sample entropy
                sample_ent = self._sample_entropy(x_window)

                window_nonlinear_features = [lyapunov, corr_dim, hurst, sample_ent]
                nonlinear_features.append(window_nonlinear_features)

            features.append(np.array(nonlinear_features))

        return np.concatenate(features, axis=1)

    def _extract_phase_space_features(self, x: np.ndarray, x_dot: np.ndarray) -> np.ndarray:
        """
        Extract phase space features.
        """
        seq_len = len(x)
        features = []

        for window_size in self.window_sizes:
            phase_features = []

            for i in range(seq_len):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(seq_len, i + window_size // 2)

                if end_idx - start_idx < window_size // 2:
                    if i > 0:
                        phase_features.append(phase_features[-1])
                    else:
                        phase_features.append(np.zeros(3))  # 3 phase space features
                    continue

                x_window = x[start_idx:end_idx]
                xdot_window = x_dot[start_idx:end_idx]

                # Phase space area (approximate)
                phase_area = self._phase_space_area(x_window, xdot_window)

                # Phase space diameter
                phase_diameter = self._phase_space_diameter(x_window, xdot_window)

                # Energy (kinetic + potential approximation)
                energy = np.mean(xdot_window**2 + x_window**2)

                window_phase_features = [phase_area, phase_diameter, energy]
                phase_features.append(window_phase_features)

            features.append(np.array(phase_features))

        return np.concatenate(features, axis=1)

    def _rolling_statistic(self, data: np.ndarray, window_size: int, stat_func) -> np.ndarray:
        """
        Compute rolling statistic.
        """
        result = np.zeros(len(data))

        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2)

            if end_idx - start_idx < window_size // 2:
                # Use available data or previous value
                if i > 0:
                    result[i] = result[i-1]
                else:
                    result[i] = stat_func(data[start_idx:end_idx]) if end_idx > start_idx else 0
            else:
                result[i] = stat_func(data[start_idx:end_idx])

        return result

    def _spectral_rolloff(self, freqs: np.ndarray, psd: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """
        Compute spectral rolloff frequency.
        """
        total_power = np.sum(psd)
        cumulative_power = np.cumsum(psd)
        rolloff_idx = np.where(cumulative_power >= rolloff_percent * total_power)[0]

        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1]

    def _approximate_lyapunov(self, x: np.ndarray) -> float:
        """
        Approximate largest Lyapunov exponent.
        """
        if len(x) < 10:
            return 0.0

        # Simple approximation using divergence of nearby trajectories
        diffs = np.diff(x)
        log_diffs = np.log(np.abs(diffs) + 1e-10)
        return np.mean(log_diffs)

    def _approximate_correlation_dimension(self, x: np.ndarray, x_dot: np.ndarray) -> float:
        """
        Approximate correlation dimension.
        """
        if len(x) < 10:
            return 1.0

        # Simple approximation using phase space points
        points = np.column_stack([x, x_dot])
        n_points = len(points)

        if n_points < 10:
            return 1.0

        # Sample subset for efficiency
        sample_size = min(50, n_points)
        indices = np.random.choice(n_points, sample_size, replace=False)
        sample_points = points[indices]

        # Compute pairwise distances
        distances = []
        for i in range(len(sample_points)):
            for j in range(i+1, len(sample_points)):
                dist = np.linalg.norm(sample_points[i] - sample_points[j])
                distances.append(dist)

        distances = np.array(distances)
        distances = distances[distances > 0]

        if len(distances) == 0:
            return 1.0

        # Simple correlation dimension approximation
        return np.log(len(distances)) / np.log(np.mean(distances) + 1e-10)

    def _hurst_exponent(self, x: np.ndarray) -> float:
        """
        Compute Hurst exponent using R/S analysis.
        """
        if len(x) < 10:
            return 0.5

        n = len(x)
        mean_x = np.mean(x)

        # Cumulative deviations
        cumdev = np.cumsum(x - mean_x)

        # Range
        R = np.max(cumdev) - np.min(cumdev)

        # Standard deviation
        S = np.std(x)

        if S == 0:
            return 0.5

        # R/S ratio
        rs = R / S

        # Hurst exponent approximation
        return np.log(rs) / np.log(n)

    def _sample_entropy(self, x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Compute sample entropy.
        """
        if len(x) < m + 1:
            return 0.0

        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([x[i:i + m] for i in range(len(x) - m + 1)])
            C = np.zeros(len(patterns))

            for i in range(len(patterns)):
                template = patterns[i]
                for j in range(len(patterns)):
                    if _maxdist(template, patterns[j], m) <= r * np.std(x):
                        C[i] += 1.0

            phi = np.mean(np.log(C / len(patterns)))
            return phi

        return _phi(m) - _phi(m + 1)

    def _phase_space_area(self, x: np.ndarray, x_dot: np.ndarray) -> float:
        """
        Approximate phase space area using convex hull.
        """
        if len(x) < 3:
            return 0.0

        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack([x, x_dot])
            hull = ConvexHull(points)
            return hull.volume
        except:
            # Fallback: use bounding box area
            return (np.max(x) - np.min(x)) * (np.max(x_dot) - np.min(x_dot))

    def _phase_space_diameter(self, x: np.ndarray, x_dot: np.ndarray) -> float:
        """
        Compute maximum distance in phase space.
        """
        if len(x) < 2:
            return 0.0

        points = np.column_stack([x, x_dot])
        max_dist = 0.0

        # Sample subset for efficiency
        sample_size = min(20, len(points))
        indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[indices]

        for i in range(len(sample_points)):
            for j in range(i+1, len(sample_points)):
                dist = np.linalg.norm(sample_points[i] - sample_points[j])
                max_dist = max(max_dist, dist)

        return max_dist

    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        """
        names = []

        if 'basic' in self.use_features:
            names.extend(['x', 'x_dot'])

        if 'statistical' in self.use_features:
            for window_size in self.window_sizes:
                names.extend([
                    f'x_mean_w{window_size}', f'x_std_w{window_size}',
                    f'x_min_w{window_size}', f'x_max_w{window_size}',
                    f'xdot_mean_w{window_size}', f'xdot_std_w{window_size}',
                    f'x_skew_w{window_size}', f'x_kurt_w{window_size}'
                ])

        if 'frequency' in self.use_features:
            for window_size in self.window_sizes:
                names.extend([
                    f'spectral_centroid_w{window_size}', f'spectral_spread_w{window_size}',
                    f'spectral_rolloff_w{window_size}', f'spectral_flux_w{window_size}',
                    f'dominant_freq_w{window_size}', f'total_power_w{window_size}'
                ])

        if 'nonlinear' in self.use_features:
            for window_size in self.window_sizes:
                names.extend([
                    f'lyapunov_w{window_size}', f'corr_dim_w{window_size}',
                    f'hurst_w{window_size}', f'sample_entropy_w{window_size}'
                ])

        if 'phase_space' in self.use_features:
            for window_size in self.window_sizes:
                names.extend([
                    f'phase_area_w{window_size}', f'phase_diameter_w{window_size}',
                    f'energy_w{window_size}'
                ])

        return names

    def get_feature_dim(self) -> int:
        """
        Get total number of features.
        """
        return len(self.get_feature_names())
