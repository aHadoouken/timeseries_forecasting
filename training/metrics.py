"""
Metrics for evaluating vibration prediction models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class VibrationMetrics:
    """
    Comprehensive metrics for vibration prediction evaluation.
    """

    def __init__(self, prediction_horizons: List[int] = [10, 50, 100]):
        """
        Initialize metrics calculator.

        Args:
            prediction_horizons: Different horizons to evaluate
        """
        self.prediction_horizons = prediction_horizons
        self.reset()

    def reset(self):
        """
        Reset accumulated metrics.
        """
        self.predictions = []
        self.targets = []
        self.amplitudes_pred = []
        self.amplitudes_true = []
        self.parameters = []
        self.trajectory_ids = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        amplitudes_pred: Optional[torch.Tensor] = None,
        amplitudes_true: Optional[torch.Tensor] = None,
        parameters: Optional[torch.Tensor] = None,
        trajectory_ids: Optional[List] = None
    ):
        """
        Update metrics with new batch of predictions.

        Args:
            predictions: Model predictions [batch_size, seq_len, features]
            targets: Ground truth targets [batch_size, seq_len, features]
            amplitudes_pred: Predicted amplitudes [batch_size, 1]
            amplitudes_true: True amplitudes [batch_size, 1]
            parameters: System parameters [batch_size, n_params]
            trajectory_ids: Trajectory identifiers
        """
        # Convert to numpy for easier processing
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.append(predictions)
        self.targets.append(targets)

        if amplitudes_pred is not None:
            if isinstance(amplitudes_pred, torch.Tensor):
                amplitudes_pred = amplitudes_pred.detach().cpu().numpy()
            self.amplitudes_pred.append(amplitudes_pred)

        if amplitudes_true is not None:
            if isinstance(amplitudes_true, torch.Tensor):
                amplitudes_true = amplitudes_true.detach().cpu().numpy()
            self.amplitudes_true.append(amplitudes_true)

        if parameters is not None:
            if isinstance(parameters, torch.Tensor):
                parameters = parameters.detach().cpu().numpy()
            self.parameters.append(parameters)

        if trajectory_ids is not None:
            self.trajectory_ids.extend(trajectory_ids)

    def compute_trajectory_metrics(self) -> Dict[str, float]:
        """
        Compute trajectory prediction metrics.
        """
        if not self.predictions:
            return {}

        # Concatenate all predictions and targets
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)

        metrics = {}

        # Overall metrics
        metrics['rmse_overall'] = np.sqrt(mean_squared_error(
            all_targets.reshape(-1),
            all_predictions.reshape(-1)
        ))

        metrics['mae_overall'] = mean_absolute_error(
            all_targets.reshape(-1),
            all_predictions.reshape(-1)
        )

        metrics['r2_overall'] = r2_score(
            all_targets.reshape(-1),
            all_predictions.reshape(-1)
        )

        # Feature-wise metrics
        n_features = all_predictions.shape[-1]
        feature_names = ['position', 'velocity'] if n_features == 2 else [f'feature_{i}' for i in range(n_features)]

        for i, feature_name in enumerate(feature_names):
            pred_feature = all_predictions[:, :, i].reshape(-1)
            target_feature = all_targets[:, :, i].reshape(-1)

            metrics[f'rmse_{feature_name}'] = np.sqrt(mean_squared_error(target_feature, pred_feature))
            metrics[f'mae_{feature_name}'] = mean_absolute_error(target_feature, pred_feature)
            metrics[f'r2_{feature_name}'] = r2_score(target_feature, pred_feature)

        # Horizon-wise metrics
        seq_len = all_predictions.shape[1]
        for horizon in self.prediction_horizons:
            if horizon <= seq_len:
                pred_horizon = all_predictions[:, :horizon, :].reshape(-1)
                target_horizon = all_targets[:, :horizon, :].reshape(-1)

                metrics[f'rmse_horizon_{horizon}'] = np.sqrt(mean_squared_error(target_horizon, pred_horizon))
                metrics[f'mae_horizon_{horizon}'] = mean_absolute_error(target_horizon, pred_horizon)

        return metrics

    def compute_amplitude_metrics(self) -> Dict[str, float]:
        """
        Compute amplitude prediction metrics.
        """
        if not self.amplitudes_pred or not self.amplitudes_true:
            return {}

        pred_amps = np.concatenate(self.amplitudes_pred, axis=0).flatten()
        true_amps = np.concatenate(self.amplitudes_true, axis=0).flatten()

        metrics = {}

        # Basic amplitude metrics
        metrics['amplitude_rmse'] = np.sqrt(mean_squared_error(true_amps, pred_amps))
        metrics['amplitude_mae'] = mean_absolute_error(true_amps, pred_amps)
        metrics['amplitude_r2'] = r2_score(true_amps, pred_amps)

        # Relative error
        relative_errors = np.abs(pred_amps - true_amps) / (true_amps + 1e-8)
        metrics['amplitude_relative_error'] = np.mean(relative_errors)

        # Correlation
        if len(pred_amps) > 1:
            correlation, _ = pearsonr(pred_amps, true_amps)
            metrics['amplitude_correlation'] = correlation

        # Bifurcation detection metrics
        # Define high amplitude threshold (e.g., 95th percentile)
        high_amp_threshold = np.percentile(true_amps, 95)

        true_bifurcations = true_amps > high_amp_threshold
        pred_bifurcations = pred_amps > high_amp_threshold

        if np.any(true_bifurcations) or np.any(pred_bifurcations):
            # Precision, Recall, F1 for bifurcation detection
            tp = np.sum(true_bifurcations & pred_bifurcations)
            fp = np.sum(~true_bifurcations & pred_bifurcations)
            fn = np.sum(true_bifurcations & ~pred_bifurcations)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics['bifurcation_precision'] = precision
            metrics['bifurcation_recall'] = recall
            metrics['bifurcation_f1'] = f1

        return metrics

    def compute_frequency_metrics(self) -> Dict[str, float]:
        """
        Compute frequency domain metrics.
        """
        if not self.predictions:
            return {}

        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)

        metrics = {}

        # Compute for position (first feature)
        for i, (pred_batch, target_batch) in enumerate(zip(all_predictions, all_targets)):
            if i >= 10:  # Limit to first 10 samples for efficiency
                break

            pred_pos = pred_batch[:, 0]
            target_pos = target_batch[:, 0]

            # Compute power spectral density
            freqs_pred, psd_pred = signal.welch(pred_pos, nperseg=min(len(pred_pos), 64))
            freqs_target, psd_target = signal.welch(target_pos, nperseg=min(len(target_pos), 64))

            # Spectral correlation
            if len(psd_pred) == len(psd_target):
                spectral_corr, _ = pearsonr(psd_pred, psd_target)
                metrics[f'spectral_correlation_{i}'] = spectral_corr

            # Dominant frequency comparison
            dom_freq_pred = freqs_pred[np.argmax(psd_pred)]
            dom_freq_target = freqs_target[np.argmax(psd_target)]

            metrics[f'dominant_freq_error_{i}'] = abs(dom_freq_pred - dom_freq_target)

        # Average frequency metrics
        if any('spectral_correlation' in k for k in metrics.keys()):
            spectral_corrs = [v for k, v in metrics.items() if 'spectral_correlation' in k]
            metrics['avg_spectral_correlation'] = np.mean(spectral_corrs)

        if any('dominant_freq_error' in k for k in metrics.keys()):
            freq_errors = [v for k, v in metrics.items() if 'dominant_freq_error' in k]
            metrics['avg_dominant_freq_error'] = np.mean(freq_errors)

        return metrics

    def compute_stability_metrics(self) -> Dict[str, float]:
        """
        Compute stability and smoothness metrics.
        """
        if not self.predictions:
            return {}

        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)

        metrics = {}

        # Compute derivatives for smoothness
        pred_derivatives = np.diff(all_predictions, axis=1)
        target_derivatives = np.diff(all_targets, axis=1)

        # Derivative RMSE (smoothness)
        metrics['derivative_rmse'] = np.sqrt(mean_squared_error(
            target_derivatives.reshape(-1),
            pred_derivatives.reshape(-1)
        ))

        # Second derivatives (acceleration)
        if all_predictions.shape[1] > 2:
            pred_second_derivatives = np.diff(pred_derivatives, axis=1)
            target_second_derivatives = np.diff(target_derivatives, axis=1)

            metrics['second_derivative_rmse'] = np.sqrt(mean_squared_error(
                target_second_derivatives.reshape(-1),
                pred_second_derivatives.reshape(-1)
            ))

        # Stability measure: variance of second derivatives
        if all_predictions.shape[1] > 2:
            pred_stability = np.var(pred_second_derivatives, axis=1).mean()
            target_stability = np.var(target_second_derivatives, axis=1).mean()

            metrics['prediction_stability'] = pred_stability
            metrics['target_stability'] = target_stability
            metrics['stability_ratio'] = pred_stability / (target_stability + 1e-8)

        return metrics

    def compute_parameter_sensitivity(self) -> Dict[str, float]:
        """
        Compute metrics related to parameter sensitivity.
        """
        if not self.parameters or not self.predictions:
            return {}

        all_parameters = np.concatenate(self.parameters, axis=0)
        all_predictions = np.concatenate(self.predictions, axis=0)

        metrics = {}

        # Compute prediction variance for different parameter ranges
        n_params = all_parameters.shape[1]
        param_names = [f'param_{i}' for i in range(n_params)]

        for i, param_name in enumerate(param_names):
            param_values = all_parameters[:, i]

            # Split into low/high parameter values
            median_val = np.median(param_values)
            low_mask = param_values < median_val
            high_mask = param_values >= median_val

            if np.any(low_mask) and np.any(high_mask):
                pred_low = all_predictions[low_mask]
                pred_high = all_predictions[high_mask]

                # Compute variance difference
                var_low = np.var(pred_low.reshape(-1))
                var_high = np.var(pred_high.reshape(-1))

                metrics[f'{param_name}_variance_ratio'] = var_high / (var_low + 1e-8)

        return metrics

    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all available metrics.
        """
        all_metrics = {}

        # Trajectory metrics
        traj_metrics = self.compute_trajectory_metrics()
        all_metrics.update(traj_metrics)

        # Amplitude metrics
        amp_metrics = self.compute_amplitude_metrics()
        all_metrics.update(amp_metrics)

        # Frequency metrics
        freq_metrics = self.compute_frequency_metrics()
        all_metrics.update(freq_metrics)

        # Stability metrics
        stability_metrics = self.compute_stability_metrics()
        all_metrics.update(stability_metrics)

        # Parameter sensitivity
        param_metrics = self.compute_parameter_sensitivity()
        all_metrics.update(param_metrics)

        return all_metrics

    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get a summary of the most important metrics.
        """
        all_metrics = self.compute_all_metrics()

        summary = {}

        # Key trajectory metrics
        if 'rmse_overall' in all_metrics:
            summary['RMSE'] = all_metrics['rmse_overall']
        if 'mae_overall' in all_metrics:
            summary['MAE'] = all_metrics['mae_overall']
        if 'r2_overall' in all_metrics:
            summary['R²'] = all_metrics['r2_overall']

        # Key amplitude metrics
        if 'amplitude_rmse' in all_metrics:
            summary['Amplitude_RMSE'] = all_metrics['amplitude_rmse']
        if 'bifurcation_f1' in all_metrics:
            summary['Bifurcation_F1'] = all_metrics['bifurcation_f1']

        # Key stability metrics
        if 'derivative_rmse' in all_metrics:
            summary['Smoothness'] = all_metrics['derivative_rmse']

        return summary

    def print_summary(self):
        """
        Print a formatted summary of metrics.
        """
        summary = self.get_summary_metrics()

        print("\n" + "="*50)
        print("VIBRATION PREDICTION METRICS SUMMARY")
        print("="*50)

        for metric_name, value in summary.items():
            print(f"{metric_name:20s}: {value:.6f}")

        print("="*50)


class MetricsTracker:
    """
    Track metrics over training epochs.
    """

    def __init__(self):
        self.train_metrics = {}
        self.val_metrics = {}
        self.epoch_history = []

    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """
        Update metrics for an epoch.
        """
        self.epoch_history.append(epoch)

        for key, value in train_metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            self.train_metrics[key].append(value)

        for key, value in val_metrics.items():
            if key not in self.val_metrics:
                self.val_metrics[key] = []
            self.val_metrics[key].append(value)

    def get_best_epoch(self, metric_name: str, mode: str = 'min') -> Tuple[int, float]:
        """
        Get the epoch with the best value for a specific metric.
        """
        if metric_name not in self.val_metrics:
            raise ValueError(f"Metric {metric_name} not found")

        values = self.val_metrics[metric_name]

        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        best_epoch = self.epoch_history[best_idx]
        best_value = values[best_idx]

        return best_epoch, best_value

    def get_metric_history(self, metric_name: str) -> Tuple[List, List]:
        """
        Get training and validation history for a metric.
        """
        train_history = self.train_metrics.get(metric_name, [])
        val_history = self.val_metrics.get(metric_name, [])

        return train_history, val_history

    def is_improving(self, metric_name: str, patience: int = 5, mode: str = 'min') -> bool:
        """
        Check if a metric is still improving.
        """
        if metric_name not in self.val_metrics:
            return True

        values = self.val_metrics[metric_name]

        if len(values) < patience:
            return True

        recent_values = values[-patience:]

        if mode == 'min':
            return min(recent_values) == recent_values[-1]
        else:
            return max(recent_values) == recent_values[-1]
