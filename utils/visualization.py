"""
Visualization utilities for vibration prediction analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.stats import gaussian_kde
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VibrationVisualizer:
    """
    Comprehensive visualization tools for vibration data and predictions.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_trajectory_comparison(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        predicted_data: np.ndarray,
        time_input: Optional[np.ndarray] = None,
        time_target: Optional[np.ndarray] = None,
        feature_names: List[str] = ['Position', 'Velocity'],
        title: str = "Trajectory Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot comparison between input, target, and predicted trajectories.

        Args:
            input_data: Input sequence [seq_len, n_features]
            target_data: Target sequence [pred_len, n_features]
            predicted_data: Predicted sequence [pred_len, n_features]
            time_input: Time array for input
            time_target: Time array for target/prediction
            feature_names: Names of features
            title: Plot title
            save_path: Path to save figure
        """
        n_features = input_data.shape[1]
        fig, axes = plt.subplots(n_features, 1, figsize=self.figsize, dpi=self.dpi)

        if n_features == 1:
            axes = [axes]

        # Create time arrays if not provided
        if time_input is None:
            time_input = np.arange(len(input_data))
        if time_target is None:
            time_target = np.arange(len(input_data), len(input_data) + len(target_data))

        for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            # Plot input
            ax.plot(time_input, input_data[:, i],
                   label='Input', color='blue', alpha=0.7, linewidth=2)

            # Plot target
            ax.plot(time_target, target_data[:, i],
                   label='Target', color='green', linewidth=2)

            # Plot prediction
            ax.plot(time_target, predicted_data[:, i],
                   label='Prediction', color='red', linestyle='--', linewidth=2)

            # Add vertical line at prediction start
            ax.axvline(x=time_input[-1], color='gray', linestyle=':', alpha=0.7)

            # Formatting
            ax.set_ylabel(feature_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Highlight prediction region
            ax.axvspan(time_target[0], time_target[-1], alpha=0.1, color='red')

        axes[-1].set_xlabel('Time')
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_phase_space(
        self,
        trajectories: List[np.ndarray],
        labels: List[str],
        title: str = "Phase Space Plot",
        save_path: Optional[str] = None
    ):
        """
        Plot phase space (position vs velocity) for multiple trajectories.

        Args:
            trajectories: List of trajectory arrays [seq_len, 2]
            labels: Labels for each trajectory
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

        for traj, label, color in zip(trajectories, labels, colors):
            if traj.shape[1] >= 2:
                ax.plot(traj[:, 0], traj[:, 1], label=label, color=color, alpha=0.7)

                # Mark start and end points
                ax.scatter(traj[0, 0], traj[0, 1], color=color, s=100, marker='o',
                          edgecolor='black', linewidth=2, label=f'{label} Start')
                ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=100, marker='s',
                          edgecolor='black', linewidth=2, label=f'{label} End')

        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_amplitude_analysis(
        self,
        predicted_amplitudes: np.ndarray,
        true_amplitudes: np.ndarray,
        parameters: Optional[np.ndarray] = None,
        parameter_names: Optional[List[str]] = None,
        title: str = "Amplitude Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot amplitude prediction analysis.

        Args:
            predicted_amplitudes: Predicted amplitudes
            true_amplitudes: True amplitudes
            parameters: System parameters for coloring
            parameter_names: Names of parameters
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

        # Scatter plot: predicted vs true
        ax = axes[0, 0]
        if parameters is not None and len(parameters.shape) > 1:
            # Color by first parameter
            scatter = ax.scatter(true_amplitudes, predicted_amplitudes,
                               c=parameters[:, 0], alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label=parameter_names[0] if parameter_names else 'Parameter 0')
        else:
            ax.scatter(true_amplitudes, predicted_amplitudes, alpha=0.6)

        # Perfect prediction line
        min_amp, max_amp = min(true_amplitudes.min(), predicted_amplitudes.min()), \
                          max(true_amplitudes.max(), predicted_amplitudes.max())
        ax.plot([min_amp, max_amp], [min_amp, max_amp], 'r--', label='Perfect Prediction')

        ax.set_xlabel('True Amplitude')
        ax.set_ylabel('Predicted Amplitude')
        ax.set_title('Predicted vs True Amplitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residuals plot
        ax = axes[0, 1]
        residuals = predicted_amplitudes - true_amplitudes
        ax.scatter(true_amplitudes, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('True Amplitude')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs True Amplitudes')
        ax.grid(True, alpha=0.3)

        # Distribution comparison
        ax = axes[1, 0]
        ax.hist(true_amplitudes, bins=30, alpha=0.7, label='True', density=True)
        ax.hist(predicted_amplitudes, bins=30, alpha=0.7, label='Predicted', density=True)
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Density')
        ax.set_title('Amplitude Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error distribution
        ax = axes[1, 1]
        relative_errors = np.abs(residuals) / (true_amplitudes + 1e-8)
        ax.hist(relative_errors, bins=30, alpha=0.7, color='orange')
        ax.set_xlabel('Relative Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Relative Error Distribution')
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_frequency_analysis(
        self,
        signals: List[np.ndarray],
        labels: List[str],
        sampling_rate: float = 100.0,
        title: str = "Frequency Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot frequency domain analysis of signals.

        Args:
            signals: List of time series signals
            labels: Labels for each signal
            sampling_rate: Sampling rate in Hz
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))

        # Time domain
        ax = axes[0, 0]
        for signal_data, label, color in zip(signals, labels, colors):
            time = np.arange(len(signal_data)) / sampling_rate
            ax.plot(time, signal_data, label=label, color=color, alpha=0.7)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Time Domain')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Power Spectral Density
        ax = axes[0, 1]
        for signal_data, label, color in zip(signals, labels, colors):
            freqs, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=min(len(signal_data), 256))
            ax.semilogy(freqs, psd, label=label, color=color)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Power Spectral Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Spectrogram (first signal only)
        ax = axes[1, 0]
        if len(signals) > 0:
            freqs, times, Sxx = signal.spectrogram(signals[0], fs=sampling_rate)
            im = ax.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Spectrogram - {labels[0]}')
            plt.colorbar(im, ax=ax, label='Power (dB)')

        # Dominant frequency comparison
        ax = axes[1, 1]
        dominant_freqs = []
        for signal_data, label, color in zip(signals, labels, colors):
            freqs, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=min(len(signal_data), 256))
            dominant_freq = freqs[np.argmax(psd)]
            dominant_freqs.append(dominant_freq)

        ax.bar(labels, dominant_freqs, color=colors, alpha=0.7)
        ax.set_ylabel('Dominant Frequency (Hz)')
        ax.set_title('Dominant Frequencies')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_parameter_sensitivity(
        self,
        parameters: np.ndarray,
        predictions: np.ndarray,
        parameter_names: List[str],
        prediction_name: str = "Prediction",
        title: str = "Parameter Sensitivity Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot parameter sensitivity analysis.

        Args:
            parameters: Parameter values [n_samples, n_params]
            predictions: Prediction values [n_samples]
            parameter_names: Names of parameters
            prediction_name: Name of prediction variable
            title: Plot title
            save_path: Path to save figure
        """
        n_params = len(parameter_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=self.dpi)

        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, param_name in enumerate(parameter_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            # Scatter plot
            ax.scatter(parameters[:, i], predictions, alpha=0.6)

            # Trend line
            z = np.polyfit(parameters[:, i], predictions, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(parameters[:, i].min(), parameters[:, i].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

            # Correlation coefficient
            corr = np.corrcoef(parameters[:, i], predictions)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

            ax.set_xlabel(param_name)
            ax.set_ylabel(prediction_name)
            ax.set_title(f'{prediction_name} vs {param_name}')
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        input_sequence: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        title: str = "Attention Weights Visualization",
        save_path: Optional[str] = None
    ):
        """
        Plot attention weights over input sequence.

        Args:
            attention_weights: Attention weights [seq_len]
            input_sequence: Input sequence [seq_len, n_features]
            time_steps: Time steps array
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)

        if time_steps is None:
            time_steps = np.arange(len(input_sequence))

        # Plot input sequence with attention overlay
        ax = axes[0]

        # Plot features
        for i in range(input_sequence.shape[1]):
            ax.plot(time_steps, input_sequence[:, i],
                   label=f'Feature {i}', alpha=0.7)

        # Overlay attention as background color
        ax2 = ax.twinx()
        ax2.fill_between(time_steps, 0, attention_weights,
                        alpha=0.3, color='red', label='Attention')
        ax2.set_ylabel('Attention Weight')
        ax2.set_ylim(0, attention_weights.max() * 1.1)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Feature Value')
        ax.set_title('Input Sequence with Attention Overlay')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot attention weights separately
        ax = axes[1]
        bars = ax.bar(time_steps, attention_weights, alpha=0.7, color='red')

        # Highlight top attention weights
        top_indices = np.argsort(attention_weights)[-5:]  # Top 5
        for idx in top_indices:
            bars[idx].set_color('darkred')
            bars[idx].set_alpha(1.0)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Attention Weights Distribution')
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_bifurcation_analysis(
        self,
        parameters: np.ndarray,
        amplitudes: np.ndarray,
        parameter_names: List[str],
        bifurcation_threshold: Optional[float] = None,
        title: str = "Bifurcation Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot bifurcation analysis showing parameter regions leading to high amplitudes.

        Args:
            parameters: Parameter values [n_samples, n_params]
            amplitudes: Amplitude values [n_samples]
            parameter_names: Names of parameters
            bifurcation_threshold: Threshold for bifurcation detection
            title: Plot title
            save_path: Path to save figure
        """
        if bifurcation_threshold is None:
            bifurcation_threshold = np.percentile(amplitudes, 95)

        # Create bifurcation mask
        bifurcation_mask = amplitudes > bifurcation_threshold

        n_params = len(parameter_names)
        fig, axes = plt.subplots(2, n_params, figsize=(5*n_params, 8), dpi=self.dpi)

        if n_params == 1:
            axes = axes.reshape(-1, 1)

        for i, param_name in enumerate(parameter_names):
            # Parameter vs Amplitude
            ax = axes[0, i]

            # Plot all points
            ax.scatter(parameters[:, i], amplitudes,
                      c=bifurcation_mask, cmap='RdYlBu_r', alpha=0.6)

            # Bifurcation threshold line
            ax.axhline(y=bifurcation_threshold, color='red', linestyle='--',
                      label=f'Bifurcation Threshold ({bifurcation_threshold:.2f})')

            ax.set_xlabel(param_name)
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Amplitude vs {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Parameter distribution for bifurcation/stable regions
            ax = axes[1, i]

            stable_params = parameters[~bifurcation_mask, i]
            bifurc_params = parameters[bifurcation_mask, i]

            ax.hist(stable_params, bins=30, alpha=0.7, label='Stable', density=True)
            ax.hist(bifurc_params, bins=30, alpha=0.7, label='Bifurcation', density=True)

            ax.set_xlabel(param_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{param_name} Distribution by Regime')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_model_performance_summary(
        self,
        metrics_dict: Dict[str, float],
        title: str = "Model Performance Summary",
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive model performance summary.

        Args:
            metrics_dict: Dictionary of metrics
            title: Plot title
            save_path: Path to save figure
        """
        # Group metrics by category
        trajectory_metrics = {k: v for k, v in metrics_dict.items()
                            if any(x in k.lower() for x in ['rmse', 'mae', 'r2'])}
        amplitude_metrics = {k: v for k, v in metrics_dict.items()
                           if 'amplitude' in k.lower()}
        bifurcation_metrics = {k: v for k, v in metrics_dict.items()
                             if 'bifurcation' in k.lower()}

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

        # Trajectory metrics
        if trajectory_metrics:
            ax = axes[0, 0]
            metrics_names = list(trajectory_metrics.keys())
            metrics_values = list(trajectory_metrics.values())

            bars = ax.bar(range(len(metrics_names)), metrics_values, alpha=0.7)
            ax.set_xticks(range(len(metrics_names)))
            ax.set_xticklabels(metrics_names, rotation=45, ha='right')
            ax.set_ylabel('Metric Value')
            ax.set_title('Trajectory Prediction Metrics')
            ax.grid(True, alpha=0.3)

            # Color bars based on performance (lower is better for RMSE/MAE)
            for i, (name, bar) in enumerate(zip(metrics_names, bars)):
                if any(x in name.lower() for x in ['rmse', 'mae']):
                    bar.set_color('lightcoral' if metrics_values[i] > 0.1 else 'lightgreen')
                else:  # R2 - higher is better
                    bar.set_color('lightgreen' if metrics_values[i] > 0.8 else 'lightcoral')

        # Amplitude metrics
        if amplitude_metrics:
            ax = axes[0, 1]
            metrics_names = list(amplitude_metrics.keys())
            metrics_values = list(amplitude_metrics.values())

            ax.bar(range(len(metrics_names)), metrics_values, alpha=0.7, color='orange')
            ax.set_xticks(range(len(metrics_names)))
            ax.set_xticklabels(metrics_names, rotation=45, ha='right')
            ax.set_ylabel('Metric Value')
            ax.set_title('Amplitude Prediction Metrics')
            ax.grid(True, alpha=0.3)

        # Bifurcation metrics
        if bifurcation_metrics:
            ax = axes[1, 0]
            metrics_names = list(bifurcation_metrics.keys())
            metrics_values = list(bifurcation_metrics.values())

            bars = ax.bar(range(len(metrics_names)), metrics_values, alpha=0.7, color='purple')
            ax.set_xticks(range(len(metrics_names)))
            ax.set_xticklabels(metrics_names, rotation=45, ha='right')
            ax.set_ylabel('Metric Value')
            ax.set_title('Bifurcation Detection Metrics')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # Add performance indicators
            for i, (name, value, bar) in enumerate(zip(metrics_names, metrics_values, bars)):
                color = 'lightgreen' if value > 0.7 else 'yellow' if value > 0.5 else 'lightcoral'
                bar.set_color(color)

        # Overall performance radar chart
        ax = axes[1, 1]

        # Select key metrics for radar chart
        key_metrics = {}
        if 'rmse_overall' in metrics_dict:
            key_metrics['RMSE'] = 1 - min(metrics_dict['rmse_overall'], 1)  # Invert for radar
        if 'r2_overall' in metrics_dict:
            key_metrics['R²'] = metrics_dict['r2_overall']
        if 'amplitude_r2' in metrics_dict:
            key_metrics['Amp R²'] = metrics_dict['amplitude_r2']
        if 'bifurcation_f1' in metrics_dict:
            key_metrics['Bifurc F1'] = metrics_dict['bifurcation_f1']

        if key_metrics:
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(key_metrics), endpoint=False)
            values = list(key_metrics.values())

            # Close the plot
            angles = np.concatenate((angles, [angles[0]]))
            values = values + [values[0]]

            ax.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax.fill(angles, values, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(key_metrics.keys())
            ax.set_ylim(0, 1)
            ax.set_title('Overall Performance')
            ax.grid(True)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig


def create_training_dashboard(
    training_history: Dict,
    metrics_history: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive training dashboard.

    Args:
        training_history: Training history dictionary
        metrics_history: Metrics history dictionary
        save_path: Path to save figure

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(20, 12))

    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(len(training_history['train_loss']))
    ax1.plot(epochs, training_history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, training_history['val_loss'], label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # RMSE curves
    ax2 = fig.add_subplot(gs[0, 2:])
    if 'rmse_overall' in metrics_history:
        ax2.plot(epochs, metrics_history['rmse_overall'], label='RMSE', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Amplitude metrics
    ax3 = fig.add_subplot(gs[1, :2])
    if 'amplitude_rmse' in metrics_history:
        ax3.plot(epochs, metrics_history['amplitude_rmse'], label='Amplitude RMSE', color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Amplitude RMSE')
        ax3.set_title('Amplitude Prediction Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Bifurcation F1
    ax4 = fig.add_subplot(gs[1, 2:])
    if 'bifurcation_f1' in metrics_history:
        ax4.plot(epochs, metrics_history['bifurcation_f1'], label='Bifurcation F1', color='purple')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Bifurcation Detection Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Final metrics summary
    ax5 = fig.add_subplot(gs[2, :])

    # Get final metrics
    final_metrics = {}
    for key, values in metrics_history.items():
        if values:
            final_metrics[key] = values[-1]

    if final_metrics:
        metric_names = list(final_metrics.keys())[:10]  # Show top 10 metrics
        metric_values = [final_metrics[name] for name in metric_names]

        bars = ax5.bar(range(len(metric_names)), metric_values, alpha=0.7)
        ax5.set_xticks(range(len(metric_names)))
        ax5.set_xticklabels(metric_names, rotation=45, ha='right')
        ax5.set_ylabel('Metric Value')
        ax5.set_title('Final Metrics Summary')
        ax5.grid(True, alpha=0.3)

        # Color code bars
        for i, (name, value, bar) in enumerate(zip(metric_names, metric_values, bars)):
            if 'rmse' in name.lower() or 'mae' in name.lower():
                color = 'lightgreen' if value < 0.1 else 'yellow' if value < 0.2 else 'lightcoral'
            else:
                color = 'lightgreen' if value > 0.8 else 'yellow' if value > 0.5 else 'lightcoral'
            bar.set_color(color)

    plt.suptitle('Training Dashboard', fontsize=20, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
