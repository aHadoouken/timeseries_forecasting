"""
Visualization utilities for vibration prediction analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.stats import gaussian_kde
import logging
import os

logger = logging.getLogger(__name__)


class VibrationVisualizer:
    """
    Comprehensive visualization tools for vibration data and predictions.
    """

    def __init__(self, width: int = 1200, height: int = 800):
        """
        Initialize visualizer.

        Args:
            width: Default figure width
            height: Default figure height
        """
        self.width = width
        self.height = height

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
        fig = make_subplots(rows=n_features, cols=1, subplot_titles=feature_names)

        # Create time arrays if not provided
        if time_input is None:
            time_input = np.arange(len(input_data))
        if time_target is None:
            time_target = np.arange(len(input_data), len(input_data) + len(target_data))

        for i, feature_name in enumerate(feature_names):
            # Plot input
            fig.add_trace(go.Scatter(
                x=time_input,
                y=input_data[:, i],
                name='Input',
                line=dict(color='blue', width=2),
                opacity=0.7,
                legendgroup='input',
                showlegend=(i == 0)
            ), row=i+1, col=1)

            # Plot target
            fig.add_trace(go.Scatter(
                x=time_target,
                y=target_data[:, i],
                name='Target',
                line=dict(color='green', width=2),
                legendgroup='target',
                showlegend=(i == 0)
            ), row=i+1, col=1)

            # Plot prediction
            fig.add_trace(go.Scatter(
                x=time_target,
                y=predicted_data[:, i],
                name='Prediction',
                line=dict(color='red', width=2, dash='dash'),
                legendgroup='prediction',
                showlegend=(i == 0)
            ), row=i+1, col=1)

            # Add vertical line at prediction start
            fig.add_vline(
                x=time_input[-1],
                line=dict(color='gray', dash='dot', width=1),
                row=i+1, col=1
            )

            # Highlight prediction region
            fig.add_vrect(
                x0=time_target[0],
                x1=time_target[-1],
                fillcolor="red",
                opacity=0.1,
                line_width=0,
                row=i+1, col=1
            )

            fig.update_yaxes(title_text=feature_name, row=i+1, col=1)

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
            width=self.width,
            height=self.height * n_features,
            showlegend=True,
            template="plotly_white"
        )

        fig.update_xaxes(title_text="Time", row=n_features, col=1)

        if save_path:
            fig.write_image(save_path)

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
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, (traj, label) in enumerate(zip(trajectories, labels)):
            if traj.shape[1] >= 2:
                # Plot trajectory
                fig.add_trace(go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode='lines',
                    name=label,
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.7
                ))

                # Mark start point
                fig.add_trace(go.Scatter(
                    x=[traj[0, 0]],
                    y=[traj[0, 1]],
                    mode='markers',
                    name=f'{label} Start',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=10,
                        symbol='circle',
                        line=dict(width=2, color='black')
                    ),
                    showlegend=True
                ))

                # Mark end point
                fig.add_trace(go.Scatter(
                    x=[traj[-1, 0]],
                    y=[traj[-1, 1]],
                    mode='markers',
                    name=f'{label} End',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=10,
                        symbol='square',
                        line=dict(width=2, color='black')
                    ),
                    showlegend=True
                ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
            xaxis_title='Position',
            yaxis_title='Velocity',
            width=self.width,
            height=self.height,
            showlegend=True,
            template="plotly_white"
        )

        if save_path:
            fig.write_image(save_path)

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
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Predicted vs True Amplitudes',
                'Residuals vs True Amplitudes',
                'Amplitude Distributions',
                'Relative Error Distribution'
            )
        )

        # Scatter plot: predicted vs true
        if parameters is not None and len(parameters.shape) > 1:
            # Color by first parameter
            fig.add_trace(go.Scatter(
                x=true_amplitudes,
                y=predicted_amplitudes,
                mode='markers',
                marker=dict(
                    color=parameters[:, 0],
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title=parameter_names[0] if parameter_names else 'Parameter 0')
                )
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=true_amplitudes,
                y=predicted_amplitudes,
                mode='markers',
                marker=dict(opacity=0.6)
            ), row=1, col=1)

        # Perfect prediction line
        min_amp = min(true_amplitudes.min(), predicted_amplitudes.min())
        max_amp = max(true_amplitudes.max(), predicted_amplitudes.max())
        fig.add_trace(go.Scatter(
            x=[min_amp, max_amp],
            y=[min_amp, max_amp],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ), row=1, col=1)

        # Residuals plot
        residuals = predicted_amplitudes - true_amplitudes
        fig.add_trace(go.Scatter(
            x=true_amplitudes,
            y=residuals,
            mode='markers',
            marker=dict(opacity=0.6)
        ), row=1, col=2)
        fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=2)

        # Distribution comparison
        fig.add_trace(go.Histogram(
            x=true_amplitudes,
            name='True',
            opacity=0.7,
            histnorm='probability density'
        ), row=2, col=1)
        fig.add_trace(go.Histogram(
            x=predicted_amplitudes,
            name='Predicted',
            opacity=0.7,
            histnorm='probability density'
        ), row=2, col=1)

        # Error distribution
        relative_errors = np.abs(residuals) / (true_amplitudes + 1e-8)
        fig.add_trace(go.Histogram(
            x=relative_errors,
            name='Relative Error',
            marker_color='orange',
            opacity=0.7
        ), row=2, col=2)

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
            width=self.width,
            height=self.height * 1.5,
            showlegend=True,
            template="plotly_white",
            barmode='overlay'
        )

        # Axis labels
        fig.update_xaxes(title_text='True Amplitude', row=1, col=1)
        fig.update_yaxes(title_text='Predicted Amplitude', row=1, col=1)
        fig.update_xaxes(title_text='True Amplitude', row=1, col=2)
        fig.update_yaxes(title_text='Residuals', row=1, col=2)
        fig.update_xaxes(title_text='Amplitude', row=2, col=1)
        fig.update_yaxes(title_text='Density', row=2, col=1)
        fig.update_xaxes(title_text='Relative Error', row=2, col=2)
        fig.update_yaxes(title_text='Frequency', row=2, col=2)

        if save_path:
            fig.write_image(save_path)

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
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Time Domain',
                'Power Spectral Density',
                'Spectrogram',
                'Dominant Frequencies'
            )
        )

        colors = px.colors.qualitative.Plotly

        # Time domain
        for i, (signal_data, label) in enumerate(zip(signals, labels)):
            time = np.arange(len(signal_data)) / sampling_rate
            fig.add_trace(go.Scatter(
                x=time,
                y=signal_data,
                name=label,
                line=dict(color=colors[i % len(colors)]),
                opacity=0.7
            ), row=1, col=1)

        # Power Spectral Density
        for i, (signal_data, label) in enumerate(zip(signals, labels)):
            freqs, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=min(len(signal_data), 256))
            fig.add_trace(go.Scatter(
                x=freqs,
                y=psd,
                name=label,
                line=dict(color=colors[i % len(colors)]),
                mode='lines'
            ), row=1, col=2)

        # Spectrogram (first signal only)
        if len(signals) > 0:
            freqs, times, Sxx = signal.spectrogram(signals[0], fs=sampling_rate)
            fig.add_trace(go.Heatmap(
                x=times,
                y=freqs,
                z=10 * np.log10(Sxx),
                colorscale='Viridis',
                name=labels[0]
            ), row=2, col=1)

        # Dominant frequency comparison
        dominant_freqs = []
        for signal_data, label in zip(signals, labels):
            freqs, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=min(len(signal_data), 256))
            dominant_freq = freqs[np.argmax(psd)]
            dominant_freqs.append(dominant_freq)

        fig.add_trace(go.Bar(
            x=labels,
            y=dominant_freqs,
            marker_color=colors[:len(labels)]
        ), row=2, col=2)

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
            width=self.width,
            height=self.height * 1.5,
            showlegend=True,
            template="plotly_white"
        )

        # Axis labels
        fig.update_xaxes(title_text='Time (s)', row=1, col=1)
        fig.update_yaxes(title_text='Amplitude', row=1, col=1)
        fig.update_xaxes(title_text='Frequency (Hz)', row=1, col=2)
        fig.update_yaxes(title_text='Power Spectral Density', row=1, col=2, type="log")
        fig.update_xaxes(title_text='Time (s)', row=2, col=1)
        fig.update_yaxes(title_text='Frequency (Hz)', row=2, col=1)
        fig.update_xaxes(title_text='Signal', row=2, col=2)
        fig.update_yaxes(title_text='Dominant Frequency (Hz)', row=2, col=2)

        if save_path:
            fig.write_image(save_path)

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
        fig = make_subplots(
            rows=(n_params + 2) // 3,
            cols=min(3, n_params),
            subplot_titles=[f'{prediction_name} vs {name}' for name in parameter_names]
        )

        for i, param_name in enumerate(parameter_names):
            row = (i // 3) + 1
            col = (i % 3) + 1

            # Scatter plot
            fig.add_trace(go.Scatter(
                x=parameters[:, i],
                y=predictions,
                mode='markers',
                marker=dict(opacity=0.6)
            ), row=row, col=col)

            # Trend line
            z = np.polyfit(parameters[:, i], predictions, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(parameters[:, i].min(), parameters[:, i].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Trend'
            ), row=row, col=col)

            # Correlation coefficient
            corr = np.corrcoef(parameters[:, i], predictions)[0, 1]
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref=f"x{i+1}",
                yref=f"y{i+1}",
                text=f'r = {corr:.3f}',
                showarrow=False,
                bgcolor="white",
                opacity=0.8
            )

            fig.update_xaxes(title_text=param_name, row=row, col=col)
            fig.update_yaxes(title_text=prediction_name, row=row, col=col)

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
            width=self.width,
            height=self.height * ((n_params + 2) // 3),
            showlegend=False,
            template="plotly_white"
        )

        if save_path:
            fig.write_image(save_path)

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
        if time_steps is None:
            time_steps = np.arange(len(input_sequence))

        fig = make_subplots(rows=2, cols=1, subplot_titles=(
            'Input Sequence with Attention Overlay',
            'Attention Weights Distribution'
        ))

        # Plot features
        for i in range(input_sequence.shape[1]):
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=input_sequence[:, i],
                name=f'Feature {i}',
                line=dict(width=2),
                opacity=0.7
            ), row=1, col=1)

        # Overlay attention as bar chart
        fig.add_trace(go.Bar(
            x=time_steps,
            y=attention_weights,
            name='Attention',
            marker=dict(color='red', opacity=0.3),
            showlegend=False
        ), row=1, col=1)

        # Plot attention weights separately
        fig.add_trace(go.Bar(
            x=time_steps,
            y=attention_weights,
            name='Attention',
            marker=dict(color='red', opacity=0.7)
        ), row=2, col=1)

        # Highlight top attention weights
        top_indices = np.argsort(attention_weights)[-5:]  # Top 5
        for idx in top_indices:
            fig.add_vrect(
                x0=time_steps[idx]-0.5,
                x1=time_steps[idx]+0.5,
                fillcolor="darkred",
                opacity=0.3,
                line_width=0,
                row=2, col=1
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
            width=self.width,
            height=self.height * 1.5,
            showlegend=True,
            template="plotly_white"
        )

        fig.update_xaxes(title_text='Time Step', row=2, col=1)
        fig.update_yaxes(title_text='Attention Weight', row=2, col=1)

        if save_path:
            fig.write_image(save_path)

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
        fig = make_subplots(
            rows=2,
            cols=n_params,
            subplot_titles=[f'Amplitude vs {name}' for name in parameter_names] +
                          [f'{name} Distribution by Regime' for name in parameter_names]
        )

        for i, param_name in enumerate(parameter_names):
            # Parameter vs Amplitude
            fig.add_trace(go.Scatter(
                x=parameters[:, i],
                y=amplitudes,
                mode='markers',
                marker=dict(
                    color=np.where(bifurcation_mask, 'red', 'blue'),
                    opacity=0.6
                ),
                showlegend=False
            ), row=1, col=i+1)

            fig.add_hline(
                y=bifurcation_threshold,
                line=dict(color='red', dash='dash'),
                annotation_text=f'Threshold: {bifurcation_threshold:.2f}',
                row=1, col=i+1
            )

            # Parameter distribution for bifurcation/stable regions
            stable_params = parameters[~bifurcation_mask, i]
            bifurc_params = parameters[bifurcation_mask, i]

            fig.add_trace(go.Histogram(
                x=stable_params,
                name='Stable',
                opacity=0.7,
                histnorm='probability density',
                marker_color='blue'
            ), row=2, col=i+1)

            fig.add_trace(go.Histogram(
                x=bifurc_params,
                name='Bifurcation',
                opacity=0.7,
                histnorm='probability density',
                marker_color='red'
            ), row=2, col=i+1)

            fig.update_xaxes(title_text=param_name, row=1, col=i+1)
            fig.update_yaxes(title_text='Amplitude', row=1, col=i+1)
            fig.update_xaxes(title_text=param_name, row=2, col=i+1)
            fig.update_yaxes(title_text='Density', row=2, col=i+1)

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
            width=self.width * n_params,
            height=self.height * 2,
            showlegend=True,
            template="plotly_white",
            barmode='overlay'
        )

        if save_path:
            fig.write_image(save_path)

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

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatterpolar"}]],
            subplot_titles=(
                'Trajectory Prediction Metrics',
                'Amplitude Prediction Metrics',
                'Bifurcation Detection Metrics',
                'Overall Performance Radar'
            )
        )

        # Trajectory metrics
        if trajectory_metrics:
            metrics_names = list(trajectory_metrics.keys())
            metrics_values = list(trajectory_metrics.values())

            colors = ['lightcoral' if ('rmse' in name.lower() or 'mae' in name.lower()) and value > 0.1
                     else 'lightgreen' if ('r2' in name.lower() and value > 0.8) or
                                         (('rmse' in name.lower() or 'mae' in name.lower()) and value <= 0.1)
                     else 'yellow' for name, value in zip(metrics_names, metrics_values)]

            fig.add_trace(go.Bar(
                x=metrics_names,
                y=metrics_values,
                marker_color=colors,
                opacity=0.7
            ), row=1, col=1)

        # Amplitude metrics
        if amplitude_metrics:
            metrics_names = list(amplitude_metrics.keys())
            metrics_values = list(amplitude_metrics.values())

            fig.add_trace(go.Bar(
                x=metrics_names,
                y=metrics_values,
                marker_color='orange',
                opacity=0.7
            ), row=1, col=2)

        # Bifurcation metrics
        if bifurcation_metrics:
            metrics_names = list(bifurcation_metrics.keys())
            metrics_values = list(bifurcation_metrics.values())

            colors = ['lightgreen' if value > 0.7
                     else 'yellow' if value > 0.5
                     else 'lightcoral' for value in metrics_values]

            fig.add_trace(go.Bar(
                x=metrics_names,
                y=metrics_values,
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)

        # Overall performance radar chart
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
            fig.add_trace(go.Scatterpolar(
                r=list(key_metrics.values()),
                theta=list(key_metrics.keys()),
                fill='toself',
                name='Performance'
            ), row=2, col=2)
            fig.update_polars(radialaxis=dict(visible=True, range=[0, 1]), row=2, col=2)

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
            width=self.width,
            height=self.height * 1.5,
            showlegend=False,
            template="plotly_white"
        )

        if save_path:
            fig.write_image(save_path)

        return fig


def create_training_dashboard(
    training_history: Dict,
    metrics_history: Dict,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive training dashboard.

    Args:
        training_history: Training history dictionary
        metrics_history: Metrics history dictionary
        save_path: Path to save figure

    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
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
