# Vibration Prediction System

A comprehensive machine learning system for predicting vibrations in milling systems using LSTM neural networks. The system is designed to predict future vibration trajectories and detect bifurcations (dangerous vibration patterns) based on historical time series data.

## Features

- **LSTM-based Prediction**: Advanced LSTM architecture with attention mechanism for accurate trajectory prediction
- **Bifurcation Detection**: Specialized loss functions and metrics for detecting dangerous vibration patterns
- **Comprehensive Metrics**: Extensive evaluation metrics including trajectory accuracy, amplitude prediction, and bifurcation detection
- **Data Augmentation**: Advanced augmentation techniques for time series data
- **Visualization Tools**: Rich visualization capabilities for analysis and interpretation
- **Modular Architecture**: Clean, extensible codebase with clear separation of concerns
- **Configuration Management**: YAML-based configuration system for easy experimentation

## Project Structure

```
project/
├── config/
│   ├── base_config.yaml          # Base configuration
│   └── lstm_config.yaml          # LSTM-specific configuration
├── data/
│   ├── dataset.py                # Dataset class with resampling and splitting
│   ├── preprocessing.py          # Feature extraction and preprocessing
│   └── augmentation.py           # Data augmentation techniques
├── models/
│   ├── base_model.py             # Base model class
│   └── lstm_predictor.py         # LSTM implementation with attention
├── training/
│   ├── trainer.py                # Training loop and management
│   ├── losses.py                 # Combined loss functions
│   └── metrics.py                # Comprehensive evaluation metrics
├── utils/
│   ├── signal_processing.py      # Signal processing utilities
│   └── visualization.py          # Visualization tools
├── main.py                       # Main entry point
├── example_usage.ipynb           # Jupyter notebook examples
└── requirements.txt              # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vibration-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

1. **Train a model**:
```bash
python main.py --mode train --config config/base_config.yaml --model-config config/lstm_config.yaml --output-dir outputs
```

2. **Evaluate a trained model**:
```bash
python main.py --mode evaluate --model-path outputs/best_model.pth --config config/base_config.yaml --model-config config/lstm_config.yaml --output-dir evaluation
```

3. **Make predictions**:
```bash
python main.py --mode predict --model-path outputs/best_model.pth --config config/base_config.yaml --model-config config/lstm_config.yaml --output-dir predictions
```

### Jupyter Notebook

For interactive exploration and experimentation, use the provided Jupyter notebook:

```bash
jupyter notebook example_usage.ipynb
```

The notebook includes:
- Data loading and exploration
- Model training and evaluation
- Visualization of results
- Attention analysis
- Performance metrics

## Configuration

The system uses YAML configuration files for easy experimentation:

### Base Configuration (`config/base_config.yaml`)

```yaml
data:
  window_size: 500              # Input sequence length
  prediction_horizon: 100      # Prediction horizon
  stride: 50                   # Sliding window stride
  sampling_rate: 10            # Target sampling rate
  train_split: 0.7             # Training data fraction
  val_split: 0.15              # Validation data fraction
  test_split: 0.15             # Test data fraction

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 15
  gradient_clipping: 1.0

loss:
  weights:
    trajectory: 1.0            # Trajectory prediction weight
    amplitude: 2.0             # Amplitude prediction weight
    stability: 0.5             # Stability regularization weight
    physics: 0.1               # Physics-informed loss weight
```

### LSTM Configuration (`config/lstm_config.yaml`)

```yaml
model:
  type: 'lstm'
  hidden_size: 256
  num_layers: 3
  dropout: 0.3
  recurrent_dropout: 0.3
  bidirectional: false
  input_normalization: 'z_score'
  parameter_embedding_dim: 32
```

## Dataset Format

The system expects datasets in pickle format with the following structure:

```python
{
    'trajectories': [
        {
            'id': int,
            'parameters': {
                'k': float,      # Damping coefficient
                'd': float,      # Stiffness parameter
                'e': float,      # Stiffness modulation amplitude
                'tau': float,    # Delay
                'T': float,      # Modulation period
                # ... other parameters
            },
            'solution': {
                't': np.array,      # Time points [500000]
                'x': np.array,      # Position [500000]
                'x_dot': np.array,  # Velocity [500000]
            },
            'metadata': {
                'max_amplitude': float,
                'mean_amplitude': float,
                'std_amplitude': float,
                'duration': float,
                'num_points': int,
                'sampling_rate': float,
                # ... other metadata
            }
        },
        # ... more trajectories
    ]
}
```

## Key Components

### 1. VibrationDataset

Handles data loading, preprocessing, and sequence generation:

- **Stratified splitting** based on amplitude distribution
- **Temporal splitting** option for time-based validation
- **Resampling** with anti-aliasing filters
- **Feature extraction** with configurable feature sets
- **Normalization** (z-score or min-max)

### 2. LSTM Predictor

Advanced LSTM architecture with:

- **Parameter embedding** for system parameters
- **Attention mechanism** for focusing on important time steps
- **Multi-head outputs** for trajectory and amplitude prediction
- **Uncertainty estimation** using Monte Carlo dropout
- **Feature importance** analysis

### 3. Combined Loss Function

Multi-component loss function:

- **Trajectory Loss**: MSE/MAE for sequence prediction
- **Amplitude Loss**: Specialized loss for amplitude prediction with bifurcation penalty
- **Stability Loss**: Regularization for smooth predictions
- **Physics Loss**: Physics-informed constraints

### 4. Comprehensive Metrics

Evaluation metrics include:

- **Trajectory Metrics**: RMSE, MAE, R² for different horizons
- **Amplitude Metrics**: Amplitude prediction accuracy and correlation
- **Bifurcation Detection**: Precision, recall, F1 for dangerous vibrations
- **Frequency Analysis**: Spectral correlation and dominant frequency errors
- **Stability Metrics**: Smoothness and derivative accuracy

### 5. Visualization Tools

Rich visualization capabilities:

- **Trajectory Comparisons**: Input vs. predicted vs. ground truth
- **Phase Space Plots**: Position-velocity phase portraits
- **Amplitude Analysis**: Prediction accuracy and error distributions
- **Frequency Analysis**: Spectral analysis and time-frequency plots
- **Attention Visualization**: Attention weight analysis
- **Bifurcation Analysis**: Parameter space analysis
- **Performance Dashboards**: Comprehensive model evaluation

## Advanced Features

### Data Augmentation

- **Noise Addition**: Gaussian noise injection
- **Time Warping**: Non-linear time distortions
- **Parameter Interpolation**: Parameter space augmentation
- **Magnitude Scaling**: Amplitude variations
- **Frequency Shifting**: Temporal scaling
- **Mixup/Cutmix**: Advanced augmentation techniques

### Signal Processing

- **Envelope Detection**: Hilbert transform, peak detection, RMS
- **Bifurcation Detection**: Automated detection of dangerous patterns
- **Spectral Analysis**: Comprehensive frequency domain features
- **Nonlinear Dynamics**: Lyapunov exponents, Hurst exponent, sample entropy
- **Phase Space Analysis**: Trajectory analysis in phase space

### Model Features

- **Multi-scale Processing**: Different temporal scales
- **Uncertainty Quantification**: Monte Carlo dropout
- **Attention Mechanisms**: Interpretable attention weights
- **Parameter Sensitivity**: Analysis of parameter influence
- **Transfer Learning**: Model adaptation capabilities

## Performance Optimization

- **Efficient Data Loading**: Multi-worker data loading with caching
- **GPU Acceleration**: CUDA support with automatic device detection
- **Memory Management**: Efficient batch processing for large datasets
- **Gradient Clipping**: Stable training for RNNs
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

## Monitoring and Logging

- **Comprehensive Logging**: Detailed training and evaluation logs
- **Progress Tracking**: Real-time training progress with tqdm
- **Metric Tracking**: Historical metric tracking and visualization
- **Model Checkpointing**: Automatic saving of best models
- **Reproducibility**: Fixed random seeds for consistent results

## Extensibility

The modular architecture allows easy extension:

- **New Models**: Implement `BaseVibrationModel` interface
- **Custom Losses**: Add new loss components to `CombinedLoss`
- **Additional Metrics**: Extend `VibrationMetrics` class
- **New Features**: Add feature extractors to `FeatureExtractor`
- **Visualization**: Create new visualization methods

## Examples

### Basic Training

```python
from main import train_model, load_config, merge_configs

# Load configuration
base_config = load_config('config/base_config.yaml')
lstm_config = load_config('config/lstm_config.yaml')
config = merge_configs(base_config, lstm_config)

# Train model
results = train_model(config, output_dir='my_experiment')
print(f"Best validation loss: {results['best_val_loss']:.6f}")
```

### Custom Prediction

```python
from main import predict_trajectory
import numpy as np

# Load your data
input_sequence = np.random.randn(500, 2)  # [time_steps, features]
parameters = np.random.randn(15)          # System parameters

# Make prediction
results = predict_trajectory(
    model_path='outputs/best_model.pth',
    input_data=input_sequence,
    parameters=parameters,
    config=config,
    horizon=100
)

print(f"Predicted amplitude: {results['amplitude']:.4f}")
```

### Visualization

```python
from utils.visualization import VibrationVisualizer

visualizer = VibrationVisualizer()

# Plot trajectory comparison
fig = visualizer.plot_trajectory_comparison(
    input_data=input_sequence,
    target_data=ground_truth,
    predicted_data=predictions,
    save_path='trajectory_comparison.png'
)

# Plot amplitude analysis
fig = visualizer.plot_amplitude_analysis(
    predicted_amplitudes=pred_amps,
    true_amplitudes=true_amps,
    save_path='amplitude_analysis.png'
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in configuration
2. **Slow Training**: Increase number of data loading workers
3. **Poor Convergence**: Adjust learning rate or add gradient clipping
4. **Overfitting**: Increase dropout or add regularization

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Optimize Batch Size**: Find the largest batch size that fits in memory
3. **Data Preprocessing**: Cache preprocessed data for faster loading
4. **Early Stopping**: Use early stopping to prevent overfitting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vibration_prediction_system,
  title={Vibration Prediction System for Milling Applications},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/vibration-prediction}
}
```

## Acknowledgments

- Built with PyTorch for deep learning capabilities
- Uses scipy for signal processing functions
- Visualization powered by matplotlib and seaborn
- Configuration management with PyYAML
