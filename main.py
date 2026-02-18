"""
Main entry point for vibration prediction system.
"""

import os
import sys
import yaml
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.dataset import VibrationDataset, create_dataloaders
from models.base_model import ModelFactory
from training.trainer import Trainer
from utils.visualization import VibrationVisualizer
from utils.signal_processing import compute_spectral_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vibration_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def merge_configs(base_config: Dict, model_config: Dict) -> Dict:
    """
    Merge base configuration with model-specific configuration.

    Args:
        base_config: Base configuration
        model_config: Model-specific configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    # Merge nested dictionaries
    for key, value in model_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value

    return merged


def setup_reproducibility(seed: int = 42):
    """
    Setup reproducible training environment.

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Reproducibility setup with seed: {seed}")


def train_model(config: Dict, output_dir: str = "outputs") -> Dict:
    """
    Train vibration prediction model.

    Args:
        config: Training configuration
        output_dir: Output directory for results

    Returns:
        Training results
    """
    # Setup reproducibility
    setup_reproducibility(config.get('random_seed', 42))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    config['output_dir'] = output_dir

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create data loaders
    logger.info("Creating data loaders...")

    # Prepare dataset configuration and create single dataset
    dataset_config = {
        'data_path': os.path.join(config['data']['dataset_path'], config['data']['dataset_name']),
        'window_size': config['data']['window_size'],
        'prediction_horizon': config['data']['prediction_horizon'],
        'stride': config['data']['stride'],
        'train_split': config['data']['train_split'],
        'val_split': config['data']['val_split'],
        'test_split': config['data']['test_split'],
        'sampling_rate': config['data']['sampling_rate'],
        'features': config['features']['use_features'],
        'random_seed': config.get('random_seed', 42)
    }
    dataset = VibrationDataset(split='train', **dataset_config)
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4
    )

    logger.info(f"Data loaders created: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")

    # Create model
    logger.info("Creating model...")
    model_config = config['model'].copy()
    model_config['input_size'] = 2  # x, x_dot
    model_config['output_size'] = 2  # x, x_dot
    model_config['n_params'] = 15  # Number of system parameters

    model = ModelFactory.create_model(model_config)
    logger.info(f"Model created: {model.get_model_info()}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device
    )

    # Train model
    logger.info("Starting training...")
    results = trainer.train()

    # Save final model
    model_path = os.path.join(output_dir, 'final_model.pth')
    model.save_checkpoint(model_path, results['total_epochs'])

    logger.info(f"Training completed. Results saved to {output_dir}")
    return results


def evaluate_model(model_path: str, config: Dict, output_dir: str = "evaluation") -> Dict:
    """
    Evaluate trained model.

    Args:
        model_path: Path to trained model
        config: Configuration
        output_dir: Output directory for evaluation results

    Returns:
        Evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    logger.info(f"Loading model from {model_path}")
    model, checkpoint = ModelFactory.create_model(config['model']).load_checkpoint(model_path, device)
    model.eval()

    # Create test data loader
    dataset_config = {
        'data_path': os.path.join(config['data']['dataset_path'], config['data']['dataset_name']),
        'window_size': config['data']['window_size'],
        'prediction_horizon': config['data']['prediction_horizon'],
        'stride': config['data']['stride'],
        'train_split': config['data']['train_split'],
        'val_split': config['data']['val_split'],
        'test_split': config['data']['test_split'],
        'sampling_rate': config['data']['sampling_rate'],
        'features': config['features']['use_features'],
        'random_seed': config.get('random_seed', 42)
    }
    dataset = VibrationDataset(split='train', **dataset_config)
    _, _, test_loader = create_dataloaders(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4
    )

    # Evaluate model
    logger.info("Evaluating model...")

    from training.metrics import VibrationMetrics
    metrics = VibrationMetrics()

    all_predictions = []
    all_targets = []
    all_amplitudes_pred = []
    all_amplitudes_true = []
    all_parameters = []

    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(batch['features'], batch['parameters'])

            # Collect results
            all_predictions.append(outputs['trajectory'].cpu().numpy())
            all_targets.append(batch['targets'].cpu().numpy())

            if 'amplitude' in outputs:
                all_amplitudes_pred.append(outputs['amplitude'].cpu().numpy())
            if 'max_amplitude' in batch:
                all_amplitudes_true.append(batch['max_amplitude'].cpu().numpy())

            all_parameters.append(batch['parameters'].cpu().numpy())

            # Update metrics
            metrics.update(
                predictions=outputs['trajectory'],
                targets=batch['targets'],
                amplitudes_pred=outputs.get('amplitude'),
                amplitudes_true=batch.get('max_amplitude'),
                parameters=batch['parameters']
            )

    # Compute final metrics
    evaluation_results = metrics.compute_all_metrics()

    # Print summary
    metrics.print_summary()

    # Create visualizations
    logger.info("Creating visualizations...")
    visualizer = VibrationVisualizer()

    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    parameters = np.concatenate(all_parameters, axis=0)

    if all_amplitudes_pred and all_amplitudes_true:
        amplitudes_pred = np.concatenate(all_amplitudes_pred, axis=0)
        amplitudes_true = np.concatenate(all_amplitudes_true, axis=0)

        # Amplitude analysis
        fig = visualizer.plot_amplitude_analysis(
            amplitudes_pred.flatten(),
            amplitudes_true.flatten(),
            parameters,
            parameter_names=[f'param_{i}' for i in range(parameters.shape[1])],
            save_path=os.path.join(output_dir, 'amplitude_analysis.png')
        )

        # Bifurcation analysis
        fig = visualizer.plot_bifurcation_analysis(
            parameters,
            amplitudes_true.flatten(),
            parameter_names=[f'param_{i}' for i in range(parameters.shape[1])],
            save_path=os.path.join(output_dir, 'bifurcation_analysis.png')
        )

    # Sample trajectory comparisons
    n_samples = min(5, len(predictions))
    for i in range(n_samples):
        # Create dummy input data (use first part of target as input)
        input_data = targets[i, :config['data']['window_size']//2, :]
        target_data = targets[i, config['data']['window_size']//2:, :]
        predicted_data = predictions[i]

        fig = visualizer.plot_trajectory_comparison(
            input_data,
            target_data,
            predicted_data,
            title=f'Sample {i+1} Trajectory Comparison',
            save_path=os.path.join(output_dir, f'trajectory_comparison_{i+1}.png')
        )

    # Model performance summary
    fig = visualizer.plot_model_performance_summary(
        evaluation_results,
        save_path=os.path.join(output_dir, 'performance_summary.png')
    )

    # Save evaluation results
    results_path = os.path.join(output_dir, 'evaluation_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(evaluation_results, f, default_flow_style=False)

    logger.info(f"Evaluation completed. Results saved to {output_dir}")
    return evaluation_results


def predict_trajectory(
    model_path: str,
    input_data: np.ndarray,
    parameters: np.ndarray,
    config: Dict,
    horizon: int = 100
) -> Dict:
    """
    Predict trajectory for given input.

    Args:
        model_path: Path to trained model
        input_data: Input sequence [seq_len, 2]
        parameters: System parameters [n_params]
        config: Configuration
        horizon: Prediction horizon

    Returns:
        Prediction results
    """
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model, _ = ModelFactory.create_model(config['model']).load_checkpoint(model_path, device)
    model.eval()

    # Prepare input
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)  # Add batch dimension
    params_tensor = torch.FloatTensor(parameters).unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        # Single step prediction
        outputs = model(input_tensor, params_tensor)

        # Multi-step trajectory prediction
        trajectory = model.predict_trajectory(input_tensor, params_tensor, horizon)

        # Uncertainty estimation (if supported)
        if hasattr(model, 'predict_with_uncertainty'):
            trajectory_mean, trajectory_std = model.predict_with_uncertainty(
                input_tensor, params_tensor, horizon, n_samples=10
            )
        else:
            trajectory_mean = trajectory
            trajectory_std = None

    results = {
        'next_step': outputs['next_step'].cpu().numpy().squeeze(),
        'trajectory': trajectory.cpu().numpy().squeeze(),
        'amplitude': outputs.get('amplitude', torch.tensor([0])).cpu().numpy().squeeze(),
        'attention_weights': outputs.get('attention_weights', torch.tensor([])).cpu().numpy().squeeze()
    }

    if trajectory_std is not None:
        results['trajectory_uncertainty'] = trajectory_std.cpu().numpy().squeeze()

    return results


def main():
    """
    Main function with command line interface.
    """
    parser = argparse.ArgumentParser(description='Vibration Prediction System')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'],
                       default='train', help='Operation mode')
    parser.add_argument('--config', default='project/config/base_config.yaml',
                       help='Path to base configuration file')
    parser.add_argument('--model-config', default='project/config/lstm_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--model-path', help='Path to trained model (for evaluate/predict modes)')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use')

    args = parser.parse_args()

    # Load configurations
    base_config = load_config(args.config)
    model_config = load_config(args.model_config)
    config = merge_configs(base_config, model_config)

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"Running in {args.mode} mode on {device}")

    try:
        if args.mode == 'train':
            results = train_model(config, args.output_dir)
            logger.info(f"Training completed successfully. Best validation loss: {results['best_val_loss']:.6f}")

        elif args.mode == 'evaluate':
            if not args.model_path:
                raise ValueError("Model path required for evaluation mode")

            results = evaluate_model(args.model_path, config, args.output_dir)
            logger.info("Evaluation completed successfully")

        elif args.mode == 'predict':
            if not args.model_path:
                raise ValueError("Model path required for prediction mode")

            # Example prediction (you would load your own data here)
            logger.info("Running example prediction...")

            # Generate example input
            np.random.seed(42)
            input_data = np.random.randn(config['data']['window_size'], 2)
            parameters = np.random.randn(15)  # 15 system parameters

            results = predict_trajectory(
                args.model_path,
                input_data,
                parameters,
                config,
                horizon=config['data']['prediction_horizon']
            )

            logger.info(f"Prediction completed. Predicted amplitude: {results['amplitude']:.4f}")

            # Save results
            output_path = os.path.join(args.output_dir, 'prediction_results.npz')
            np.savez(output_path, **results)
            logger.info(f"Prediction results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}")
        raise


if __name__ == '__main__':
    main()
