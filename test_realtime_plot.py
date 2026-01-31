"""
Test script to demonstrate real-time loss plotting during training.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_predictor import LSTMPredictor
from training.trainer import Trainer
import yaml

# Create dummy data for testing
def create_dummy_data(n_samples=100, seq_len=50, horizon_len=20, n_features=2):
    """Create dummy vibration data for testing."""
    # Generate synthetic vibration data
    features = torch.randn(n_samples, seq_len, n_features)
    targets = torch.randn(n_samples, horizon_len, n_features)
    parameters = torch.randn(n_samples, 3)  # 3 parameters

    # Create dataset
    dataset = TensorDataset(features, targets, parameters)

    # Create data dictionary for each batch
    def collate_fn(batch):
        features_batch = torch.stack([b[0] for b in batch])
        targets_batch = torch.stack([b[1] for b in batch])
        parameters_batch = torch.stack([b[2] for b in batch])

        return {
            'features': features_batch,
            'targets': targets_batch,
            'parameters': parameters_batch
        }

    return dataset, collate_fn

# Load configuration
with open('config/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update config for testing with real-time plotting
config['training']['epochs'] = 20  # Fewer epochs for testing
config['training']['plot_realtime'] = True  # Enable real-time plotting
config['training']['plot_update_interval'] = 1  # Update plot every epoch
config['training']['log_interval'] = 5
config['training']['save_interval'] = 10
config['training']['output_dir'] = 'test_outputs'

# Create dummy datasets
train_dataset, collate_fn = create_dummy_data(n_samples=200)
val_dataset, _ = create_dummy_data(n_samples=50)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

# Initialize model
model_config = config['model']
model = LSTMPredictor(
    input_dim=2,
    hidden_dim=model_config['hidden_dim'],
    num_layers=model_config['num_layers'],
    output_dim=2,
    dropout=model_config['dropout'],
    bidirectional=model_config.get('bidirectional', False),
    attention_heads=model_config.get('attention_heads', 4),
    use_layer_norm=model_config.get('use_layer_norm', True),
    parameter_dim=3,
    horizon_len=20
)

# Initialize trainer with real-time plotting enabled
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config['training'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("=" * 60)
print("REAL-TIME LOSS PLOTTING TEST")
print("=" * 60)
print("Starting training with real-time loss visualization...")
print("You should see a plot window that updates automatically during training.")
print("The plot shows:")
print("  - Training and Validation Loss curves")
print("  - Learning Rate over time")
print("  - Teacher Forcing Ratio (if scheduled sampling is enabled)")
print("  - Loss Difference (Train - Val) to monitor overfitting")
print("-" * 60)

# Run training with real-time plotting
results = trainer.train()

print("\n" + "=" * 60)
print("Training completed!")
print(f"Best validation loss: {results['best_val_loss']:.6f}")
print(f"Total epochs: {results['total_epochs']}")
print(f"Training time: {results['training_time']:.2f} seconds")
print("=" * 60)
