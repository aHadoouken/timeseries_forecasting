"""
VibrationDataset class for handling vibration prediction data.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataResampler:
    """
    Handles data resampling with various strategies.
    """

    def __init__(self, target_sampling_rate: Optional[float] = None, adaptive: bool = False):
        self.target_sampling_rate = target_sampling_rate
        self.adaptive = adaptive

    def resample(self, t: np.ndarray, x: np.ndarray, x_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample time series data.

        Args:
            t: Time array
            x: Position array
            x_dot: Velocity array

        Returns:
            Resampled (t, x, x_dot) arrays
        """
        if self.target_sampling_rate is None:
            return t, x, x_dot

        original_rate = 1.0 / np.mean(np.diff(t))

        if abs(original_rate - self.target_sampling_rate) < 1e-6:
            return t, x, x_dot

        upsample_factor = self.target_sampling_rate / original_rate
        t_resampled = np.linspace(t[0], t[-1], int(len(t) * upsample_factor))
        x_resampled = np.interp(t_resampled, t, x)
        x_dot_resampled = np.interp(t_resampled, t, x_dot)

        return t_resampled, x_resampled, x_dot_resampled

    def adaptive_resample(self, t: np.ndarray, x: np.ndarray, x_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Adaptive resampling based on signal characteristics.
        """
        # Calculate local variation
        dx_dt = np.gradient(x)
        variation = np.abs(dx_dt)

        # Determine sampling density based on variation
        high_variation_mask = variation > np.percentile(variation, 75)

        # Create adaptive time grid
        dense_indices = np.where(high_variation_mask)[0]
        sparse_indices = np.where(~high_variation_mask)[0][::2]  # Every other point

        selected_indices = np.sort(np.concatenate([dense_indices, sparse_indices]))

        return t[selected_indices], x[selected_indices], x_dot[selected_indices]


class VibrationDataset(Dataset):
    """
    Dataset class for vibration prediction.
    """

    def __init__(
        self,
        data_path: str,
        window_size: int = 500,
        prediction_horizon: int = 100,
        stride: int = 50,
        split: str = 'train',
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        sampling_rate: Optional[float] = None,
        features: List[str] = ['basic'],
        normalization: str = 'z_score',
        temporal_split: bool = False,
        exclude_params: Optional[List[Dict]] = None,
        random_seed: int = 42
    ):
        """
        Initialize VibrationDataset.

        Args:
            data_path: Path to the dataset pickle file
            window_size: Size of input window
            prediction_horizon: Number of steps to predict
            stride: Stride for sliding window
            split: 'train', 'val', or 'test'
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
            sampling_rate: Target sampling rate (Hz)
            features: List of feature types to use
            normalization: 'z_score' or 'min_max'
            temporal_split: Use temporal splitting instead of random
            exclude_params: Parameter combinations to exclude from training
            random_seed: Random seed for reproducibility
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.split = split
        self.features = features
        self.normalization = normalization
        self.temporal_split = temporal_split
        self.random_seed = random_seed

        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Load data
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.trajectories = data['trajectories']
        self.config = data.get('config', {})

        # Initialize resampler
        self.resampler = DataResampler(target_sampling_rate=sampling_rate)

        # Split data
        self.train_trajectories, self.val_trajectories, self.test_trajectories = self._split_data(
            train_split, val_split, test_split, exclude_params
        )

        # Get current split data
        if split == 'train':
            self.current_trajectories = self.train_trajectories
        elif split == 'val':
            self.current_trajectories = self.val_trajectories
        else:
            self.current_trajectories = self.test_trajectories

        # Create sequences
        self.sequences = self._create_sequences()

        # Initialize scalers
        self.scalers = {}
        self._fit_scalers()

        logger.info(f"Dataset initialized: {len(self.sequences)} sequences for {split} split")

    def _split_data(
        self,
        train_split: float,
        val_split: float,
        test_split: float,
        exclude_params: Optional[List[Dict]] = None
    ) -> Tuple[List, List, List]:
        """
        Split trajectories into train/val/test sets.
        """
        trajectories = self.trajectories.copy()

        train_trajectories, temp_trajectories = train_test_split(
            trajectories,
            test_size=(1 - train_split),
            random_state=self.random_seed
        )
        val_size = val_split / (val_split + test_split)
        val_trajectories, test_trajectories = train_test_split(
            temp_trajectories,
            test_size=(1 - val_size),
            random_state=self.random_seed
        )

        logger.info(f"Data split: {len(train_trajectories)} train, {len(val_trajectories)} val, {len(test_trajectories)} test")
        return train_trajectories, val_trajectories, test_trajectories

    def _create_sub_trajectory(self, traj: Dict, start_idx: int, end_idx: int) -> Dict:
        """
        Create a sub-trajectory from a larger trajectory.
        """
        sub_traj = {
            'id': f"{traj['id']}_{start_idx}_{end_idx}",
            'parameters': traj['parameters'].copy(),
            'solution': {
                't': traj['solution']['t'][start_idx:end_idx],
                'x': traj['solution']['x'][start_idx:end_idx],
                'x_dot': traj['solution']['x_dot'][start_idx:end_idx]
            },
            'metadata': traj['metadata'].copy()  # Keep original metadata
        }
        return sub_traj

    def _create_sequences(self) -> List[Dict]:
        """
        Create sliding window sequences from trajectories.
        """
        sequences = []

        for traj in self.current_trajectories:
            # Resample if needed
            t, x, x_dot = self.resampler.resample(
                traj['solution']['t'],
                traj['solution']['x'],
                traj['solution']['x_dot']
            )

            # Create sliding windows
            total_length = len(t)
            max_start = total_length - self.window_size - self.prediction_horizon

            if max_start <= 0:
                logger.warning(f"Trajectory {traj['id']} too short for window size")
                continue

            for start_idx in range(0, max_start, self.stride):
                end_idx = start_idx + self.window_size
                pred_end_idx = end_idx + self.prediction_horizon

                sequence = {
                    'trajectory_id': traj['id'],
                    'parameters': traj['parameters'],
                    'input_t': t[start_idx:end_idx],
                    'input_x': x[start_idx:end_idx],
                    'input_x_dot': x_dot[start_idx:end_idx],
                    'target_t': t[end_idx:pred_end_idx],
                    'target_x': x[end_idx:pred_end_idx],
                    'target_x_dot': x_dot[end_idx:pred_end_idx],
                    'metadata': traj['metadata']
                }
                sequences.append(sequence)

        return sequences

    def _fit_scalers(self):
        """
        Fit normalization scalers on training data.
        """
        if self.split != 'train':
            return

        # Collect all training data
        all_x = []
        all_x_dot = []
        all_params = []

        for seq in self.sequences:
            all_x.extend(seq['input_x'])
            all_x_dot.extend(seq['input_x_dot'])
            all_params.append(list(seq['parameters'].values()))

        all_x = np.array(all_x).reshape(-1, 1)
        all_x_dot = np.array(all_x_dot).reshape(-1, 1)
        all_params = np.array(all_params)

        # Fit scalers
        if self.normalization == 'z_score':
            self.scalers['x'] = StandardScaler().fit(all_x)
            self.scalers['x_dot'] = StandardScaler().fit(all_x_dot)
            self.scalers['params'] = StandardScaler().fit(all_params)
        else:
            self.scalers['x'] = MinMaxScaler().fit(all_x)
            self.scalers['x_dot'] = MinMaxScaler().fit(all_x_dot)
            self.scalers['params'] = MinMaxScaler().fit(all_params)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.
        """
        seq = self.sequences[idx]

        # Prepare input features
        input_x = torch.FloatTensor(seq['input_x'])
        input_x_dot = torch.FloatTensor(seq['input_x_dot'])

        # Normalize if scalers are available
        if 'x' in self.scalers:
            input_x = torch.FloatTensor(
                self.scalers['x'].transform(input_x.numpy().reshape(-1, 1)).flatten()
            )
            input_x_dot = torch.FloatTensor(
                self.scalers['x_dot'].transform(input_x_dot.numpy().reshape(-1, 1)).flatten()
            )

        # Stack features based on configuration
        if 'basic' in self.features:
            features = torch.stack([input_x, input_x_dot], dim=1)  # [seq_len, 2]
        else:
            features = torch.stack([input_x, input_x_dot], dim=1)

        # Parameters
        params = torch.FloatTensor(list(seq['parameters'].values()))
        if 'params' in self.scalers:
            params = torch.FloatTensor(
                self.scalers['params'].transform(params.numpy().reshape(1, -1)).flatten()
            )

        # Targets
        target_x = torch.FloatTensor(seq['target_x'])
        target_x_dot = torch.FloatTensor(seq['target_x_dot'])

        if 'x' in self.scalers:
            target_x = torch.FloatTensor(
                self.scalers['x'].transform(target_x.numpy().reshape(-1, 1)).flatten()
            )
            target_x_dot = torch.FloatTensor(
                self.scalers['x_dot'].transform(target_x_dot.numpy().reshape(-1, 1)).flatten()
            )

        targets = torch.stack([target_x, target_x_dot], dim=1)  # [pred_horizon, 2]

        return {
            'features': features,
            # 'parameters': params,
            'targets': targets,
            # 'trajectory_id': seq['trajectory_id'],
            # 'max_amplitude': torch.FloatTensor([seq['metadata']['max_amplitude']])
        }

    def get_scalers(self) -> Dict:
        """
        Get fitted scalers for use in other splits.
        """
        return self.scalers

    def set_scalers(self, scalers: Dict):
        """
        Set scalers from training split.
        """
        self.scalers = scalers


def create_dataloaders(
    config: Dict,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    """
    # Create datasets
    train_dataset = VibrationDataset(split='train', **config)
    val_dataset = VibrationDataset(split='val', **config)
    test_dataset = VibrationDataset(split='test', **config)

    # Share scalers
    scalers = train_dataset.get_scalers()
    val_dataset.set_scalers(scalers)
    test_dataset.set_scalers(scalers)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
