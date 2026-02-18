"""
VibrationDataset class for handling vibration prediction data.
"""

import pickle
import numpy as np
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from scipy.ndimage import maximum_filter1d
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

def load_and_split_data(
    config: Dict,
) -> Tuple[List, List, List]:
    """
    Split trajectories into train/val/test sets.
    """
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)

    trajectories = data['trajectories']

    train_trajectories, temp_trajectories = train_test_split(
        trajectories,
        test_size=(1 - config['train_split']),
        random_state=config['random_seed']
    )
    val_size = config['val_split'] / (config['val_split'] + config['test_split'])
    val_trajectories, test_trajectories = train_test_split(
        temp_trajectories,
        test_size=(1 - val_size),
        random_state=config['random_seed']
    )

    return train_trajectories, val_trajectories, test_trajectories


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
        # print("Original rate: {}".format(original_rate))

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
        trajectories: list,
        window_size: int = 500,
        prediction_horizon: int = 100,
        stride: int = 50,
        split: str = 'train',
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        sampling_rate: Optional[float] = None,
        # features: List[str] = ['basic'],
        normalization: str = None,
        exclude_params: Optional[List[Dict]] = None,
        random_seed: int = 42,
        max_samples_for_scaling=5000,
        rm_start_t=0,
        window_normalization=None,
        calc_max_diff=False,
        balance_by_diff=False,
        diff_quantiles=[],
        balance_type: str = 'quantile',
        **kwargs
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
        self.normalization = normalization
        self.random_seed = random_seed
        self.max_samples_for_scaling = max_samples_for_scaling
        self.rm_start_t = rm_start_t
        self.sampling_rate = sampling_rate
        self.window_normalization = window_normalization
        self.calc_max_diff = calc_max_diff
        self.balance_by_diff = balance_by_diff
        self.diff_quantiles = diff_quantiles
        self.balance_type = balance_type

        # Start timing
        start_time = time.time()

        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.trajectories = trajectories
        # self.config = data.get('config', {})

        # Initialize resampler
        self.resampler = DataResampler(target_sampling_rate=sampling_rate)

        # Create sequences
        seq_start = time.time()
        self.sequences = self._create_sequences()
        print(f"Sequence creation took {time.time() - seq_start:.2f} seconds")
        print(f"{len(self.sequences)} sequences created")

        if self.balance_by_diff:
            self.make_balance_by_diff()  # uses balance_type: 'quantile' or 'equal_width'

        # Initialize scalers
        # scaler_start = time.time()
        # self.scalers = {}
        # self._fit_scalers()
        # print(f"Scaler fitting took {time.time() - scaler_start:.2f} seconds")

        total_time = time.time() - start_time
        print(f"Dataset initialized: {len(self.sequences)} sequences in {total_time:.2f} seconds")

    def make_balance_by_diff(self):
        """
        Balance self.sequences by max_diff. Total size unchanged; resample with replacement.

        - balance_type='quantile': bins from diff_quantiles (value-based, merged if tied).
          Each bin contributes equal count; histogram with bins=_diff_bounds is even.

        - balance_type='equal_width': bins = N equal-width intervals in [min(max_diff), max(max_diff)].
          Sampling probability per sequence: p[seq in bin i] = 1/(N_bins * n_i), so each bin
          is drawn equally often (inverse frequency). N_bins = len(diff_quantiles) - 1.

        If calc_max_diff is False, does nothing.
        """
        if not self.diff_quantiles or not self.calc_max_diff:
            return

        rng = np.random.default_rng(self.random_seed)
        diffs = np.array([s['max_diff'] for s in self.sequences])
        n = len(self.sequences)
        n_bins = len(self.diff_quantiles) - 1
        if n_bins < 1:
            return

        if self.balance_type == 'equal_width':
            # [min, max] split into N equal-width intervals
            d_min, d_max = diffs.min(), diffs.max()
            if d_max <= d_min:
                return
            bounds = np.linspace(d_min, d_max, n_bins + 1)
            bounds[0], bounds[-1] = d_min, d_max  # avoid float noise
            # Assign each sequence to a bin: [lo, hi) except last [lo, hi]
            bin_id = np.zeros(n, dtype=np.intp)
            for i in range(n_bins):
                lo, hi = bounds[i], bounds[i + 1]
                if i < n_bins - 1:
                    mask = (diffs >= lo) & (diffs < hi)
                else:
                    mask = (diffs >= lo) & (diffs <= hi)
                bin_id[mask] = i
            # p[seq] = 1/(N_bins * n_i) so each bin has total probability 1/N_bins
            n_per_bin = np.bincount(bin_id, minlength=n_bins)
            p = np.zeros(n)
            for i in range(n_bins):
                if n_per_bin[i] > 0:
                    p[bin_id == i] = 1.0 / (n_bins * n_per_bin[i])
            p = p / p.sum()
            chosen = rng.choice(np.arange(n), size=n, replace=True, p=p)
            self.sequences = [self.sequences[i] for i in chosen]
            self._diff_bounds = bounds.tolist()
            return

        # balance_type == 'quantile'
        q_bounds = np.quantile(diffs, self.diff_quantiles)
        effective_bounds = [float(q_bounds[0])]
        for i in range(1, len(q_bounds)):
            if q_bounds[i] > effective_bounds[-1]:
                effective_bounds.append(float(q_bounds[i]))
        n_bins = len(effective_bounds) - 1
        if n_bins < 1:
            return

        bin_indices = []
        for i in range(n_bins):
            lo, hi = effective_bounds[i], effective_bounds[i + 1]
            if i < n_bins - 1:
                mask = (diffs >= lo) & (diffs < hi)
            else:
                mask = (diffs >= lo) & (diffs <= hi)
            bin_indices.append(np.where(mask)[0])

        base_count = n // n_bins
        remainder = n % n_bins
        counts = [base_count + (1 if i < remainder else 0) for i in range(n_bins)]

        non_empty = [i for i in range(n_bins) if len(bin_indices[i]) > 0]
        if not non_empty:
            return
        extra = sum(counts[i] for i in range(n_bins) if i not in non_empty)
        for i in range(n_bins):
            if i not in non_empty:
                counts[i] = 0
        if extra > 0:
            counts[non_empty[0]] += extra

        sampled = []
        for idx_list, count in zip(bin_indices, counts):
            if count == 0:
                continue
            chosen = rng.choice(idx_list, size=count, replace=True)
            sampled.append(chosen)
        if not sampled:
            return
        all_indices = np.concatenate(sampled)
        rng.shuffle(all_indices)

        self.sequences = [self.sequences[i] for i in all_indices]
        self._diff_bounds = effective_bounds

        return self.sequences

    def _create_sequences(self) -> List[Dict]:
        """
        Create sliding window sequences from trajectories.
        """
        sequences = []

        for traj in tqdm(self.trajectories):
            # Resample if needed
            t, x, x_dot = self.resampler.resample(
                traj['solution']['t'],
                traj['solution']['x'],
                traj['solution']['x_dot']
            )
            rm_start_idxs = int(self.rm_start_t * self.sampling_rate)
            t = t[rm_start_idxs:]
            x = x[rm_start_idxs:]
            x_dot = x_dot[rm_start_idxs:]

            # Create sliding windows
            total_length = len(t)
            max_start = total_length - self.window_size - self.prediction_horizon

            if max_start <= 0:
                logger.warning(f"Trajectory {traj['id']} too short for window size")
                continue

            for start_idx in range(0, max_start, self.stride):
                end_idx = start_idx + self.window_size
                pred_end_idx = end_idx + self.prediction_horizon

                input_x_mean = np.mean(x[start_idx:end_idx])
                input_x_dot_mean = np.mean(x_dot[start_idx:end_idx])
                input_x_std = np.std(x[start_idx:end_idx])
                input_x_dot_std = np.std(x_dot[start_idx:end_idx])

                if self.window_normalization:
                    input_x_norm = (x[start_idx:end_idx] - input_x_mean) / input_x_std
                    input_x_dot_norm = (x_dot[start_idx:end_idx] - input_x_dot_mean) / input_x_dot_std
                    target_x_norm = (x[end_idx:pred_end_idx] - input_x_mean) / input_x_std
                    target_x_dot_norm = (x_dot[end_idx:pred_end_idx] - input_x_dot_mean) / input_x_dot_std
                else:
                    input_x_norm = x[start_idx:end_idx]
                    input_x_dot_norm = x_dot[start_idx:end_idx]
                    target_x_norm = x[end_idx:pred_end_idx]
                    target_x_dot_norm = x_dot[end_idx:pred_end_idx]

                if self.calc_max_diff:
                    all_x_norm = np.concatenate([input_x_norm, target_x_norm], axis=0)
                    envelope = maximum_filter1d(all_x_norm, size=400)
                    diff = (np.max(envelope) - np.min(envelope)) / (np.max(envelope))
                else:
                    diff = 0

                sequence = {
                    'trajectory_id': traj['id'],
                    'parameters': traj['parameters'],
                    'input_t': t[start_idx:end_idx],
                    'input_x': x[start_idx:end_idx],
                    'input_x_dot': x_dot[start_idx:end_idx],
                    'target_t': t[end_idx:pred_end_idx],
                    'target_x': x[end_idx:pred_end_idx],
                    'target_x_dot': x_dot[end_idx:pred_end_idx],
                    'metadata': traj['metadata'],
                    'input_x_mean': input_x_mean,
                    'input_x_dot_mean': input_x_dot_mean,
                    'input_x_std': input_x_std,
                    'input_x_dot_std': input_x_dot_std,
                    'input_x_norm': input_x_norm,
                    'input_x_dot_norm': input_x_dot_norm,
                    'target_x_norm': target_x_norm,
                    'target_x_dot_norm': target_x_dot_norm,
                    'max_diff': diff,
                }
                sequences.append(sequence)

        return sequences

    def _fit_scalers(self):
        """
        Fit normalization scalers on training data.
        """
        if self.split != 'train':
            return

        # Time data collection
        collect_start = time.time()

        # Vectorized data collection
        sequences = np.random.choice(
            self.sequences,
            size=self.max_samples_for_scaling,
            replace=False
        )

        all_x = np.concatenate([seq['input_x'] for seq in sequences])
        all_x_dot = np.concatenate([seq['input_x_dot'] for seq in sequences])
        # Reshape for scalers
        all_x = all_x.reshape(-1, 1)
        all_x_dot = all_x_dot.reshape(-1, 1)

        print(f"Vectorized data collection took {time.time() - collect_start:.2f} seconds")
        print(f"Total points: {len(all_x)}")

        # Time scaler fitting
        fit_start = time.time()
        if self.normalization is None:
            self.scalers['x'] = FunctionTransformer(lambda x: x)
            self.scalers['x_dot'] = FunctionTransformer(lambda x: x)
        elif self.normalization == 'z_score':
            self.scalers['x'] = StandardScaler().fit(all_x)
            self.scalers['x_dot'] = StandardScaler().fit(all_x_dot)
        else:
            self.scalers['x'] = MinMaxScaler((-1, 1)).fit(all_x)
            self.scalers['x_dot'] = MinMaxScaler((-1, 1)).fit(all_x_dot)
        print(f"Scaler fitting took {time.time() - fit_start:.2f} seconds")

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

        input_x_norm = torch.FloatTensor(seq['input_x_norm'])
        input_x_dot_norm = torch.FloatTensor(seq['input_x_dot_norm'])

        # Normalize if scalers are available
        # if 'x' in self.scalers:
        #     input_x = torch.FloatTensor(
        #         self.scalers['x'].transform(input_x.numpy().reshape(-1, 1)).flatten()
        #     )
        #     input_x_dot = torch.FloatTensor(
        #         self.scalers['x_dot'].transform(input_x_dot.numpy().reshape(-1, 1)).flatten()
        #     )

        # Stack features based on configuration
        features = torch.stack([input_x, input_x_dot], dim=1)
        features_norm = torch.stack([input_x_norm, input_x_dot_norm], dim=1)

        # Parameters
        # params = torch.FloatTensor(list(seq['parameters'].values()))
        # if 'params' in self.scalers:
        #     params = torch.FloatTensor(
        #         self.scalers['params'].transform(params.numpy().reshape(1, -1)).flatten()
        #     )

        # Targets
        target_x = torch.FloatTensor(seq['target_x'])
        target_x_dot = torch.FloatTensor(seq['target_x_dot'])

        target_x_norm = torch.FloatTensor(seq['target_x_norm'])
        target_x_dot_norm = torch.FloatTensor(seq['target_x_dot_norm'])

        # if 'x' in self.scalers:
        #     target_x = torch.FloatTensor(
        #         self.scalers['x'].transform(target_x.numpy().reshape(-1, 1)).flatten()
        #     )
        #     target_x_dot = torch.FloatTensor(
        #         self.scalers['x_dot'].transform(target_x_dot.numpy().reshape(-1, 1)).flatten()
        #     )

        targets = torch.stack([target_x, target_x_dot], dim=1)  # [pred_horizon, 2]
        targets_norm = torch.stack([target_x_norm, target_x_dot_norm], dim=1)

        return {
            'features': features,
            'features_norm': features_norm,
            # 'parameters': params,
            'targets': targets,
            'targets_norm': targets_norm,
            'diff': seq['max_diff'],
            'trajectory_id': seq['trajectory_id'],
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
    dataset: VibrationDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader
