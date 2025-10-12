"""
Unit tests for signal processing utilities.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.signal_processing import (
    compute_envelope, detect_bifurcation_points, compute_instantaneous_frequency,
    compute_spectral_features, compute_phase_space_features, compute_nonlinear_features,
    filter_signal, resample_signal
)


class TestSignalProcessing(unittest.TestCase):
    """Test signal processing functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test signals
        self.fs = 100.0
        self.t = np.linspace(0, 10, int(10 * self.fs))

        # Sinusoidal signal
        self.sine_signal = np.sin(2 * np.pi * 5 * self.t)

        # Amplitude modulated signal
        self.am_signal = (1 + 0.5 * np.sin(2 * np.pi * 1 * self.t)) * np.sin(2 * np.pi * 10 * self.t)

        # Noisy signal
        np.random.seed(42)
        self.noisy_signal = self.sine_signal + 0.1 * np.random.randn(len(self.t))

        # Position and velocity for phase space
        self.x = self.sine_signal
        self.x_dot = np.gradient(self.x, self.t[1] - self.t[0])

    def test_compute_envelope_hilbert(self):
        """Test envelope computation using Hilbert transform."""
        envelope = compute_envelope(self.am_signal, method='hilbert')

        self.assertEqual(len(envelope), len(self.am_signal))
        self.assertTrue(np.all(envelope >= 0))

        # Envelope should be larger than or equal to signal magnitude
        self.assertTrue(np.all(envelope >= np.abs(self.am_signal)))

    def test_compute_envelope_peak(self):
        """Test envelope computation using peak detection."""
        envelope = compute_envelope(self.am_signal, method='peak')

        self.assertEqual(len(envelope), len(self.am_signal))
        self.assertTrue(np.all(envelope >= 0))

    def test_compute_envelope_rms(self):
        """Test envelope computation using RMS."""
        envelope = compute_envelope(self.am_signal, method='rms')

        self.assertEqual(len(envelope), len(self.am_signal))
        self.assertTrue(np.all(envelope >= 0))

    def test_detect_bifurcation_points(self):
        """Test bifurcation point detection."""
        # Create signal with amplitude jumps
        signal_with_jumps = np.concatenate([
            np.sin(2 * np.pi * 5 * self.t[:300]) * 1.0,
            np.sin(2 * np.pi * 5 * self.t[300:600]) * 3.0,  # Amplitude jump
            np.sin(2 * np.pi * 5 * self.t[600:]) * 1.0
        ])

        bifurcation_points, amplitudes = detect_bifurcation_points(
            signal_with_jumps, threshold_factor=2.0
        )

        self.assertIsInstance(bifurcation_points, np.ndarray)
        self.assertIsInstance(amplitudes, np.ndarray)
        self.assertEqual(len(bifurcation_points), len(amplitudes))

        # Should detect the high amplitude region
        if len(bifurcation_points) > 0:
            self.assertTrue(np.any(bifurcation_points > 300))
            self.assertTrue(np.any(bifurcation_points < 600))

    def test_compute_instantaneous_frequency(self):
        """Test instantaneous frequency computation."""
        inst_freq = compute_instantaneous_frequency(self.sine_signal, fs=self.fs)

        self.assertEqual(len(inst_freq), len(self.sine_signal))

        # For a 5 Hz sine wave, instantaneous frequency should be around 5 Hz
        mean_freq = np.mean(inst_freq[100:-100])  # Exclude edges
        self.assertAlmostEqual(mean_freq, 5.0, delta=1.0)

    def test_compute_spectral_features(self):
        """Test spectral features computation."""
        features = compute_spectral_features(self.sine_signal, fs=self.fs)

        # Check required keys
        required_keys = [
            'spectral_centroid', 'spectral_spread', 'spectral_rolloff',
            'spectral_flux', 'dominant_frequency', 'total_power',
            'spectral_entropy', 'frequencies', 'psd'
        ]

        for key in required_keys:
            self.assertIn(key, features)

        # For a 5 Hz sine wave, dominant frequency should be around 5 Hz
        self.assertAlmostEqual(features['dominant_frequency'], 5.0, delta=1.0)

        # Total power should be positive
        self.assertGreater(features['total_power'], 0)

        # Spectral entropy should be reasonable
        self.assertGreater(features['spectral_entropy'], 0)

    def test_compute_phase_space_features(self):
        """Test phase space features computation."""
        features = compute_phase_space_features(self.x, self.x_dot)

        required_keys = [
            'phase_area', 'phase_diameter', 'mean_energy',
            'energy_variance', 'trajectory'
        ]

        for key in required_keys:
            self.assertIn(key, features)

        # All features should be non-negative
        self.assertGreaterEqual(features['phase_area'], 0)
        self.assertGreaterEqual(features['phase_diameter'], 0)
        self.assertGreaterEqual(features['energy_variance'], 0)

        # Trajectory should have correct shape
        self.assertEqual(features['trajectory'].shape, (len(self.x), 2))

    def test_compute_nonlinear_features(self):
        """Test nonlinear dynamics features computation."""
        features = compute_nonlinear_features(self.sine_signal)

        required_keys = [
            'lyapunov_exponent', 'hurst_exponent',
            'sample_entropy', 'dfa_alpha'
        ]

        for key in required_keys:
            self.assertIn(key, features)

        # Hurst exponent should be between 0 and 1
        self.assertGreaterEqual(features['hurst_exponent'], 0)
        self.assertLessEqual(features['hurst_exponent'], 1)

        # Sample entropy should be non-negative
        self.assertGreaterEqual(features['sample_entropy'], 0)

    def test_filter_signal_lowpass(self):
        """Test lowpass filtering."""
        filtered = filter_signal(
            self.noisy_signal,
            filter_type='lowpass',
            cutoff=10.0,
            fs=self.fs
        )

        self.assertEqual(len(filtered), len(self.noisy_signal))

        # Filtered signal should be smoother (less high-frequency content)
        filtered_std = np.std(np.diff(filtered))
        original_std = np.std(np.diff(self.noisy_signal))
        self.assertLess(filtered_std, original_std)

    def test_filter_signal_highpass(self):
        """Test highpass filtering."""
        # Add low-frequency component
        signal_with_dc = self.sine_signal + 2.0

        filtered = filter_signal(
            signal_with_dc,
            filter_type='highpass',
            cutoff=1.0,
            fs=self.fs
        )

        self.assertEqual(len(filtered), len(signal_with_dc))

        # DC component should be removed
        self.assertLess(np.abs(np.mean(filtered)), 0.1)

    def test_filter_signal_bandpass(self):
        """Test bandpass filtering."""
        # Create signal with multiple frequencies
        multi_freq_signal = (np.sin(2 * np.pi * 2 * self.t) +
                           np.sin(2 * np.pi * 10 * self.t) +
                           np.sin(2 * np.pi * 50 * self.t))

        filtered = filter_signal(
            multi_freq_signal,
            filter_type='bandpass',
            cutoff=[5.0, 15.0],
            fs=self.fs
        )

        self.assertEqual(len(filtered), len(multi_freq_signal))

        # Should preserve 10 Hz component, attenuate others
        features_orig = compute_spectral_features(multi_freq_signal, fs=self.fs)
        features_filt = compute_spectral_features(filtered, fs=self.fs)

        # Dominant frequency should be around 10 Hz after filtering
        self.assertAlmostEqual(features_filt['dominant_frequency'], 10.0, delta=2.0)

    def test_resample_signal_scipy(self):
        """Test signal resampling using scipy method."""
        target_fs = 50.0  # Downsample from 100 to 50 Hz

        resampled = resample_signal(
            self.sine_signal,
            original_fs=self.fs,
            target_fs=target_fs,
            method='scipy'
        )

        expected_length = int(len(self.sine_signal) * target_fs / self.fs)
        self.assertAlmostEqual(len(resampled), expected_length, delta=2)

    def test_resample_signal_decimate(self):
        """Test signal resampling using decimation."""
        target_fs = 50.0  # Downsample from 100 to 50 Hz

        resampled = resample_signal(
            self.sine_signal,
            original_fs=self.fs,
            target_fs=target_fs,
            method='decimate'
        )

        # Decimation by factor of 2
        expected_length = len(self.sine_signal) // 2
        self.assertAlmostEqual(len(resampled), expected_length, delta=5)

    def test_resample_signal_interpolate(self):
        """Test signal resampling using interpolation."""
        target_fs = 50.0  # Downsample from 100 to 50 Hz

        resampled = resample_signal(
            self.sine_signal,
            original_fs=self.fs,
            target_fs=target_fs,
            method='interpolate'
        )

        # Should preserve signal characteristics
        self.assertGreater(len(resampled), 0)

        # Check that resampled signal maintains frequency content
        features_orig = compute_spectral_features(self.sine_signal, fs=self.fs)
        features_resampled = compute_spectral_features(resampled, fs=target_fs)

        # Dominant frequency should be preserved
        self.assertAlmostEqual(
            features_orig['dominant_frequency'],
            features_resampled['dominant_frequency'],
            delta=1.0
        )

    def test_invalid_filter_type(self):
        """Test invalid filter type."""
        with self.assertRaises(ValueError):
            filter_signal(self.sine_signal, filter_type='invalid')

    def test_invalid_resampling_method(self):
        """Test invalid resampling method."""
        with self.assertRaises(ValueError):
            resample_signal(self.sine_signal, 100.0, 50.0, method='invalid')


if __name__ == '__main__':
    unittest.main()
