"""
Signal processing utilities for vibration data analysis.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def compute_envelope(x: np.ndarray, method: str = 'hilbert') -> np.ndarray:
    """
    Compute the envelope of a signal.

    Args:
        x: Input signal
        method: Method to use ('hilbert', 'peak', 'rms')

    Returns:
        Signal envelope
    """
    if method == 'hilbert':
        analytic_signal = signal.hilbert(x)
        envelope = np.abs(analytic_signal)
    elif method == 'peak':
        # Peak envelope using local maxima
        peaks, _ = signal.find_peaks(np.abs(x))
        if len(peaks) > 1:
            envelope_interp = interp1d(peaks, np.abs(x[peaks]),
                                     kind='cubic', fill_value='extrapolate')
            envelope = envelope_interp(np.arange(len(x)))
        else:
            envelope = np.abs(x)
    elif method == 'rms':
        # RMS envelope using sliding window
        window_size = max(10, len(x) // 100)
        envelope = np.zeros_like(x)
        for i in range(len(x)):
            start = max(0, i - window_size // 2)
            end = min(len(x), i + window_size // 2)
            envelope[i] = np.sqrt(np.mean(x[start:end] ** 2))
    else:
        raise ValueError(f"Unknown envelope method: {method}")

    return envelope


def detect_bifurcation_points(
    x: np.ndarray,
    threshold_factor: float = 2.0,
    min_distance: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect potential bifurcation points in a time series.

    Args:
        x: Input signal
        threshold_factor: Factor above mean amplitude to consider bifurcation
        min_distance: Minimum distance between bifurcation points

    Returns:
        Bifurcation indices and amplitudes
    """
    # Compute envelope
    envelope = compute_envelope(x, method='hilbert')

    # Smooth envelope
    envelope_smooth = signal.savgol_filter(envelope,
                                         window_length=min(51, len(envelope)//10*2+1),
                                         polyorder=3)

    # Find threshold
    mean_amplitude = np.mean(envelope_smooth)
    threshold = mean_amplitude * threshold_factor

    # Find peaks above threshold
    peaks, properties = signal.find_peaks(
        envelope_smooth,
        height=threshold,
        distance=min_distance
    )

    bifurcation_amplitudes = envelope_smooth[peaks]

    return peaks, bifurcation_amplitudes


def compute_instantaneous_frequency(x: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Compute instantaneous frequency using Hilbert transform.

    Args:
        x: Input signal
        fs: Sampling frequency

    Returns:
        Instantaneous frequency
    """
    analytic_signal = signal.hilbert(x)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    # Pad to match original length
    instantaneous_frequency = np.concatenate([[instantaneous_frequency[0]],
                                            instantaneous_frequency])

    return instantaneous_frequency


def compute_spectral_features(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None
) -> dict:
    """
    Compute comprehensive spectral features.

    Args:
        x: Input signal
        fs: Sampling frequency
        nperseg: Length of each segment for Welch's method

    Returns:
        Dictionary of spectral features
    """
    if nperseg is None:
        nperseg = min(len(x), 256)

    # Power spectral density
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)

    # Remove DC component
    if freqs[0] == 0:
        freqs = freqs[1:]
        psd = psd[1:]

    # Spectral centroid
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)

    # Spectral spread
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))

    # Spectral rolloff (85% of energy)
    cumulative_psd = np.cumsum(psd)
    rolloff_idx = np.where(cumulative_psd >= 0.85 * cumulative_psd[-1])[0]
    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

    # Spectral flux
    spectral_flux = np.sum(np.diff(psd) ** 2) if len(psd) > 1 else 0

    # Dominant frequency
    dominant_freq = freqs[np.argmax(psd)]

    # Total power
    total_power = np.sum(psd)

    # Spectral entropy
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    return {
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread,
        'spectral_rolloff': spectral_rolloff,
        'spectral_flux': spectral_flux,
        'dominant_frequency': dominant_freq,
        'total_power': total_power,
        'spectral_entropy': spectral_entropy,
        'frequencies': freqs,
        'psd': psd
    }


def compute_phase_space_features(x: np.ndarray, x_dot: np.ndarray) -> dict:
    """
    Compute phase space features.

    Args:
        x: Position signal
        x_dot: Velocity signal

    Returns:
        Dictionary of phase space features
    """
    # Phase space trajectory
    trajectory = np.column_stack([x, x_dot])

    # Compute convex hull area (if possible)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(trajectory)
        phase_area = hull.volume
    except:
        # Fallback: bounding box area
        phase_area = (np.max(x) - np.min(x)) * (np.max(x_dot) - np.min(x_dot))

    # Phase space diameter
    distances = []
    n_samples = min(100, len(trajectory))  # Sample for efficiency
    indices = np.random.choice(len(trajectory), n_samples, replace=False)
    sample_trajectory = trajectory[indices]

    for i in range(len(sample_trajectory)):
        for j in range(i+1, len(sample_trajectory)):
            dist = np.linalg.norm(sample_trajectory[i] - sample_trajectory[j])
            distances.append(dist)

    phase_diameter = np.max(distances) if distances else 0

    # Energy (kinetic + potential approximation)
    kinetic_energy = 0.5 * x_dot ** 2
    potential_energy = 0.5 * x ** 2  # Assuming harmonic potential
    total_energy = kinetic_energy + potential_energy

    # Energy statistics
    mean_energy = np.mean(total_energy)
    energy_variance = np.var(total_energy)

    return {
        'phase_area': phase_area,
        'phase_diameter': phase_diameter,
        'mean_energy': mean_energy,
        'energy_variance': energy_variance,
        'trajectory': trajectory
    }


def compute_nonlinear_features(x: np.ndarray) -> dict:
    """
    Compute nonlinear dynamics features.

    Args:
        x: Input signal

    Returns:
        Dictionary of nonlinear features
    """
    # Approximate Lyapunov exponent
    def lyapunov_exponent(data, m=3, tau=1):
        """Approximate largest Lyapunov exponent."""
        if len(data) < m + 1:
            return 0.0

        # Embed the time series
        embedded = np.array([data[i:i+m] for i in range(len(data)-m+1)])

        # Find nearest neighbors and compute divergence
        divergences = []
        for i in range(len(embedded) - tau):
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            # Find nearest neighbor (excluding self)
            distances[i] = np.inf
            nearest_idx = np.argmin(distances)

            if i + tau < len(embedded) and nearest_idx + tau < len(embedded):
                initial_dist = distances[nearest_idx]
                final_dist = np.linalg.norm(embedded[i + tau] - embedded[nearest_idx + tau])

                if initial_dist > 0 and final_dist > 0:
                    divergences.append(np.log(final_dist / initial_dist))

        return np.mean(divergences) if divergences else 0.0

    # Hurst exponent
    def hurst_exponent(data):
        """Compute Hurst exponent using R/S analysis."""
        if len(data) < 10:
            return 0.5

        n = len(data)
        mean_data = np.mean(data)

        # Cumulative deviations
        cumdev = np.cumsum(data - mean_data)

        # Range
        R = np.max(cumdev) - np.min(cumdev)

        # Standard deviation
        S = np.std(data)

        if S == 0:
            return 0.5

        # R/S ratio
        rs = R / S

        # Hurst exponent
        return np.log(rs) / np.log(n)

    # Sample entropy
    def sample_entropy(data, m=2, r=0.2):
        """Compute sample entropy."""
        if len(data) < m + 1:
            return 0.0

        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
            C = np.zeros(len(patterns))

            for i in range(len(patterns)):
                template = patterns[i]
                for j in range(len(patterns)):
                    if _maxdist(template, patterns[j]) <= r * np.std(data):
                        C[i] += 1.0

            phi = np.mean(np.log(C / len(patterns)))
            return phi

        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0

    # Compute features
    lyap_exp = lyapunov_exponent(x)
    hurst_exp = hurst_exponent(x)
    samp_ent = sample_entropy(x)

    # Detrended fluctuation analysis (simplified)
    def dfa_alpha(data, min_scale=4, max_scale=None):
        """Simplified DFA scaling exponent."""
        if max_scale is None:
            max_scale = len(data) // 4

        if max_scale <= min_scale:
            return 1.0

        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 10).astype(int)
        scales = np.unique(scales)

        fluctuations = []

        for scale in scales:
            if scale >= len(data):
                continue

            # Integrate the signal
            integrated = np.cumsum(data - np.mean(data))

            # Divide into non-overlapping segments
            n_segments = len(integrated) // scale
            segments = integrated[:n_segments * scale].reshape(n_segments, scale)

            # Detrend each segment
            segment_fluctuations = []
            for segment in segments:
                x_vals = np.arange(len(segment))
                coeffs = np.polyfit(x_vals, segment, 1)
                trend = np.polyval(coeffs, x_vals)
                detrended = segment - trend
                segment_fluctuations.append(np.sqrt(np.mean(detrended ** 2)))

            fluctuations.append(np.mean(segment_fluctuations))

        if len(fluctuations) < 2:
            return 1.0

        # Fit power law
        log_scales = np.log10(scales[:len(fluctuations)])
        log_fluctuations = np.log10(fluctuations)

        try:
            alpha = np.polyfit(log_scales, log_fluctuations, 1)[0]
            return alpha
        except:
            return 1.0

    dfa_alpha_val = dfa_alpha(x)

    return {
        'lyapunov_exponent': lyap_exp,
        'hurst_exponent': hurst_exp,
        'sample_entropy': samp_ent,
        'dfa_alpha': dfa_alpha_val
    }


def filter_signal(
    x: np.ndarray,
    filter_type: str = 'lowpass',
    cutoff: float = 0.1,
    fs: float = 1.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply digital filter to signal.

    Args:
        x: Input signal
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
        cutoff: Cutoff frequency (or frequencies for bandpass/bandstop)
        fs: Sampling frequency
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs

    if filter_type in ['lowpass', 'highpass']:
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype=filter_type)
    elif filter_type in ['bandpass', 'bandstop']:
        if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
            low, high = cutoff
            normal_cutoff = [low / nyquist, high / nyquist]
            b, a = signal.butter(order, normal_cutoff, btype=filter_type)
        else:
            raise ValueError("Bandpass/bandstop filters require two cutoff frequencies")
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Apply filter
    filtered_signal = signal.filtfilt(b, a, x)

    return filtered_signal


def resample_signal(
    x: np.ndarray,
    original_fs: float,
    target_fs: float,
    method: str = 'scipy'
) -> np.ndarray:
    """
    Resample signal to different sampling rate.

    Args:
        x: Input signal
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency
        method: Resampling method ('scipy', 'decimate', 'interpolate')

    Returns:
        Resampled signal
    """
    if abs(original_fs - target_fs) < 1e-6:
        return x

    if method == 'scipy':
        # Use scipy's resample function
        num_samples = int(len(x) * target_fs / original_fs)
        resampled = signal.resample(x, num_samples)

    elif method == 'decimate':
        # Decimation (downsampling only)
        if target_fs > original_fs:
            raise ValueError("Decimation can only downsample")

        decimation_factor = int(original_fs / target_fs)
        resampled = signal.decimate(x, decimation_factor, ftype='iir')

    elif method == 'interpolate':
        # Interpolation-based resampling
        original_time = np.arange(len(x)) / original_fs
        target_time = np.arange(0, original_time[-1], 1/target_fs)

        interp_func = interp1d(original_time, x, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        resampled = interp_func(target_time)

    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return resampled


def compute_time_frequency_features(
    x: np.ndarray,
    fs: float = 1.0,
    window: str = 'hann',
    nperseg: Optional[int] = None
) -> dict:
    """
    Compute time-frequency domain features using spectrogram.

    Args:
        x: Input signal
        fs: Sampling frequency
        window: Window function
        nperseg: Length of each segment

    Returns:
        Dictionary of time-frequency features
    """
    if nperseg is None:
        nperseg = min(len(x) // 8, 256)

    # Compute spectrogram
    freqs, times, Sxx = signal.spectrogram(x, fs=fs, window=window, nperseg=nperseg)

    # Time-frequency features
    # Spectral centroid over time
    spectral_centroid_time = np.sum(freqs[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)

    # Spectral bandwidth over time
    spectral_bandwidth_time = np.sqrt(
        np.sum(((freqs[:, np.newaxis] - spectral_centroid_time) ** 2) * Sxx, axis=0) /
        np.sum(Sxx, axis=0)
    )

    # Spectral rolloff over time
    cumulative_Sxx = np.cumsum(Sxx, axis=0)
    total_energy = np.sum(Sxx, axis=0)
    rolloff_indices = np.argmax(cumulative_Sxx >= 0.85 * total_energy, axis=0)
    spectral_rolloff_time = freqs[rolloff_indices]

    # Spectral flux over time
    spectral_flux_time = np.sum(np.diff(Sxx, axis=1) ** 2, axis=0)
    spectral_flux_time = np.concatenate([[0], spectral_flux_time])  # Pad first value

    return {
        'spectrogram': Sxx,
        'frequencies': freqs,
        'times': times,
        'spectral_centroid_time': spectral_centroid_time,
        'spectral_bandwidth_time': spectral_bandwidth_time,
        'spectral_rolloff_time': spectral_rolloff_time,
        'spectral_flux_time': spectral_flux_time,
        'mean_spectral_centroid': np.mean(spectral_centroid_time),
        'std_spectral_centroid': np.std(spectral_centroid_time),
        'mean_spectral_bandwidth': np.mean(spectral_bandwidth_time),
        'std_spectral_bandwidth': np.std(spectral_bandwidth_time)
    }
