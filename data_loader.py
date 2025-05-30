from __future__ import annotations
import numpy as np
import soundfile as sf  # ← new: preserves native scale
import librosa  # still used for synthetic helpers
from synthesis import *

"""Returns noisy signal and noise signal as a 2-tuple of numpy arrays
Prints error message and returns None when it fails"""


def load_real_data(
    noisy_signal_path: str,
    noise_path: str,
    *,
    level_match: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads real noisy and noise audio files.

    Converts stereo to mono if needed, optionally matches levels (RMS), 
    and trims both signals to the same length.

    Args:
        noisy_signal_path (str): Path to the noisy audio file.
        noise_path (str): Path to the noise audio file.
        level_match (bool, optional): Whether to match the noise level to the noisy signal. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Tuple containing:
            - noisy_signal: 1D array of noisy signal samples.
            - noise_signal: 1D array of noise signal samples.
            - sr_d: Sampling rate (Hz).

    Raises:
        ValueError: If sampling rates of the two files differ.
    """
    noisy_signal, sr_d = sf.read(noisy_signal_path, dtype="float32")
    noise_signal, sr_x = sf.read(noise_path, dtype="float32")

    if noisy_signal.ndim > 1:
        noisy_signal = noisy_signal.mean(axis=1)

    if noise_signal.ndim > 1:
        noise_signal = noise_signal.mean(axis=1)

    if sr_d != sr_x:
        raise ValueError(f"Sampling-rate mismatch: D={sr_d} Hz,  X={sr_x} Hz")

    if level_match:
        rms_d = np.std(noisy_signal)
        rms_x = np.std(noise_signal) + 1e-12 # Avoid division by zero
        noise_signal = noise_signal * (rms_d / rms_x)

    # Trim both signals to the same length
    n = min(len(noisy_signal), len(noise_signal))
    return noisy_signal[:n], noise_signal[:n], sr_d


"""Generate synthetic signal. Returns true signal, noisy signal and noise signal as a 3-tuple of numpy arrays
Prints error message and returns None when it fails"""


def generate_signal(num_samples, low, high, type_signal):
    t = np.linspace(low, high, num_samples)
    f = 0.01
    if type_signal == "sinus":
        return np.sin(2 * np.pi * f * t)
    elif type_signal == "musical":
        return _generate_complex_musical_signal(t)
    else:
        raise ValueError("type_signal must be 'sinus' or 'musical'")


def generate_synthetic_data(
    num_samples,
    low,
    high,
    switching_interval,
    filter_size,
    filter_type,
    filter_changing_speed,
    noise_power,
    noise_type,
    type_signal="sinus",
):
    num_samples: int,
    low: float,
    high: float,
    switching_interval: int,
    filter_size: int,
    filter_type: str,
    filter_changing_speed: float,
    noise_power: float | None,
    noise_power_change: float,
    noise_distribution_change: float,
    type_signal: str = "sinus",
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Generates a synthetic dataset composed of a clean signal, noise, and filtered noisy signal.

    The function builds a base signal (sinusoidal or binary), optionally generates noise with
    changing characteristics, applies a dynamic filter to the noise, and combines the result with the signal.

    Args:
        num_samples (int): Total number of time samples.
        low (float): Lower bound for time axis (used to generate the base signal).
        high (float): Upper bound for time axis.
        switching_interval (int): Number of samples between changes in filter or noise parameters.
        filter_size (int): Length of the finite impulse response filter.
        filter_type (str): Filter type to generate (e.g. "lowpass", "random").
        filter_changing_speed (float): Rate at which the filter coefficients change over time.
        noise_power (float | None): Power of the generated noise. If None, no noise is added.
        noise_power_change (float): Amplitude of noise power variation over time.
        noise_distribution_change (float): Degree to which the noise distribution changes over time.
        type_signal (str, optional): Type of clean signal to generate. Either "sinus" or "binary". Defaults to "sinus".

    Returns:
        tuple[np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None]:
            - noisy_signal: Signal + filtered noise. `None` if no noise was added.
            - clean_signal: The clean, generated signal.
            - raw_noise: The generated raw noise signal. `None` if no noise was added.
            - filter_matrix: 2D array where each row is a filter at a given time step. `None` if no noise.

    Raises:
        ValueError: If `type_signal` is invalid.
    """
    # generate signal
    signal = generate_signal(
        num_samples=num_samples,
        low=low,
        high=high,
        type_signal=type_signal,
    )

    # If no noise is to be added, return early with clean signal only
    if noise_power == None:
        return None, signal, None, None

    # Generate dynamic noise with varying power and distribution
    noise = generate_noise(
        num_samples,
        noise_power,
        noise_type,
        switching_interval,
    )
    
    # Create a time-varying filter (one filter per time step)
    H = generate_filter(
        filter_type,
        filter_size,
        num_samples,
        switching_interval,
        filter_changing_speed,
    )
    # Filter the noise signal manually using time-varying convolution
    noise_filtered = np.zeros(H.shape[0])
    padded_noise = np.pad(noise, (filter_size - 1, 0), mode="constant")
    for idx, h in enumerate(H):
        # Take the most recent `filter_size` samples, reverse for causal filtering
        segment = padded_noise[idx : idx + filter_size][::-1]
        noise_filtered[idx] = np.dot(h, segment)
    
    # Add filtered noise to clean signal
    noisy_signal = signal + noise_filtered
    return noisy_signal, signal, noise, H
