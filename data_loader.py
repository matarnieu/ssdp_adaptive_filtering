from __future__ import annotations
import numpy as np
import soundfile as sf  # ← new: preserves native scale
import librosa  # still used for synthetic helpers
from synthesis import *


def load_real_data(
    noisy_signal_path: str,
    noise_path: str,
    *,
    level_match: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads real-world audio data from two file paths.

    Args:
        noisy_signal_path (str): Path to the noisy signal file.
        noise_path (str): Path to the noise-only signal file.
        level_match (bool, optional): If True, matches RMS energy of noise to noisy signal.

    Returns:
        tuple: (noisy_signal, noise_signal, sample_rate)
            noisy_signal (np.ndarray): The noisy recorded signal.
            noise_signal (np.ndarray): The noise reference signal.
            sample_rate (int): Sampling rate of both signals.

    Raises:
        ValueError: If the sampling rates of the two input files do not match.

    Notes:
        If loading fails, prints an error and returns None.
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
        rms_x = np.std(noise_signal) + 1e-12
        noise_signal = noise_signal * (rms_d / rms_x)

    # Trim to equal length
    n = min(len(noisy_signal), len(noise_signal))
    return noisy_signal[:n], noise_signal[:n], sr_d


def _generate_complex_musical_signal(t, sample_rate=10000):
    """Generates a complex synthetic musical-like signal with harmonics and transients.

    Args:
        t (np.ndarray): Time array.
        sample_rate (int): Sampling rate in Hz.

    Returns:
        np.ndarray: Normalized synthetic musical signal.
    """
    N = len(t)

    # 1. Varying base frequency over time (e.g., vibrato, modulation)
    freq_base = 220 + 30 * np.sin(2 * np.pi * t / 1000)  # modulate around 220 Hz
    base_phase = 2 * np.pi * np.cumsum(freq_base) / sample_rate
    base = np.sin(base_phase)

    # 2. Modulate harmonic content over time
    harmonic_strength = 0.3 + 0.2 * np.sin(2 * np.pi * t / 800)
    harmonic = harmonic_strength * np.sin(2 * base_phase) + 0.25 * np.sin(
        3 * base_phase + np.pi / 6
    )

    # 3. Envelope with jitter
    envelope_base = 0.6 + 0.4 * np.sin(2 * np.pi * t / 400)
    envelope_noise = 0.05 * np.random.randn(N)
    envelope = np.clip(envelope_base + envelope_noise, 0, 1)

    # 4. Grouped, stronger transients (clustered note attacks)
    transients = np.zeros_like(t)
    num_groups = 10
    group_centers = np.linspace(0, N, num_groups + 2, dtype=int)[1:-1]
    for center in group_centers:
        cluster = np.random.randint(center - 20, center + 20, size=5)
        cluster = np.clip(cluster, 0, N - 1)
        transients[cluster] += np.random.uniform(0.7, 1.2, size=len(cluster))

    # 5. Optional reverberation tail (optional long memory)
    decay = np.exp(-np.linspace(0, 3, N))
    reverb = np.convolve(base, decay, mode="same") * 0.05

    # Combine all elements
    sig = envelope * (base + harmonic) + transients + reverb

    # Normalize to unit power
    sig = sig / np.sqrt(np.mean(sig**2))
    return sig


def generate_signal(num_samples, low, high, type_signal):
    """Generates a synthetic clean signal (sinusoidal or musical).

    Args:
        num_samples (int): Number of samples to generate.
        low (float): Start of the time range.
        high (float): End of the time range.
        type_signal (str): Type of signal to generate ('sinus' or 'musical').

    Returns:
        np.ndarray: Generated clean signal.

    Raises:
        ValueError: If `type_signal` is not 'sinus' or 'musical'.
    """
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
    """Generates synthetic noisy data by filtering and adding noise to a clean signal.

    Args:
        num_samples (int): Number of samples in the generated data.
        low (float): Start of time range.
        high (float): End of time range.
        switching_interval (int): Interval at which filters change.
        filter_size (int): Length of the FIR filter.
        filter_type (str): Type of filter to apply (e.g., 'random', 'lowpass').
        filter_changing_speed (float): Speed of filter parameter change.
        noise_power (float | None): Power of the added noise. If None, returns clean signal only.
        noise_type (str): Type of noise (e.g., 'white', 'pink').
        type_signal (str): Type of clean signal to generate ('sinus' or 'musical').

    Returns:
        tuple: (noisy_signal, clean_signal, noise_signal, filters)
            noisy_signal (np.ndarray or None): Combined noisy signal.
            clean_signal (np.ndarray): Original clean signal.
            noise_signal (np.ndarray or None): Generated noise signal.
            filters (np.ndarray or None): Time-varying filter bank.
    """
    # generate signal
    signal = generate_signal(
        num_samples=num_samples,
        low=low,
        high=high,
        type_signal=type_signal,
    )

    # if noise_power is None, no noise added, hence no need to return the filter, noisy signal and the noise
    if noise_power == None:
        return None, signal, None, None

    noise = generate_noise(
        num_samples,
        noise_power,
        noise_type,
        switching_interval,
    )
    # noise = generate_noise(power_noise=power_noise, size=signal.shape[0], timestep=None)
    # filter changes at every timestep, hence H.shape() = (num_sample, filter_size)
    # H = generate_mixed_filter(size_filter, num_samples)
    H = generate_filter(
        filter_type,
        filter_size,
        num_samples,
        switching_interval,
        filter_changing_speed,
    )
    # noise_filtered = np.convolve(noise, h, mode='same')
    noise_filtered = np.zeros(H.shape[0])
    padded_noise = np.pad(noise, (filter_size - 1, 0), mode="constant")
    for idx, h in enumerate(H):
        # Doing the convolution manually as we dont need the whole convolution
        segment = padded_noise[idx : idx + filter_size][::-1]  # keep the filter causal
        noise_filtered[idx] = np.dot(h, segment)
    noisy_signal = signal + noise_filtered
    return noisy_signal, signal, noise, H
