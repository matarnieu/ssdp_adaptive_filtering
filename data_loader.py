from __future__ import annotations
import numpy as np
import soundfile as sf  # ← new: preserves native scale
import librosa  # still used for synthetic helpers
from synthesis import *  # noqa: F403, F401

"""Returns noisy signal and noise signal as a 2-tuple of numpy arrays
Prints error message and returns None when it fails"""


def load_real_data(
    noisy_signal_path: str,
    noise_path: str,
    *,
    level_match: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

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


"""Generate synthetic signal. Returns true signal, noisy signal and noise signal as a 3-tuple of numpy arrays
Prints error message and returns None when it fails"""


def generate_signal(num_samples, low, high, type_signal):
    t = np.linspace(low, high, num_samples)
    f = 0.01
    if type_signal == "sinus":
        return np.sin(2 * np.pi * f * t)
    elif type_signal == "binary":
        sig = np.zeros_like(t)
        sig[::2] = 1.0
        return sig
    else:
        raise ValueError("type_signal must be 'sinus' or 'binary'")


def generate_synthetic_data(
    num_samples,
    low,
    high,
    switching_interval,
    filter_size,
    filter_type,
    filter_changing_speed,
    noise_power,
    noise_power_change,
    noise_distribution_change,
    type_signal="sinus",
):
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
        noise_power_change,
        noise_distribution_change,
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


def main(_plot=True):
    # begining and end value of the signal we want to generate
    low, high = 0, 1000 * np.pi
    # size of the signal, which is equal to the number of timestep
    num_samples = 10000
    # size of the filter
    size_filter = 10
    # amount of noise we want to have in the UNFILTERED NOISE (snr will be bigger after filtering it as it should lower the variance)
    snr = 5

    type_signal = "sinus"  # can be sinus or binary
    type_filter = "moving_average"  # can be moving_average or exponential_decay

    noisy_signal, signal, noise, h = generate_synthetic_data(
        num_samples=num_samples,
        low=low,
        high=high,
        size_filter=size_filter,
        snr=snr,
        type_signal=type_signal,
        type_filter=type_filter,
    )

    if _plot:
        plot(
            noisy_signal,
            signal,
            h,
            type_signal,
            type_filter,
            snr,
            window_start=0,
            window_size=200,
        )

    return signal, noisy_signal, noise, size_filter


if __name__ == "__main__":
    main(True)
