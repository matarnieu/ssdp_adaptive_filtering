import numpy as np
import librosa
from utils import *
from scipy.signal import lfilter

"""Returns noisy signal and noise signal as a 2-tuple of numpy arrays
Prints error message and returns None when it fails"""


def load_real_data(noisy_signal_path, noise_path):
    noisy_signal, srns = librosa.load(
        noisy_signal_path
    )  # srns = sampling rate noisy signal
    noise_signal, srn = librosa.load(noise_path)  # srn = sampling rate noise
    return noisy_signal, noise_signal  # , srns, srn


"""Generate synthetic signal. Returns true signal, noisy signal and noise signal as a 3-tuple of numpy arrays
Prints error message and returns None when it fails"""


def generate_signal(num_samples, low, high, type_signal):
    t = np.linspace(low, high, num_samples)
    frequency = 0.001
    if type_signal == "sinus":
        signal = np.sin(2 * np.pi * frequency * t)
    elif type_signal == "binary":
        signal = np.zeros(shape=num_samples)
        signal[::2] = 1
    else:
        raise ValueError
    return signal


def generate_synthetic_data(
    num_samples,
    low,
    high,
    size_filter,
    type_filter,
    snr,
    type_signal="sinus",
):
    # generate signal
    signal = generate_signal(
        num_samples=num_samples, low=low, high=high, type_signal=type_signal
    )

    # if snr is None, no noise added, hence no need to return the filter, noisy signal and the noise
    if snr == None:
        return None, signal, None, None

    power_noise = compute_power_noise(signal, snr)
    noise = generate_noise(power_noise=power_noise, size=signal.shape[0])
    # filter changes at every timestep, hence H.shape() = (num_sample, filter_size)
    H = generate_time_varying_filter(
        type=type_filter, size_filter=size_filter, timestep=num_samples
    )
    # noise_filtered = np.convolve(noise, h, mode='same')
    noise_filtered = np.zeros(H.shape[0])
    #    for idx, h in enumerate(H):
    #        noise_filtered[idx] = lfilter(h, 1, noise)[idx]
    padded_noise = np.pad(noise, (size_filter - 1, 0), mode="constant")
    for idx, h in enumerate(H):
        # Doing the convolution manually as we dont need the whole convolution
        segment = padded_noise[idx : idx + size_filter][::-1]  # keep the filter causal
        noise_filtered[idx] = np.dot(h, segment)
    noisy_signal = signal + noise_filtered
    return noisy_signal, signal, noise, H


def main(plot=True):
    # begining and end value of the signal we want to generate
    low, high = 0, 1000 * np.pi
    # size of the signal, which is equal to the number of timestep
    num_samples = 10000
    # size of the filter
    size_filter = 10
    # amount of noise we want to have in the UNFILTERED NOISE (snr will be bigger after filtering it as it should lower the variance)
    snr = 10

    type_signal = "sinus"  # can be sinus or binary
    type_filter = "exponential_decay"  # can be moving_average or exponential_decay

    noisy_signal, signal, noise, h = generate_synthetic_data(
        num_samples=num_samples,
        low=low,
        high=high,
        size_filter=size_filter,
        snr=snr,
        type_signal=type_signal,
        type_filter=type_filter,
    )

    if plot:
        plot_2(
            noisy_signal,
            signal,
            h,
            type_signal,
            type_filter,
            snr,
            window_start=1000,
            window_size=200,
        )

    return signal, noisy_signal, noise, size_filter


if __name__ == "__main__":
    main()
