import numpy as np


def filter_signal_vslms(noisy_signal, noise, K, args):
    """
    Apply time-varying step-size LMS to filter a noisy signal.

    Parameters:
    - noisy_signal: observed signal (numpy array)
    - noise: reference noise signal (numpy array)
    - K: filter length (int)
    - args: dict with optional keys 'mu' (float) and 'lambda' (float)

    Returns:
    - filtered_signal: numpy array, or None on error
    """
    try:
        N = noisy_signal.shape[0]
        # Check that noise and signal have the same length
        if noise.shape[0] != N:
            print("Error: noisy_signal and noise must have the same length.")
            return None

        # Signal must be at least as long as the filter length
        if N < K:
            print("Error: signal length must be at least equal to K.")
            return None

        mu = args["mu"]
        lambda_ = args["lambda"]

        # Basic validation
        if lambda_ < 0 or mu <= 0:
            print("Error: 'mu' must be > 0 and 'lambda' must be >= 0.")
            return None

        filter_ = np.zeros(K)
        filter_history = [np.copy(filter_) for _ in range(N)]
        output = np.zeros(N)
        for n in range(K - 1, N):
            # Get the current input window (reversed for convolution)
            Xn = noise[n - K + 1 : n + 1][::-1]
            # Desired signal (noisy observation)
            dn = noisy_signal[n]
            # Filter output
            y = np.dot(Xn, filter_)
            # Error between desired and output
            e = dn - y
            # Save error (filtered signal)
            output[n] = e
            t = n - (K - 1)  # makes t start at 0
            # Update step-size (decays with time)
            mu_n = mu / (1 + lambda_ * t)
            # Update filter coefficients using LMS rule
            filter_ += mu_n * Xn * e
            filter_history[n] = np.copy(filter_)

        return output, filter_history

    except Exception as e:
        print(f"Error: {e}")
        return None
