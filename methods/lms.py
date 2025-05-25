import numpy as np

def filter_signal_lms(noisy_signal, noise, K, args):
    """
    Apply LMS adaptive filtering to denoise a signal.

    Parameters:
    - noisy_signal: observed signal (numpy array)
    - noise: reference noise signal (numpy array)
    - K: filter length (int)
    - args: dict with optional key 'mu' (float)

    Returns:
    - filtered_signal: numpy array of shape (N,), or None on error
    """
    try:
        N = noisy_signal.shape[0]
        if noise.shape[0] != N:
            print("Error: noisy_signal and noise must have the same length.")
            return None
        if N < K:
            print("Error: signal length must be at least equal to K.")
            return None

        mu = args.get('mu', 0.1)
        filter_ = np.zeros(K)
        output = np.zeros(N)

        for n in range(K - 1, N):
            Xn = noise[n - K + 1 : n + 1][::-1]
            dn = noisy_signal[n]
            y = np.dot(Xn, filter_)
            e = dn - y
            output[n] = e
            filter_ += mu * Xn * e

        return output

    except Exception as e:
        print(f"Error: {e}")
        return None
