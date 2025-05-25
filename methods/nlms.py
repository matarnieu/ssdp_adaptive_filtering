import numpy as np


def filter_signal_nlms(noisy_signal, noise, K, args):
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
        # Check that noise and signal have the same length
        if noise.shape[0] != N:
            print("Error: noisy_signal and noise must have the same length.")
            return None
        # Signal must be at least as long as the filter length
        if N < K:
            print("Error: signal length must be at least equal to K.")
            return None

        mu = args.get("mu", 0.1)
        eps = args.get('eps', 1e-8)
        # Basic validation
        if mu <= 0:
            print("Error: 'mu' must be > 0")
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
            output[n] = e
            mu_n = mu/(np.linalg.norm(Xn)**2 + eps)
            # Update filter coefficients using NLMS rule
            filter_ += mu_n * Xn * e
            filter_history[n] = np.copy(filter_)

        return output, filter_history

    except Exception as e:
        print(f"Error: {e}")
        return None
