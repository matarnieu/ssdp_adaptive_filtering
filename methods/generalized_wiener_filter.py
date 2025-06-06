import numpy as np
from scipy.linalg import toeplitz

"""Use generalized wiener filter to extract filtered signal from
noisy_signal and noise (numpy arrays). Approximate K-tap filter. Return filtered_signal.
In case of error, print error message and return None."""


def filter_signal_gwf_fc(noisy_signal, noise, K, args):
    """Apply generalized Wiener filter with full correlation.

    Args:
        noisy_signal (np.ndarray): Noisy observed signal.
        noise (np.ndarray): Noise-only signal reference.
        K (int): Number of filter taps.
        args (dict): Dictionary of additional parameters. No params needed.

    Returns:
        tuple: (filtered_signal, filter_history)
            filtered_signal (np.ndarray): Denoised output signal.
            filter_history (list of np.ndarray): History of filter coefficients.
    """
    return _filter_gwf_hard_cut(noisy_signal, noise, K, False, args)


def filter_signal_gwf_swc(noisy_signal, noise, K, args):
    """Apply generalized Wiener filter with sliding window cutoff (SWC).

    Args:
        noisy_signal (np.ndarray): Noisy observed signal.
        noise (np.ndarray): Noise-only signal reference.
        K (int): Number of filter taps.
        args (dict): Dictionary containing key 'window_size' (int): multiplier for window length.

    Returns:
        tuple: (filtered_signal, filter_history)
            filtered_signal (np.ndarray): Denoised output signal.
            filter_history (list of np.ndarray): History of filter coefficients.
    """
    return _filter_gwf_hard_cut(noisy_signal, noise, K, True, args)


def _filter_gwf_hard_cut(noisy_signal, noise, K, use_sliding_window, args):
    """Internal function to apply generalized Wiener filtering using correlation estimates.

    Args:
        noisy_signal (np.ndarray): Noisy observed signal (reference 'd').
        noise (np.ndarray): Noise-only reference signal (reference 'x').
        K (int): Number of taps in the FIR Wiener filter.
        use_sliding_window (bool): Whether to use a sliding window for estimating correlations.
        args (dict): Dictionary containing filter parameters. Must contain 'window_size' if sliding window is used.

    Returns:
        tuple: (filtered_signal, filter_history)
            filtered_signal (np.ndarray): Denoised output signal.
            filter_history (list of np.ndarray): Per-sample FIR filter coefficients.

    Notes:
        If the autocorrelation matrix R_x is singular, falls back to a zero filter and prints a warning.
    """

    N = len(noisy_signal)
    filter_history = [np.zeros(K) for _ in range(N)]
    filtered_signal = np.zeros(N)
    filtered_signal[:K] = noisy_signal[:K]  # Initialize first K samples
    # We simulate an on-the-fly situation where we store the last K samples
    # We can only start estimating the correlation matrix after K samples
    for n in range(K, N):
        # Use last K samples
        # Compute sample correlation matrix for noise
        if use_sliding_window:
            n_samples_Rx = min(n, int(args["window_size"]) * K)
        else:
            n_samples_Rx = n
        n_samples_rdx = n_samples_Rx

        x = noise[n - n_samples_Rx : n]
        x = x[::-1]  # Reverse order
        x_rdx = noise[n - n_samples_rdx : n]
        x_rdx = x_rdx[::-1]  # Reverse order
        d = noisy_signal[n - n_samples_rdx : n]
        d = d[::-1]  # Reverse order
        # R_x = np.outer(x, x) # Instantaneous correlation matrix, does not work
        # Compute autocorrelation vector for lags 0 to K-1
        r = np.zeros(K)
        for lag in range(K):
            num_terms = n_samples_Rx - lag
            r[lag] = (
                np.sum(x[:num_terms] * x[lag : lag + num_terms]) / n_samples_Rx
            )  # average over all valid pairs
        R_x = toeplitz(r)
        # r_dx = noisy_signal[n] * x # Instantaneous cross-correlation vector, does not work
        r_dx = np.zeros(K)
        for lag in range(K):
            num_terms = n_samples_rdx - lag
            r_dx[lag] = (
                np.sum(d[:num_terms] * x_rdx[lag : lag + num_terms]) / n_samples_rdx
            )
        # Solve Wiener-Hopf equation: f = R_x^{-1} * r_dx
        try:
            f = np.linalg.solve(R_x, r_dx)
        except np.linalg.LinAlgError:
            f = np.zeros(K)  # fallback in case R_x is singular
            print(f"Warning: R_x is singular for index {n-1}, using zero filter.")
        filter_history[n] = f
        # Apply filter to noise
        x_filter = noise[n - K : n][::-1]
        filtered_noise = np.dot(x_filter, f)
        # Substract filtered noise from noisy signal
        filtered_signal[n - 1] = noisy_signal[n - 1] - filtered_noise
    return filtered_signal, filter_history


def filter_signal_gwf_ema(noisy_signal, noise, K, args):
    """Apply generalized Wiener filter using exponential moving average (EMA).

    Args:
        noisy_signal (np.ndarray): Noisy observed signal.
        noise (np.ndarray): Noise-only signal reference.
        K (int): Number of filter taps.
        args (dict): Must contain 'lambda' (float), the forgetting factor in [0, 1].

    Returns:
        tuple: (filtered_signal, filter_history)
            filtered_signal (np.ndarray): Denoised output signal.
            filter_history (list of np.ndarray): History of filter coefficients.

    Notes:
        Regularizes the correlation matrix by adding a small multiple of the identity matrix.
        Falls back to a zero filter if matrix is singular.
    """
    N = len(noisy_signal)
    filtered_signal = np.zeros(N)
    filtered_signal[:K] = noisy_signal[:K]  # Initialize first K samples
    filtered_signal[:K] = noisy_signal[:K]
    R_x = np.zeros((K, K))
    r_dx = np.zeros(K)
    lambda_ = args["lambda"]
    filter_history = [np.zeros(K) for _ in range(N)]

    for n in range(K, N):
        x = noise[n - K + 1 : n + 1][::-1]
        d = noisy_signal[n]

        R_x = lambda_ * R_x + (1 - lambda_) * np.outer(x, x)
        r_dx = lambda_ * r_dx + (1 - lambda_) * d * x

        try:
            f = np.linalg.solve(R_x + 1e-6 * np.eye(K), r_dx)  # regularized
        except np.linalg.LinAlgError:
            f = np.zeros(K)

        filter_history[n] = f
        filtered_noise = np.dot(f, x)
        filtered_signal[n] = noisy_signal[n] - filtered_noise
    return filtered_signal, filter_history
