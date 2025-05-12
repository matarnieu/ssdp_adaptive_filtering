import numpy as np
from scipy.linalg import toeplitz

"""Use generalized wiener filter to extract filtered signal from
noisy_signal and noise (numpy arrays). Approximate K-tap filter. Return filtered_signal.
In case of error, print error message and return None."""

LIMIT = 3


def filter_gwf(noisy_signal, noise, K):
    # In lecture: noisy_signal = D
    #             noise = X

    N = len(noisy_signal)
    filtered_signal = np.zeros(N)
    filtered_signal[:K] = noisy_signal[:K]  # Initialize first K samples
    # We simulate an on-the-fly situation where we store the last K samples
    # We can only start estimating the correlation matrix after K samples
    """for n in range(K, N):
        # Use last K samples
        # Compute sample correlation matrix for noise
        n_samples_Rx = min(n, 50 * K)  # n
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
        # Apply filter to noise
        x_filter = noise[n - K : n][::-1]
        filtered_noise = np.dot(x_filter, f)
        # Substract filtered noise from noisy signal
        filtered_signal[n - 1] = noisy_signal[n - 1] - filtered_noise
        if abs(filtered_signal[n - 1]) > LIMIT:
            print(
                f"Warning: Filtered signal value exceeds limit at index {n-1}: {filtered_signal[n-1]}"
            )
    return filtered_signal
    """
    R_x = np.zeros((K, K))
    r_dx = np.zeros(K)
    lambda_ = 0.999

    for n in range(K, N):
        x = noise[n - K + 1 : n + 1][::-1]
        d = noisy_signal[n]

        R_x = lambda_ * R_x + (1 - lambda_) * np.outer(x, x)
        r_dx = lambda_ * r_dx + (1 - lambda_) * d * x

        try:
            f = np.linalg.solve(R_x + 1e-6 * np.eye(K), r_dx)  # regularized
        except np.linalg.LinAlgError:
            f = np.zeros(K)

        filtered_noise = np.dot(f, x)
        filtered_signal[n] = noisy_signal[n] - filtered_noise
    return filtered_signal
