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
    for n in range(K, N):
        # Use last K samples
        # Compute sample correlation matrix for noise
        # In lecture, K+1 samples are used, but here we start at n-K+1 to have a K tap filter
        x = noise[: n + 1]  # n - K + 1
        x = x[::-1]  # Reverse order
        d = noisy_signal[: n + 1]
        d = d[::-1]  # Reverse order
        # R_x = np.outer(x, x) # Instantaneous correlation matrix, does not work
        # Compute autocorrelation vector for lags 0 to K-1
        r = np.zeros(K)
        for lag in range(K):
            num_terms = n + 1 - lag
            r[lag] = np.mean(
                x[:num_terms] * x[lag : lag + num_terms]
            )  # average over all valid pairs
        R_x = toeplitz(r)
        # r_dx = noisy_signal[n] * x # Instantaneous cross-correlation vector, does not work
        r_dx = np.zeros(K)
        for lag in range(K):
            num_terms = n + 1 - lag
            r_dx[lag] = np.mean(d[:num_terms] * x[lag : lag + num_terms])
        # Solve Wiener-Hopf equation: f = R_x^{-1} * r_dx
        try:
            f = np.linalg.solve(R_x, r_dx)
        except np.linalg.LinAlgError:
            f = np.zeros_like(x)  # fallback in case R_x is singular
            print(f"Warning: R_x is singular for index {n}, using zero filter.")
        # Apply filter to noise
        x_filter = noise[n - K + 1 : n + 1][::-1]
        filtered_noise = np.dot(x_filter, f)
        # Substract filtered noise from noisy signal
        filtered_signal[n] = noisy_signal[n] - filtered_noise
        if abs(filtered_signal[n]) > LIMIT:
            print(
                f"Warning: Filtered signal value exceeds limit at index {n}: {filtered_signal[n]}"
            )
    return filtered_signal
