import numpy as np

"""Use generalized wiener filter to extract filtered signal from
noisy_signal and noise (numpy arrays). Approximate K-tap filter. Return filtered_signal.
In case of error, print error message and return None."""


def filter_gwf(noisy_signal, noise, K, args):
    # In lecture: noisy_signal = D
    #             noise = X

    N = len(noisy_signal)
    filtered_signal = np.zeros(N)
    # We simulate an on-the-fly situation where we store the last K samples
    # We can only start estimating the correlation matrix after K samples
    # What to do for the first K samples? --> Don't filter
    for n in range(K, N):
        # Use last K samples
        # Compute sample correlation matrix for noise
        # In lecture, K+1 samples are used, but here we start at n-K+1 to have a K tap filter
        x = noise[n - K + 1 : n + 1]
        x = x[::-1]  # Reverse order
        R_x = np.outer(x, x)
        # Instantaneous cross-correlation betweeen noise and signal at time n
        r_dx = noisy_signal[n] * x
        # Solve Wiener-Hopf equation: f = R_x^{-1} * r_dx
        try:
            f = np.linalg.solve(R_x, r_dx)
        except np.linalg.LinAlgError:
            f = np.zeros_like(x)  # fallback in case R_x is singular
            print(f"Warning: R_x is singular for index {n}, using zero filter.")
        # Apply filter to noise
        filtered_noise = np.dot(x, f)
        # Substract filtered noise from noisy signal
        filtered_signal[n] = noisy_signal[n] - filtered_noise
    return filtered_signal
