# methods/nlms.py

import numpy as np


# --------------------------------------------------------------------- #
def _nlms_core(
    d: np.ndarray,
    x: np.ndarray,
    mu: float,
    eps: float,
    K: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Core Normalized Least Mean Squares (NLMS) adaptive filter.

    This function applies the NLMS algorithm to iteratively estimate filter coefficients
    and reduce noise from a desired signal using a reference input.

    Args:
        d (np.ndarray): Desired signal (noisy observation).
        x (np.ndarray): Reference input signal (noise).
        mu (float): Base step size (learning rate).
        eps (float): Small constant to prevent division by zero during normalization.
        K (int): Length of the adaptive filter.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: 
            - e: Error signal over time (estimated clean signal).
            - w_hist: History of filter weights at each time step.

    Raises:
        ValueError: If signal lengths are mismatched, too short, or if parameters are invalid.
    """
    N = len(d)
    if len(x) != N:
        raise ValueError("Signals 'd' and 'x' must have the same length.")
    if N < K:
        raise ValueError("Signal length must be at least equal to filter length K.")
    if mu <= 0 or eps <= 0:
        raise ValueError("Parameters 'mu' and 'eps' must be greater than zero.")

    # Initialization
    w = np.zeros(K, dtype=np.float32)
    e = np.zeros(N, dtype=np.float32)
    w_hist = [np.copy(w) for _ in range(N)]
    # Adaptive filtering loop
    for n in range(K - 1, N):
        # Extract most recent K samples (reversed for causal filter)
        x_vec = x[n - K + 1 : n + 1][::-1]
        # Filter output
        y = np.dot(w, x_vec)
        # Error between desired and output
        e[n] = d[n] - y
        # Normalize step size using input energy
        norm_factor = np.dot(x_vec, x_vec) + eps
        mu_n = mu / norm_factor
        w += mu_n * x_vec * e[n]
        w_hist[n] = np.copy(w)

    return e, w_hist


# --------------------------------------------------------------------- #
def filter_signal_nlms(
    noisy_signal: np.ndarray,
    noise: np.ndarray,
    K: int,
    args: dict,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Applies Normalized LMS (NLMS) adaptive filtering to denoise a signal.

    This function wraps `_nlms_core` with argument extraction and error handling.

    Args:
        noisy_signal (np.ndarray): Noisy observed signal.
        noise (np.ndarray): Reference noise signal.
        K (int): Length of the adaptive filter.
        args (dict): Optional arguments:
            - 'mu' (float): Base step size. Default is 0.1.
            - 'eps' (float): Regularization constant. Default is 1e-8.

    Returns:
        tuple[np.ndarray, list[np.ndarray]] | None: 
            - e: Error signal (estimated clean signal).
            - w_hist: Filter weight history.
            Returns None if an error occurs during filtering.
    """
    try:
        # Extract hyperparameters from args, or use defaults
        mu = float(args.get("mu", 0.1))
        eps = float(args.get("eps", 1e-8))
        return _nlms_core(noisy_signal, noise, mu, eps, K)

    except Exception as err:
        print(f"[NLMS Error] {err}")
        return None
