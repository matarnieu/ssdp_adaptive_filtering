# methods/vslms.py

import numpy as np


# --------------------------------------------------------------------- #
def _vslms_core(
    d: np.ndarray,
    x: np.ndarray,
    mu: float,
    lambda_: float,
    K: int,
) -> np.ndarray:
    """Core implementation of the Variable Step-Size LMS (VSLMS) adaptive filter.

    Applies an LMS filter with a time-varying learning rate to reduce noise from
    a desired signal using a reference input.

    Args:
        d (np.ndarray): Desired signal (e.g., noisy target).
        x (np.ndarray): Reference input signal (e.g., noise).
        mu (float): Initial learning rate (step size).
        lambda_ (float): Step size decay rate over time.
        K (int): Number of filter coefficients.

    Returns:
        np.ndarray: Error signal (i.e., estimated clean signal).

    Raises:
        ValueError: If signals differ in length, are too short, or if parameters are invalid.
    """
    N = len(d)
    if len(x) != N:
        raise ValueError("Signals 'd' and 'x' must have the same length.")
    if N < K:
        raise ValueError("Signal length must be at least equal to filter length K.")
    if mu <= 0 or lambda_ < 0:
        raise ValueError("Step size 'mu' must be > 0 and 'lambda' must be ≥ 0.")

    # Initialize filter weights and output error signal
    w = np.zeros(K, dtype=np.float32)
    e = np.zeros(N, dtype=np.float32)
    
    # Loop over signal, starting after enough samples to fill filter window
    for n in range(K - 1, N):
        # Get most recent K samples, reversed for causal filtering
        x_vec = x[n - K + 1 : n + 1][::-1]
        # Filter output 
        y = np.dot(w, x_vec)
        # Error = desired - prediction
        e[n] = d[n] - y
        # Update learning rate over time
        t = n - (K - 1)
        mu_n = mu / (1 + lambda_ * t)
        w += mu_n * x_vec * e[n]

    return e


def filter_signal_vslms(
    noisy_signal: np.ndarray,
    noise: np.ndarray,
    K: int,
    args: dict,
) -> np.ndarray | None:
    """Applies VSLMS adaptive filtering to denoise a signal.

    Wraps the core VSLMS function and handles parameter input and exceptions.

    Args:
        noisy_signal (np.ndarray): Signal to denoise (contains both signal + noise).
        noise (np.ndarray): Reference noise-only signal.
        K (int): Filter length (number of coefficients).
        args (dict): Optional hyperparameters:
            - 'mu' (float): Initial step size. Default is 0.01.
            - 'lambda' (float): Decay rate for the step size. Default is 0.01.

    Returns:
        np.ndarray | None: Error signal (denoised output), or None on failure.
    """
    try:
        mu = float(args.get("mu", 0.01))
        lambda_ = float(args.get("lambda", 0.01))
        return _vslms_core(noisy_signal, noise, mu, lambda_, K)

    except Exception as err:
        print(f"[VSLMS Error] {err}")
        return None