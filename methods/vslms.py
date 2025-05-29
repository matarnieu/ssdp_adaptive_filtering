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
    """
    Core time-varying step-size LMS (VSLMS) adaptive filter.

    Parameters
    ----------
    d        : np.ndarray
        Desired signal (noisy observation).
    x        : np.ndarray
        Input signal (reference noise).
    mu       : float
        Initial step size (learning rate).
    lambda_  : float
        Decay rate of the step size over time.
    K        : int
        Filter length (number of coefficients).

    Returns
    -------
    e        : np.ndarray
        Error signal (denoised output).
    """
    N = len(d)
    if len(x) != N:
        raise ValueError("Signals 'd' and 'x' must have the same length.")
    if N < K:
        raise ValueError("Signal length must be at least equal to filter length K.")
    if mu <= 0 or lambda_ < 0:
        raise ValueError("Step size 'mu' must be > 0 and 'lambda' must be ≥ 0.")

    w = np.zeros(K, dtype=np.float32)
    e = np.zeros(N, dtype=np.float32)

    for n in range(K - 1, N):
        x_vec = x[n - K + 1 : n + 1][::-1]
        y = np.dot(w, x_vec)
        e[n] = d[n] - y
        t = n - (K - 1)
        mu_n = mu / (1 + lambda_ * t)
        w += mu_n * x_vec * e[n]

    return e


# --------------------------------------------------------------------- #
def filter_signal_vslms(
    noisy_signal: np.ndarray,
    noise: np.ndarray,
    K: int,
    args: dict,
) -> np.ndarray | None:
    """
    Apply time-varying step-size LMS to denoise a signal.

    Parameters
    ----------
    noisy_signal : np.ndarray
        Observed signal with noise.
    noise        : np.ndarray
        Reference noise signal.
    K            : int
        Filter length (number of coefficients).
    args         : dict
        Optional parameters:
            - 'mu'      : initial step size (default = 0.01)
            - 'lambda'  : decay rate (default = 0.01)

    Returns
    -------
    np.ndarray
        Denoised signal (error signal), or None on error.
    """
    try:
        mu = float(args.get("mu", 0.01))
        lambda_ = float(args.get("lambda", 0.01))
        return _vslms_core(noisy_signal, noise, mu, lambda_, K)

    except Exception as err:
        print(f"[VSLMS Error] {err}")
        return None