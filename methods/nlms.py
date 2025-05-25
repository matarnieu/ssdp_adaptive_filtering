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
    """
    Core Normalized LMS (NLMS) adaptive filter.

    Parameters
    ----------
    d    : np.ndarray
        Desired signal (noisy observation).
    x    : np.ndarray
        Input signal (reference noise).
    mu   : float
        Base step size (learning rate).
    eps  : float
        Regularization term to prevent division by zero.
    K    : int
        Filter length (number of coefficients).

    Returns
    -------
    e     : np.ndarray
        Error signal (denoised output).
    w_hist: list of np.ndarray
        History of filter coefficients at each time step.
    """
    N = len(d)
    if len(x) != N:
        raise ValueError("Signals 'd' and 'x' must have the same length.")
    if N < K:
        raise ValueError("Signal length must be at least equal to filter length K.")
    if mu <= 0 or eps <= 0:
        raise ValueError("Parameters 'mu' and 'eps' must be greater than zero.")

    w = np.zeros(K, dtype=np.float32)
    e = np.zeros(N, dtype=np.float32)
    w_hist = [np.copy(w) for _ in range(N)]

    for n in range(K - 1, N):
        x_vec = x[n - K + 1 : n + 1][::-1]
        y = np.dot(w, x_vec)
        e[n] = d[n] - y
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
    """
    Apply Normalized LMS (NLMS) adaptive filtering to denoise a signal.

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
            - 'mu'  : base step size (default = 0.1)
            - 'eps' : regularization term (default = 1e-8)

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        Error signal and filter coefficient history, or None on error.
    """
    try:
        mu = float(args.get("mu", 0.1))
        eps = float(args.get("eps", 1e-8))
        return _nlms_core(noisy_signal, noise, mu, eps, K)

    except Exception as err:
        print(f"[NLMS Error] {err}")
        return None
