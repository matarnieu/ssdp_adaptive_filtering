# methods/lms.py

import numpy as np


# --------------------------------------------------------------------- #
def _lms_core(
    d: np.ndarray,
    x: np.ndarray,
    mu: float,
    K: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Core LMS adaptive filter.

    Parameters
    ----------
    d   : np.ndarray
        Desired signal (noisy observation).
    x   : np.ndarray
        Input signal (reference noise).
    mu  : float
        Adaptation step size (learning rate).
    K   : int
        Filter length (number of coefficients).

    Returns
    -------
    e   : np.ndarray
        Error signal (denoised output).
    w_hist : list of np.ndarray
        History of filter coefficients at each time step.
    """
    N = len(d)
    if len(x) != N:
        raise ValueError("Signals 'd' and 'x' must have the same length.")
    if N < K:
        raise ValueError("Signal length must be at least equal to filter length K.")
    if mu <= 0:
        raise ValueError("Step size 'mu' must be greater than zero.")

    w = np.zeros(K, dtype=np.float32)
    w_hist = [np.copy(w) for _ in range(N)]
    e = np.zeros(N, dtype=np.float32)

    for n in range(K - 1, N):
        x_vec = x[n - K + 1 : n + 1][::-1]
        y = np.dot(w, x_vec)
        e[n] = d[n] - y
        w += mu * x_vec * e[n]
        w_hist[n] = np.copy(w)

    return e, w_hist


# --------------------------------------------------------------------- #
def filter_signal_lms(
    noisy_signal: np.ndarray,
    noise: np.ndarray,
    K: int,
    args: dict,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """
    Apply LMS adaptive filtering to denoise a signal.

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
            - 'mu': step size (default = 0.1)

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        Error signal and filter coefficient history,
        or None on error.
    """
    try:
        mu = float(args.get("mu", 0.1))
        return _lms_core(noisy_signal, noise, mu, K)

    except Exception as err:
        print(f"[LMS Error] {err}")
        return None
