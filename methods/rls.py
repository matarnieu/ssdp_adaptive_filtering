# methods/rls.py

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------- #
def _rls_core(
    d: np.ndarray,
    x: np.ndarray,
    lam: float,
    delta: float,
    K: int,
) -> np.ndarray:
    """
    Core RLS adaptive filter.

    Parameters
    ----------
    d     : np.ndarray
        Desired signal (noisy signal).
    x     : np.ndarray
        Input signal (reference noise).
    lam   : float
        Forgetting factor (0 < lam ≤ 1).
    delta : float
        Initial value for the inverse correlation matrix.
    K     : int
        Filter order (number of taps).

    Returns
    -------
    e     : np.ndarray
        Error signal (estimated clean signal).
    """
    N = len(d)
    if len(x) != N:
        raise ValueError("Signals 'd' and 'x' must have the same length.")
    if N < K:
        raise ValueError("Signal length must be at least equal to filter length K.")

    w = np.zeros(K, dtype=np.float32)
    P = np.eye(K, dtype=np.float32) / delta
    e = np.zeros(N, dtype=np.float32)

    for n in range(K - 1, N):
        x_vec = x[n - K + 1 : n + 1][::-1]
        pi = P @ x_vec
        k = pi / (lam + x_vec @ pi)
        y = np.dot(w, x_vec)
        e[n] = d[n] - y
        w += k * e[n]
        P = (P - np.outer(k, pi)) / lam

    return e


# --------------------------------------------------------------------- #
def filter_signal_rls(
    noisy_signal: np.ndarray,
    noise: np.ndarray,
    K: int,
    args: dict,
) -> np.ndarray | None:
    """
    Apply RLS adaptive filtering to remove noise from a signal.

    Parameters
    ----------
    noisy_signal : np.ndarray
        Observed signal containing noise.
    noise        : np.ndarray
        Reference noise signal.
    K            : int
        Filter length (number of coefficients).
    args         : dict
        Optional parameters:
            - 'lam'   : forgetting factor (default = 0.998)
            - 'delta' : initial value for inverse correlation matrix (default = 0.1)

    Returns
    -------
    np.ndarray
        Filtered signal (same shape as input).
    """
    try:
        lam = float(args.get("lam", 0.998))
        delta = float(args.get("delta", 0.1))

        return _rls_core(noisy_signal, noise, lam, delta, K)

    except Exception as err:
        print(f"[RLS Error] {err}")
        return None
