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
    N = len(d)
    x_pad = np.concatenate([x, np.zeros(K)])
    w = np.zeros(K, dtype=np.float32)
    P = np.eye(K, dtype=np.float32) / delta
    e = np.empty(N, dtype=np.float32)

    for n in range(N):
        x_vec = x_pad[n : n + K][::-1]
        pi = P @ x_vec
        k = pi / (lam + x_vec @ pi)
        y = np.dot(w, x_vec)
        e[n] = d[n] - y
        w += k * e[n]
        P = (P - np.outer(k, pi)) / lam

    return e


# --------------------------------------------------------------------- #
# Required entry-point
# --------------------------------------------------------------------- #
def filter_signal_rls(
    noisy_signal: np.ndarray,
    noise: np.ndarray,
    K: int,
    args: dict,
) -> np.ndarray:
    """
    Parameters
    ----------
    lam   : float, default 0.998
        Forgetting factor (closer to 1 → slower forgetting).
    delta : float, default 0.1
        Initial diagonal loading for P(0)=I/δ (smaller → faster conv., but
        beware of ill-conditioning).
    """
    lam   = float(args.get("lam",   0.998))
    delta = float(args.get("delta", 0.1))
    return _rls_core(noisy_signal, noise, lam, delta, K)
