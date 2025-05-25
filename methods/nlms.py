# methods/nlms.py
"""
Normalized LMS noise canceller with optional *Nit* inner iterations.
Defaults are tuned for robust, low-misadjustment performance on
float-scale audio (±1.0).
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------- #
def _nlms_core(
    d: np.ndarray,
    x: np.ndarray,
    mu: float,
    K: int,
    eps: float,
    Nit: int,
) -> np.ndarray:
    """Vectorised NLMS.

    Returns
    -------
    e : ndarray
        The cleaned/error signal  e[n] = d[n] − wᵀx.
    """
    N = len(d)
    x_pad = np.concatenate([x, np.zeros(K)])
    w = np.zeros(K, dtype=np.float32)
    e = np.empty(N, dtype=np.float32)

    for n in range(N):
        x_vec = x_pad[n : n + K][::-1]

        # --- inner gradient steps -------------------------------------
        for _ in range(Nit):
            err = d[n] - np.dot(w, x_vec)
            w += (mu / (np.dot(x_vec, x_vec) + eps)) * err * x_vec

        # final error after Nit updates
        e[n] = d[n] - np.dot(w, x_vec)

    return e


# --------------------------------------------------------------------- #
# Required entry-point
# --------------------------------------------------------------------- #
def filter_signal_nlms(
    noisy_signal: np.ndarray,
    noise: np.ndarray,
    K: int,
    args: dict,
) -> np.ndarray:
    """
    Parameters
    ----------
    mu   : float, default 0.05
        Step size (recommend 0 < μ ≤ 0.2 when Nit ≥ 5).
    eps  : float, default 1e-3
        Regulariser in the µ_normalisation term.
    Nit  : int,   default 5
        Inner iterations per sample.
    """
    mu  = float(args.get("mu", 0.05))
    eps = float(args.get("eps", 1e-3))
    Nit = int  (args.get("Nit", 5))
    return _nlms_core(noisy_signal, noise, mu, K, eps, Nit)
