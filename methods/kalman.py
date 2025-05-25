# methods/kalman.py
from __future__ import annotations
import numpy as np


# ------------------------------------------------------------------ #
def _kalman_core(
    d: np.ndarray,
    x: np.ndarray,
    Q: float,
    R: float,
    delta0: float,
    K: int,
) -> np.ndarray:
    """
    Low-level Kalman recursion (scalar Q, scalar R).

    delta0 sets the initial inverse-covariance P(0)=I/δ₀.
    """
    N = len(d)
    x_pad = np.concatenate([x.astype(np.float32), np.zeros(K, np.float32)])

    # state mean & covariance
    w = np.zeros(K, dtype=np.float32)
    P = np.eye(K, dtype=np.float32) / delta0     # <<< safer than 1e3·I

    I = np.eye(K, dtype=np.float32)
    e = np.empty(N, dtype=np.float32)

    for n in range(K-1, N):
        x_vec = x[n - K + 1 : n + 1][::-1] # latest K refs

        # -------- time update ------------------------------------------
        P += Q * I                               # random-walk model

        # -------- measurement update -----------------------------------
        Px   = P @ x_vec
        Kk   = Px / (x_vec @ Px + R)             # Kalman gain (K×1)
        err  = d[n] - w @ x_vec                  # a-priori error
        w   += Kk * err                          # posterior weights
        P   -= np.outer(Kk, Px)                  # Joseph form omitted

        e[n] = d[n] - w @ x_vec                  # a-posteriori error

    return e


# ------------------------------------------------------------------ #
# public entry-point used by main.py
# ------------------------------------------------------------------ #
def filter_signal_kalman(
    noisy_signal: np.ndarray,
    noise: np.ndarray,
    K: int,
    args: dict,
) -> np.ndarray:
    """
    Hyper-parameters (pass via CLI as --Q=… --R=… --delta0=…):

    Q      process-noise variance, default 1e-6
    R      measurement-noise variance, default var(noisy_signal[:1000])
    delta0 initial inverse-covariance factor (RLS analogue), default 0.1
    """
    Q      = float(args.get("Q",      1e-6))
    #R      = float(args.get("R",      np.var(noisy_signal[:1000])))
    R      = float(args.get("R",      1e-1))
    delta0 = float(args.get("delta0", 0.1))

    return _kalman_core(noisy_signal, noise, Q, R, delta0, K)
