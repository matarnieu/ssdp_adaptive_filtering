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
    """Core implementation of the Recursive Least Squares (RLS) adaptive filter.

    Uses matrix-based RLS update to adaptively estimate filter coefficients for
    denoising a signal with a reference noise input.

    Args:
        d (np.ndarray): Desired signal (e.g., noisy target).
        x (np.ndarray): Reference noise-only signal.
        lam (float): Forgetting factor (0 < lam ≤ 1). Controls memory decay.
        delta (float): Initial diagonal value of the inverse correlation matrix P.
        K (int): Filter order (number of taps/coefficients).

    Returns:
        np.ndarray: Error signal (i.e., estimated clean signal).

    Raises:
        ValueError: If input lengths are inconsistent or signal is too short.
    """
    N = len(d)
    if len(x) != N:
        raise ValueError("Signals 'd' and 'x' must have the same length.")
    if N < K:
        raise ValueError("Signal length must be at least equal to filter length K.")
    
    # Initialize weights and inverse correlation matrix
    w = np.zeros(K, dtype=np.float32)
    P = np.eye(K, dtype=np.float32) / delta
    e = np.zeros(N, dtype=np.float32)
    
    # Adaptive filtering loop
    for n in range(K - 1, N):
        # Get the K most recent samples (reversed for causal filtering)
        x_vec = x[n - K + 1 : n + 1][::-1]
        # Compute gain vector
        pi = P @ x_vec
        k = pi / (lam + x_vec @ pi)
        # Predicted output
        y = np.dot(w, x_vec)
        # Error between desired and predicted
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
    """Applies Recursive Least Squares (RLS) adaptive filtering to denoise a signal.

    Wraps the core RLS function and handles argument parsing and exceptions.

    Args:
        noisy_signal (np.ndarray): Signal to denoise (noisy observation).
        noise (np.ndarray): Reference noise-only signal.
        K (int): Length of the adaptive filter.
        args (dict): Optional parameters:
            - 'lambda' (float): Forgetting factor. Default is 0.998.
            - 'delta' (float): Initial value for inverse correlation matrix. Default is 0.1.

    Returns:
        np.ndarray | None: Error signal (estimated clean output), or None on failure.
    """
    try:
        lambda_ = float(args.get("lambda", 0.998))
        delta = float(args.get("delta", 0.1))

        return _rls_core(noisy_signal, noise, lambda_, delta, K)

    except Exception as err:
        print(f"[RLS Error] {err}")
        return None
