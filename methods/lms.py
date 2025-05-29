# methods/lms.py

import numpy as np


def _lms_core(
    d: np.ndarray,
    x: np.ndarray,
    mu: float,
    K: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Core implementation of the Least Mean Squares (LMS) adaptive filter.

    The LMS algorithm adaptively estimates filter coefficients to minimize the
    error between the desired signal and the filter output.

    Args:
        d (np.ndarray): Desired signal (typically the noisy target).
        x (np.ndarray): Reference input signal (typically the noise source).
        mu (float): Adaptation step size (learning rate).
        K (int): Filter length (number of coefficients).

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: 
            - e: Error signal over time (estimated clean signal).
            - w_hist: List containing the filter coefficient vector at each time step.

    Raises:
        ValueError: If input lengths are inconsistent, if signal is too short,
                    or if mu is non-positive.
    """
    N = len(d)
    if len(x) != N:
        raise ValueError("Signals 'd' and 'x' must have the same length.")
    if N < K:
        raise ValueError("Signal length must be at least equal to filter length K.")
    if mu <= 0:
        raise ValueError("Step size 'mu' must be greater than zero.")

    # Initialize filter weights, history, and error signal
    w = np.zeros(K, dtype=np.float32)
    w_hist = [np.copy(w) for _ in range(N)]
    e = np.zeros(N, dtype=np.float32)

    # Run LMS update for each time step
    for n in range(K - 1, N):
        # Extract most recent K samples (reversed for causal filtering)
        x_vec = x[n - K + 1 : n + 1][::-1]
        # Filter output
        y = np.dot(w, x_vec)
        # Error between desired and predicted
        e[n] = d[n] - y
        # LMS weight update
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
    """Applies LMS adaptive filtering to denoise a signal.

    Wraps `_lms_core` with error handling and default parameter support.

    Args:
        noisy_signal (np.ndarray): Observed signal with embedded noise.
        noise (np.ndarray): Reference noise-only signal.
        K (int): Filter length (number of coefficients).
        args (dict): Optional parameters:
            - 'mu' (float): Step size (learning rate). Default is 0.1.

    Returns:
        tuple[np.ndarray, list[np.ndarray]] | None:
            - e: Error signal (estimated clean signal).
            - w_hist: List of filter weights at each time step.
            Returns None in case of an error.
    """
    try:
        mu = float(args.get("mu", 0.1))
        return _lms_core(noisy_signal, noise, mu, K)

    except Exception as err:
        print(f"[LMS Error] {err}")
        return None
