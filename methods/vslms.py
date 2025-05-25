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
        N = noisy_signal.shape[0]
        # Check that noise and signal have the same length
        if noise.shape[0] != N:
            print("Error: noisy_signal and noise must have the same length.")
            return None

        # Signal must be at least as long as the filter length
        if N < K:
            print("Error: signal length must be at least equal to K.")
            return None

        mu = args["mu"]
        lambda_ = args["lambda"]

        # Basic validation
        if lambda_ < 0 or mu <= 0:
            print("Error: 'mu' must be > 0 and 'lambda' must be >= 0.")
            return None

        filter_ = np.zeros(K)
        filter_history = [np.copy(filter_) for _ in range(N)]
        output = np.zeros(N)
        for n in range(K - 1, N):
            # Get the current input window (reversed for convolution)
            Xn = noise[n - K + 1 : n + 1][::-1]
            # Desired signal (noisy observation)
            dn = noisy_signal[n]
            # Filter output
            y = np.dot(Xn, filter_)
            # Error between desired and output
            e = dn - y
            # Save error (filtered signal)
            output[n] = e
            t = n - (K - 1)  # makes t start at 0
            # Update step-size (decays with time)
            mu_n = mu / (1 + lambda_ * t)
            # Update filter coefficients using LMS rule
            filter_ += mu_n * Xn * e
            filter_history[n] = np.copy(filter_)

        return output, filter_history

    except Exception as e:
        print(f"Error: {e}")
        mu = float(args.get("mu", 0.01))
        lambda_ = float(args.get("lambda", 0.01))
        return _vslms_core(noisy_signal, noise, mu, lambda_, K)

    except Exception as err:
        print(f"[VSLMS Error] {err}")
        return None
