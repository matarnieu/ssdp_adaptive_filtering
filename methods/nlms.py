# methods/nlms.py
import numpy as np

def _nlms_core(d, x, mu, K, eps):
    """Vectorised NLMS (noise canceller).  Returns e = signal estimate."""
    N = len(d)
    x_pad = np.concatenate([x, np.zeros(K)])
    w = np.zeros(K)
    e = np.empty(N)
    for n in range(N):
        x_vec = x_pad[n:n+K][::-1]
        y = w @ x_vec
        e[n] = d[n] - y
        w += (mu / (x_vec @ x_vec + eps)) * e[n] * x_vec
    return e

#  Required entry-point -------------------------------------------------
def filter_signal_nlms(noisy_signal, noise, K, args):
    """
    Parameters
    ----------
    noisy_signal : 1-D ndarray
        Primary input  D = S + h*X   (called `D` in the paper/README).
    noise : 1-D ndarray
        Reference noise  X  (uncorrelated with S).
    K : int
        FIR tap-size of the adaptive filter.
    args : dict
        Extra CLI parameters (floats / flags) e.g. {"mu":0.6,"eps":1e-6}.
    """
    mu  = float(args.get("mu", 0.8))
    eps = float(args.get("eps", 1e-6))
    return _nlms_core(noisy_signal, noise, mu, K, eps)
