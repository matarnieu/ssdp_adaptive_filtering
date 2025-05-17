# methods/rls.py
import numpy as np

def _rls_core(d, x, lam, delta, K):
    """Vectorised RLS (noise canceller).  Returns e = signal estimate."""
    N = len(d)
    x_pad = np.concatenate([x, np.zeros(K)])
    w = np.zeros(K)
    P = np.eye(K) / delta
    e = np.empty(N)
    for n in range(N):
        x_vec = x_pad[n:n+K][::-1]
        pi = P @ x_vec
        k  = pi / (lam + x_vec @ pi)
        y  = w @ x_vec
        e[n] = d[n] - y
        w  += k * e[n]
        P  = (P - np.outer(k, pi)) / lam
    return e

#  Required entry-point -------------------------------------------------
def filter_signal_rls(noisy_signal, noise, K, args):
    lam   = float(args.get("lam",   0.999))
    delta = float(args.get("delta", 10.0))
    return _rls_core(noisy_signal, noise, lam, delta, K)
