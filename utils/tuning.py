import numpy as np
from methods.nlms import _nlms_core
from methods.rls  import _rls_core

def tune_nlms(d, x, s,
              mu_list=(0.4, 0.6, 0.8, 1.0),
              L_list=(32, 48, 64),
              eps=1e-6,
              val_len=1000):
    """Find (μ, K) minimizing MSE on the first val_len samples."""
    best = (np.inf, 0.4, 32)
    for μ in mu_list:
        for K in L_list:
            e = _nlms_core(d[:val_len], x[:val_len], μ, K, eps)
            m = np.mean((s[:val_len] - e)**2)
            if m < best[0]:
                best = (m, μ, K)
    return best[1], best[2]

def tune_rls(d, x, s,
             lam_list=(0.997, 0.998, 0.999),
             delta_list=(1, 5, 10),
             L_list=(32, 64),
             val_len=1000):
    """Find (λ, δ, K) minimizing MSE on the first val_len samples."""
    best = (np.inf, 0.999, 10, 32)
    for lam in lam_list:
        for δ in delta_list:
            for K in L_list:
                e = _rls_core(d[:val_len], x[:val_len], lam, δ, K)
                m = np.mean((s[:val_len] - e)**2)
                if m < best[0]:
                    best = (m, lam, δ, K)
    return best[1], best[2], best[3]
