# methods/freq_kalman.py   (patched)

import numpy as np
from numpy.fft import rfft, irfft


def _fkf_core(d, x, K, Q, R, dtype=np.float64):
    d = np.asarray(d, dtype=dtype)
    x = np.asarray(x, dtype=dtype)

    N = len(d)
    L = 2 * K                       # FFT length
    n_bins = L // 2 + 1

    if len(x) < N + K:
        x = np.concatenate([x, np.zeros(N + K - len(x), dtype=dtype)])
    x_pad = np.concatenate([x, np.zeros(K, dtype=dtype)])

    W_f = np.zeros(n_bins, dtype=np.complex128)
    P   = np.ones(n_bins, dtype=dtype) * 1e2

    Q = float(Q)
    R = float(R)

    out_err = np.empty(N, dtype=dtype)
    x_buf   = np.zeros(L, dtype=dtype)

    hop = K
    n = 0
    while n < N:
        # ---- overlap-save buffer ------------------------------------
        x_buf[:-hop] = x_buf[hop:]
        x_buf[-hop:] = x_pad[n : n + hop]
        X_f = rfft(x_buf)

        Y_f = W_f * X_f
        y_block = irfft(Y_f, L)[hop:]         # last K samples

        d_block = d[n : n + hop]
        if len(d_block) < hop:                # final partial block
            y_block = y_block[:len(d_block)]

        err_block = d_block - y_block

        # ---------- store safely -------------------------------------
        avail = len(err_block)                # may be < hop at final block
        out_err[n : n + avail] = err_block

        # ---------- Kalman update (full hop length) ------------------
        # zero-pad error to hop so FFT sizes stay consistent
        E_pad = np.concatenate(
            (np.zeros(hop, dtype=dtype),           # overlap-save rule
             err_block,
             np.zeros(hop - avail, dtype=dtype))   # only if final block short
        )
        E_f = rfft(E_pad, L)

        X_pwr = (X_f.conj() * X_f).real + 1e-12
        S     = P * X_pwr + R
        K_gain = P * X_f.conj() / S
        W_f  += K_gain * E_f
        P     = (1.0 - (K_gain * X_f).real) * P + Q

        n += avail                              # move exactly by samples processed

    return out_err[:N]


# public wrapper unchanged
def filter_signal_fkf(noisy_signal, noise, K, args):
    Q = float(args.get("Q", 1e-7))
    R = float(args.get("R", 0.3))
    return _fkf_core(noisy_signal, noise, K, Q, R)
