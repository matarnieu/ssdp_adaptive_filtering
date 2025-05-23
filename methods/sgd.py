import numpy as np


def _update_mu(args, t):
    """
    if mode is specified, will use a time varying mu.
    """
    mode = args.get("mode", None)
    mu = args.get("mu")

    if mode == None:
        return mu

    elif mode == "inverse_time":
        lambda_ = args.get("lambda")
        return mu / (1 + lambda_ * t)
    else:
        raise NotImplemented


"""Use stochastic gradient descent to extract filtered signal from
noisy_signal and noise (numpy arrays). Approximate K-tap filter. Return filtered_signal.
In case of error, print error message and return None."""


def filter_signal_sgd(noisy_signal, noise, K, args):
    # d[n] = noisy_signal
    # x[n] = noise
    N = noisy_signal.shape[0]
    filter_ = np.zeros(K)
    output = np.zeros(N)
    for n in range(K, N):
        Xn = noise[n - K : n][::-1]
        dn = noisy_signal[n]
        y = np.dot(Xn, filter_)
        e = dn - y
        output[n] = e

        t = n - K
        mu_n = _update_mu(args, t)
        filter_ = filter_ + mu_n * Xn * e
    return output
