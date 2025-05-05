"""Visualize list of signals with given labels.
Input signals is an array of tuples (signal_description, signal[np array])"""


def plot_signals(signals):
    raise NotImplementedError


"""Compute MSE between two signals represented as numpy arrays"""


def compute_mse(true_signal, filtered_signal):
    raise NotImplementedError


"""Plot MSE for multiple filtered signals computed with different methods.
Input filtered_signals is an array of tuples (signal_description, signal[np array])"""


def plot_mses(true_signal, filtered_signals):
    raise NotImplementedError
