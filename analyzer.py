import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch

def plot_signals(signals, title, window=None):
    """
    Visualize list of signals with given labels.
    signals: list of (label, np.array)
    window: tuple (start, end) to zoom in, defaults to entire signal
    """
    plt.figure(figsize=(10, 4))
    for label, sig in signals:
        if window:
            start, end = window
            sig = sig[start:end]
            x = np.arange(start, end)
        else:
            x = np.arange(len(sig))
        plt.plot(x, sig, label=label)
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.title(f'{title}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_mse(true_signal, filtered_signal):
    """
    Compute MSE between two signals represented as numpy arrays.
    """
    true_signal = np.asarray(true_signal)
    filtered_signal = np.asarray(filtered_signal)
    mse = np.mean((true_signal - filtered_signal) ** 2)
    return mse

def plot_mses(true_signal, filtered_signals, title):
    """
    Plot MSE for multiple filtered signals.
    filtered_signals: list of (label, np.array)
    """
    mses = []
    labels = []
    for label, sig in filtered_signals:
        mse = compute_mse(true_signal, sig)
        mses.append(mse)
        labels.append(label)
    plt.figure(figsize=(6, 4))
    plt.bar(labels, mses)
    plt.xlabel('Method')
    plt.ylabel('MSE')
    plt.title(f'{title}')
    plt.tight_layout()
    plt.show()


def plot_error_convergence(error_histories, title, labels=None):
    """
    Plot error convergence curves.
    error_histories: list of 1D arrays of error metric per sample/iteration
    labels: list of labels for each history
    """
    plt.figure(figsize=(8, 4))
    for idx, hist in enumerate(error_histories):
        label = labels[idx] if labels else f'Method {idx}'
        plt.plot(np.arange(len(hist)), hist, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(f'{title}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_psd(signals, fs, title, labels=None, nperseg=None):
    """
    Plot Power Spectral Density of signals.
    signals: list of np.array
    fs: sampling frequency
    labels: list of labels
    """
    plt.figure(figsize=(8, 4))
    for idx, sig in enumerate(signals):
        f, Pxx = welch(sig, fs=fs, nperseg=nperseg)
        label = labels[idx] if labels else f'Signal {idx}'
        plt.semilogy(f, Pxx, label=label)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.title(f'{title}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_histogram(error, title, bins=50):
    """
    Plot histogram of error signal.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(error, bins=bins, edgecolor='k')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f'{title}')
    plt.tight_layout()
    plt.show()


def plot_residual_autocorr(error, title, max_lag=100):
    """
    Plot autocorrelation of residual error.
    """
    error = np.asarray(error)
    error = error - np.mean(error)
    acf_full = np.correlate(error, error, mode='full')
    acf = acf_full[acf_full.size // 2:]
    acf = acf / acf[0]  # normalize
    lags = np.arange(len(acf))
    plt.figure(figsize=(6, 4))
    plt.stem(lags[:max_lag], acf[:max_lag], use_line_collection=True)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'{title}')
    plt.tight_layout()
    plt.show()


def plot_snr_sweep(results_df, title):
    """
    Plot performance vs SNR sweep.
    results_df: pandas.DataFrame with columns ['snr', 'method', 'mse']
    """
    import pandas as pd
    df = results_df.copy()
    pivot = df.pivot(index='snr', columns='method', values='mse')
    plt.figure(figsize=(8, 5))
    for method in pivot.columns:
        plt.plot(pivot.index, pivot[method], marker='o', label=method)
    plt.xlabel('Input SNR (dB)')
    plt.ylabel('MSE')
    plt.title(f'{title}')
    plt.legend()
    plt.tight_layout()
    plt.show()
