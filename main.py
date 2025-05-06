import sys
import random
import numpy as np

# Import adaptive filtering algorithms
from methods.simple import filter_signal_simple
from methods.sgd import filter_signal_sgd

from data_loader import load_real_data, generate_synthetic_data
from analyzer import plot_signals, compute_mse, plot_mses

import os

# Seed for reproducibility of randomness
DEFAULT_SEED = 42
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)
# Paths to real data
DATA_PATH = "data"
NOISY_SIGNAL_PATH = os.path.join(DATA_PATH, "bassLineTalkingNoise.mp3")
NOISE_SIGNAL_PATH = os.path.join(DATA_PATH, "talkingNoise.mp3")

# All adaptive filtering algorithms used, as well as the cmd line arguments used to select them
methods = {
    "simple": filter_signal_simple,
    "sgd": filter_signal_sgd,
    "all": None,
}

# Baseline signal
true_signal = None
# Baseline Signal + Noise that went through filter (e.g. echo)
noisy_signal = None
# Noise before it went through filter
noise = None
# Filtering method used
filter_signal = None

# Get command line arguments
if len(sys.argv) != 3:
    print("Usage: python main.py <data: real|synthetic> <method: simple|sgd>")
    sys.exit(1)

data_arg = sys.argv[1].lower()
method_arg = sys.argv[2].lower()

if data_arg not in ["real", "synthetic"]:
    print("Invalid data type. Choose 'real' or 'synthetic'.")
    sys.exit(1)
else:
    # Load / create data
    if data_arg == "real":
        res = load_real_data(NOISY_SIGNAL_PATH, NOISE_SIGNAL_PATH)
        # Check for error
        if res is None:
            sys.exit(1)
        else:
            noisy_signal, noise = res
    else:
        res = generate_synthetic_data()
        if res is None:
            sys.exit(1)
        else:
            true_signal, noisy_signal, noise = res


if method_arg not in methods.keys():
    print(f"Invalid method. Choose one of {methods.keys()}.")
    sys.exit(1)
else:
    # Select adaptive filtering method
    filter_signal = methods[method_arg]

print(f"Running '{method_arg}' method on '{data_arg}' data...")

# Perform adaptive filtering
# TODO: Measure running time
if not filter_signal is None:
    # TEST ONE METHOD

    filtered_signal = filter_signal(noisy_signal, noise)
    if filtered_signal is None:
        sys.exit(1)

    # Visualize results
    signals_to_plot = [
        ("Noisy signal", noisy_signal),
        ("Filtered signal", filtered_signal),
    ]

    if true_signal is not None:
        # Plot true signal, if available
        signals_to_plot.insert(0, ("True signal", true_signal))
        # Compute MSE
        mse = compute_mse(true_signal, filtered_signal)
        print(f"MSE: {mse}")
        # TODO: Compute and measure more stuff...

    plot_signals(signals_to_plot)
else:
    # TEST AND COMPARE ALL METHODS
    filtered_signals = []

    for method_name, fs in methods.items():
        if fs is None:
            continue
        # Filter with each method
        filtered_signal = fs(noisy_signal, noise)
        if filtered_signal is None:
            sys.exit(1)
        filtered_signals.append((method_name, filtered_signal))

    signals_to_plot = filtered_signals.copy()
    signals_to_plot.insert(0, ("Noisy signal", noisy_signal))

    if true_signal is not None:
        # Plot true signal, if available
        signals_to_plot.insert(0, ("True signal", true_signal))
        # Plot MSE's for different methods
        plot_mses(true_signal, filtered_signals)

    plot_signals(signals_to_plot)
