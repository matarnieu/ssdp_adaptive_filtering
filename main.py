import sys
import random
import argparse
import numpy as np

# Import adaptive filtering algorithms
from methods.generalized_wiener_filter import filter_gwf
from methods.sgd import filter_signal_sgd
from methods.baseline import get_baseline_signal

from data_loader import load_real_data, generate_synthetic_data
from analyzer import plot_signals, compute_mse, plot_mses, plot_psd, plot_snr_sweep

import os

# Seed for reproducibility of randomness
DEFAULT_SEED = 42
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)
# Paths to real data
BEST_REAL_K = 500  # TODO: Change
DATA_PATH = "data"
NOISY_SIGNAL_PATH = os.path.join(DATA_PATH, "bassLineTalkingNoise.mp3")
NOISE_SIGNAL_PATH = os.path.join(DATA_PATH, "talkingNoise.mp3")

# All adaptive filtering algorithms used, as well as the cmd line arguments used to select them
methods = {
    "gwf": filter_gwf,
    "sgd": filter_signal_sgd,
    "all": None,
}

# Baseline signal
true_signal = None
# Different synthetic filter sizes that should be tried out
K_true = None
Ks_to_try = None
# Baseline Signal + Noise that went through filter (e.g. echo)
noisy_signal = None
# Noise before it went through filter
noise = None
# Filtering method used
filter_signal = None

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(
    description="Adaptive filtering on real or synthetic signals."
)
parser.add_argument(
    "data",
    choices=["real", "synthetic"],
    help="Type of data to use: 'real' or 'synthetic'",
)
parser.add_argument(
    "method",
    choices=["gwf", "sgd", "all"],
    help="Filtering method to apply: 'gwf', 'sgd', or 'all'",
)
parser.add_argument(
    "--K",
    choices=["best", "sweep"],
    default="best",
    help="Which K strategy should be used: 'best' (default) or 'sweep'",
)

# Optional arguments (only needed for synthetic data)
parser.add_argument(
    "--snr",
    type=float,
    help="SNR (Signal-to-Noise Ratio), required for synthetic data",
)

parser.add_argument(
    "--filter_type",
    choices=["moving_average", "exponential_decay"],
    help="Filter type used to generate synthetic data",
)

parser.add_argument(
    "--filter_size",
    type=int,
    help="Size of impulse response of filter used to generate synthetic data",
)

args = parser.parse_args()

# --- Validate required arguments for synthetic ---
if args.data == "synthetic":
    if args.snr is None or args.filter_type is None or args.filter_size is None:
        parser.error(
            "When using synthetic data, you must provide --snr , --filter_type and --filter_size."
        )
        sys.exit(1)

if args.data == "real":
    res = load_real_data(NOISY_SIGNAL_PATH, NOISE_SIGNAL_PATH)
    # Check for error
    if res is None:
        sys.exit(1)
    else:
        noisy_signal, noise = res
        K_true = BEST_REAL_K
        # Either try different filter sizes or use the best one
        if args.K == "best":
            # Use best filter size
            Ks_to_try = [BEST_REAL_K]
        else:
            Ks_to_try = [50, 100, 200, 500, 1000]
        # Get true signal, if available
        try:
            true_signal = get_baseline_signal(noisy_signal, noise)
        except NotImplementedError:
            print("Warning: Baseline signal extraction not implemented.")
            true_signal = None
else:
    res = generate_synthetic_data(args.filter_type, args.snr, args.filter_size)
    if res is None:
        sys.exit(1)
    else:
        true_signal, noisy_signal, noise, K_true = res
        # Either try different filter sizes or use the true one
        if args.K == "best":
            Ks_to_try = [K_true]
        else:
            Ks_to_try = list(range(1, K_true + 5))

# --- RUN SELECTED FILTERING METHOD ---

if args.method not in methods.keys():
    print(f"Invalid method. Choose one of {methods.keys()}.")
    sys.exit(1)
else:
    # Select adaptive filtering method
    filter_signal = methods[args.method]

print(f"Running '{args.method}' method on '{args.data}' data...")

# Perform adaptive filtering
if filter_signal is not None:
    # TEST ONE METHOD

    # Try out for each filter size
    for K in Ks_to_try:
        print(f"Trying filter size {K}...")
        # TODO: Measure running time
        filtered_signal = filter_signal(noisy_signal, noise, K)
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
            mse = compute_mse(true_signal, filtered_signal, f"{args.method} (K={K})")
            print(f"MSE: {mse}")
            # TODO: Compute and measure more stuff...

        plot_signals(signals_to_plot, title=f"{args.method} (K={K})")
        # plot_psd(signals_to_plot)
else:
    # TEST AND COMPARE ALL METHODS
    filtered_signals = []

    for method_name, fs in methods.items():
        if fs is None:
            continue
        # Filter with each method
        filtered_signal = fs(noisy_signal, noise, K=K_true)
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

    plot_signals(
        signals_to_plot,
    )
