import sys
import random
import argparse
import numpy as np

# Import adaptive filtering algorithms
from methods.generalized_wiener_filter import (
    filter_signal_gwf_fc,
    filter_signal_gwf_swc,
    filter_signal_gwf_ema,
)
from methods.sgd import filter_signal_sgd
from methods.baseline import get_baseline_signal
from methods.nlms import filter_signal_nlms
from methods.rls import filter_signal_rls

from data_loader import load_real_data, generate_synthetic_data
from analyzer import plot_signals, compute_mse, plot_mses, plot_psd, plot_snr_sweep

import os

# Number of samples in synthetic signal
NUM_SAMPLES = 10000
SWITCHING_INTERVAL = 1300
# Define synthetic sinus signal
LOW, HIGH = 0, 1000 * np.pi
# Seed for reproducibility of randomness
DEFAULT_SEED = 41
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)
# Paths to real data
BEST_REAL_K = 500  # TODO: Change
DATA_PATH = "data"
NOISY_SIGNAL_PATH = os.path.join(DATA_PATH, "bassLineTalkingNoise.mp3")
NOISE_SIGNAL_PATH = os.path.join(DATA_PATH, "talkingNoise.mp3")

# TITLE = "GWF - Sample Correlation"

# All adaptive filtering algorithms used, as well as the cmd line arguments used to select them
methods = {
    "gwf_fc": filter_signal_gwf_fc,
    "gwf_swc": filter_signal_gwf_swc,
    "gwf_ema": filter_signal_gwf_ema,
    "sgd": filter_signal_sgd,
    "nlms": filter_signal_nlms,
    "rls": filter_signal_rls,
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
filter_function = None

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
    choices=list(methods.keys()),
    help="Filtering method to apply.",
)
parser.add_argument(
    "--dont_show_plot",
    dest="dont_show_plot",
    action="store_true",
    default=False,
    help="Dont show plot",
)

parser.add_argument(
    "--plot_filename",
    dest="plot_filename",
    default=None,
    help="Filename of stored plot",
)

parser.add_argument(
    "--K_sweep",
    dest="K_sweep",
    action="store_true",
    default=False,
    help="Which K strategy should be used: 'best' (default) or 'sweep'",
)

# Optional arguments (only needed for synthetic data)
parser.add_argument(
    "--noise_power",
    type=float,
    help="Power of X, required for synthetic data",
)

parser.add_argument(
    "--noise_power_change",
    dest="noise_power_change",
    action="store_true",
    default=False,
    help="Add this flag to make noise power change over time (piecewise std).",
)

parser.add_argument(
    "--noise_distribution_change",
    dest="noise_distribution_change",
    action="store_true",
    default=False,
    help="Add this flag to make noise distribution change over time (e.g., alternating between Gaussian and chaotic).",
)

parser.add_argument(
    "--filter_type",
    choices=["moving_average", "exponential_decay", "mixed"],
    help="Filter type used to generate synthetic data",
)

parser.add_argument(
    "--filter_changing_speed",
    type=float,
    default=0.0,
    help="How quickly the parameters (smoothly) change when generating the filter for the synthetic data",
)

parser.add_argument(
    "--filter_size",
    type=int,
    help="Size of impulse response of filter used to generate synthetic data",
)

args, unknown = parser.parse_known_args()
# Convert unknown args into a dictionary
extra_args = {}
for arg in unknown:
    if arg.startswith("--"):
        key_val = arg.lstrip("--").split("=", 1)
        if len(key_val) == 2:
            key, val = key_val
            try:
                extra_args[key] = float(val)
            except ValueError:
                extra_args[key] = val  # fallback to string
        else:
            key = key_val[0]
            extra_args[key] = True  # handle flags without value

# --- Validate required arguments for synthetic ---
if args.data == "synthetic":
    if args.noise_power is None or args.filter_type is None or args.filter_size is None:
        parser.error(
            "When using synthetic data, you must provide --noise_power , --filter_type and --filter_size."
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
        if not args.K_sweep:
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
    res = generate_synthetic_data(
        NUM_SAMPLES,
        LOW,
        HIGH,
        SWITCHING_INTERVAL,
        # Filter parameters
        args.filter_size,
        args.filter_type,
        args.filter_changing_speed,
        # Noise parameters
        args.noise_power,
        args.noise_power_change,
        args.noise_distribution_change,
    )
    if res is None:
        sys.exit(1)
    else:
        noisy_signal, true_signal, noise, _ = res
        K_true = args.filter_size
        # Either try different filter sizes or use the true one
        if not args.K_sweep:
            Ks_to_try = [K_true]
        else:
            Ks_to_try = list(range(1, K_true + 5))


# --- RUN SELECTED FILTERING METHOD ---

if args.method not in methods.keys():
    print(f"Invalid method. Choose one of {methods.keys()}.")
    sys.exit(1)
else:
    # Select adaptive filtering method
    filter_function = methods[args.method]

print(f"Running '{args.method}' method on '{args.data}' data...")

# Perform adaptive filtering
if filter_function is not None:
    # TEST ONE METHOD

    # Try out for each filter size
    for K in Ks_to_try:
        print(f"Trying filter size {K}...")
        # TODO: Measure running time
        filtered_signal = filter_function(noisy_signal, noise, K, extra_args)
        if filtered_signal is None:
            sys.exit(1)

        # Visualize results
        signals_to_plot = [
            ("Noisy signal", noisy_signal),
            ("Filtered signal", filtered_signal),
        ]

        if true_signal is not None:
            # Plot true signal, if available
            signals_to_plot.append(("True signal", true_signal))
            # Compute MSE
            mse_before_filtering = compute_mse(true_signal, noisy_signal)
            print(f"MSE before filtering: {mse_before_filtering}")
            mse = compute_mse(true_signal, filtered_signal)
            print(f"MSE: {mse}")
            # TODO: Compute and measure more stuff...

        plot_signals(
            signals_to_plot,
            filename=args.plot_filename,
            show=not args.dont_show_plot,
        )
        # plot_psd(signals_to_plot)
else:
    # TEST AND COMPARE ALL METHODS
    filtered_signals = []

    for method_name, fs in methods.items():
        if fs is None:
            continue
        # Filter with each method
        filtered_signal = fs(noisy_signal, noise, K_true, extra_args)
        if filtered_signal is None:
            sys.exit(1)
        filtered_signals.append((method_name, filtered_signal))

    signals_to_plot = filtered_signals.copy()
    signals_to_plot.insert(0, ("Noisy signal", noisy_signal))

    if true_signal is not None:
        # Plot true signal, if available
        signals_to_plot.append(("True signal", true_signal))
        # Plot MSE's for different methods
        plot_mses(true_signal, filtered_signals)

    plot_signals(
        signals_to_plot,
    )
