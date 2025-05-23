import sys
import random
import argparse
import numpy as np
import os

# Import adaptive filtering algorithms
from methods.generalized_wiener_filter import (
    filter_signal_gwf_fc,
    filter_signal_gwf_swc,
    filter_signal_gwf_ema,
)
from methods.sgd import filter_signal_sgd
from methods.baseline import get_baseline_signal
from methods.nlms import filter_signal_nlms
from methods.rls  import filter_signal_rls

import utils.tuning as tuning

from data_loader import load_real_data, generate_synthetic_data
from analyzer import plot_signals, compute_mse, plot_mses, plot_psd, plot_snr_sweep

# Number of samples in synthetic signal
NUM_SAMPLES = 10000
SWITCHING_INTERVAL = 1400
LOW, HIGH = 0, 1000 * np.pi    # synthetic time axis bounds

# Seed for reproducibility
DEFAULT_SEED = 41
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)

# Paths to real data
BEST_REAL_K = 500  # TODO: adjust as needed
DATA_PATH = "data"
NOISY_SIGNAL_PATH = os.path.join(DATA_PATH, "bassLineTalkingNoise.mp3")
NOISE_SIGNAL_PATH = os.path.join(DATA_PATH, "talkingNoise.mp3")

# Available methods
methods = {
    "gwf_fc": filter_signal_gwf_fc,
    "gwf_swc": filter_signal_gwf_swc,
    "gwf_ema": filter_signal_gwf_ema,
    "sgd": filter_signal_sgd,
    "nlms": filter_signal_nlms,
    "rls": filter_signal_rls,
    "all": None,
}

# Placeholders
true_signal    = None
noisy_signal   = None
noise          = None
K_true         = None
Ks_to_try      = None
filter_function= None

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
    "--K_sweep",
    dest="K_sweep",
    action="store_true",
    default=False,
    help="Which K strategy should be used: 'best' (default) or 'sweep'",
)

# Synthetic-only options
parser.add_argument("--noise_power", type=float,
                    help="Power of X, required for synthetic data")
parser.add_argument("--noise_power_change", dest="noise_power_change",
                    action="store_true", default=False,
                    help="Make noise power change over time.")
parser.add_argument("--noise_distribution_change", dest="noise_distribution_change",
                    action="store_true", default=False,
                    help="Make noise distribution change over time.")
parser.add_argument("--filter_type",
                    choices=["moving_average", "exponential_decay", "mixed"],
                    help="Filter type used to generate synthetic data")
parser.add_argument("--filter_changing_speed", type=float, default=0.0,
                    help="Speed at which synthetic filter parameters change")
parser.add_argument("--filter_size", type=int,
                    help="Size of impulse response of synthetic filter")

# Auto-tuning flag
parser.add_argument("--auto_tune", action="store_true",
                    help="Grid-search to pick best NLMS/RLS parameters")

# NLMS hyper-parameters
parser.add_argument("--mu", type=float, default=0.8,
                    help="NLMS step-size μ (0<μ≤2)")
parser.add_argument("--eps", type=float, default=1e-6,
                    help="NLMS regulariser ε")

# RLS hyper-parameters
parser.add_argument("--lam", type=float, default=0.999,
                    help="RLS forgetting factor λ (≈1)")
parser.add_argument("--delta", type=float, default=10.0,
                    help="RLS δ for initial P(0)=I/δ")

# Parse known args + collect any extra flags as floats/strings
args, unknown = parser.parse_known_args()
extra_args = {}
for arg in unknown:
    if arg.startswith("--"):
        kv = arg.lstrip("--").split("=", 1)
        if len(kv) == 2:
            key, val = kv
            try:
                extra_args[key] = float(val)
            except ValueError:
                extra_args[key] = val
        else:
            extra_args[kv[0]] = True

# --- Validate synthetic prerequisites ---
if args.data == "synthetic":
    if args.noise_power is None or args.filter_type is None or args.filter_size is None:
        parser.error(
            "When using synthetic data, you must provide --noise_power, --filter_type and --filter_size."
        )
        sys.exit(1)

# --- Load or generate signals ------------------------------------------
mode   = args.data         # "real" or "synthetic"
method = args.method       # e.g. "nlms" or "rls"

if mode == "real":
    res = load_real_data(NOISY_SIGNAL_PATH, NOISE_SIGNAL_PATH)
    if res is None:
        sys.exit(1)
    noisy_signal, noise = res
    K_true = BEST_REAL_K
    Ks_to_try = [BEST_REAL_K] if not args.K_sweep else [50, 100, 200, 500, 1000]
    try:
        true_signal = get_baseline_signal(noisy_signal, noise)
    except NotImplementedError:
        print("Warning: Baseline signal extraction not implemented.")
        true_signal = None

else:  # synthetic
    res = generate_synthetic_data(
        NUM_SAMPLES,
        LOW,
        HIGH,
        SWITCHING_INTERVAL,
        args.filter_size,
        args.filter_type,
        args.filter_changing_speed,
        args.noise_power,
        args.noise_power_change,
        args.noise_distribution_change,
    )
    if res is None:
        sys.exit(1)
    noisy_signal, true_signal, noise, _ = res
    K_true = args.filter_size
    Ks_to_try = [K_true] if not args.K_sweep else list(range(1, K_true + 5))

# Aliases for tuning
D, X, S = noisy_signal, noise, true_signal

# --- AUTO-TUNE for NLMS/RLS (synthetic only) ----------------------------
if args.auto_tune and mode == "synthetic" and method in ("nlms", "rls"):
    if method == "nlms":
        mu_opt, K_opt = tuning.tune_nlms(
            D, X, S,
            mu_list=(0.4, 0.6, 0.8, 1.0),
            L_list=(32, 48, 64),
            eps=args.eps,
            val_len=1000
        )
        args.mu = mu_opt
        args.filter_size = K_opt
        print(f"[auto_tune] NLMS → μ={mu_opt}, K={K_opt}")
    else:  # method == "rls"
        lam_opt, delta_opt, K_opt = tuning.tune_rls(
            D, X, S,
            lam_list=(0.997, 0.998, 0.999),
            delta_list=(1, 5, 10),
            L_list=(32, 64),
            val_len=1000
        )
        args.lam = lam_opt
        args.delta = delta_opt
        args.filter_size = K_opt
        print(f"[auto_tune] RLS  → λ={lam_opt}, δ={delta_opt}, K={K_opt}")

# --- RUN SELECTED FILTERING METHOD -------------------------------------
if method not in methods:
    print(f"Invalid method '{method}'. Choose one of {list(methods.keys())}.")
    sys.exit(1)

filter_function = methods[method]
print(f"Running '{method}' on '{mode}' data...")

if filter_function is not None:
    for K in (Ks_to_try if method != "all" else [K_true]):
        print(f"Trying filter size {K}...")
        filtered_signal = filter_function(noisy_signal, noise, K, extra_args)
        if filtered_signal is None:
            sys.exit(1)

        to_plot = [
            ("Noisy signal", noisy_signal),
            ("Filtered signal", filtered_signal),
        ]
        if true_signal is not None:
            to_plot.append(("True signal", true_signal))
            print("MSE before:", compute_mse(true_signal, noisy_signal))
            print("MSE after: ", compute_mse(true_signal, filtered_signal))

        plot_signals(to_plot)

else:
    # Compare all methods
    results = []
    for name, func in methods.items():
        if func is None: continue
        sig = func(noisy_signal, noise, K_true, extra_args)
        if sig is None: sys.exit(1)
        results.append((name, sig))

    to_plot = [("Noisy signal", noisy_signal)] + results
    if true_signal is not None:
        plot_mses(true_signal, results)
        to_plot.append(("True signal", true_signal))

    plot_signals(to_plot)
