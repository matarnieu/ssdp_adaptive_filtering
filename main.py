"""
Adaptive Signal Filtering Script
================================

This script applies various **adaptive filtering algorithms** to either real-world audio data or synthetically generated signals.
Its main purpose is to evaluate how well different adaptive filters remove noise while preserving the original signal.

The script uses model the following noise model:

D[n] = S[n] + h[n]*X[n]

where:
- D[n] is the noisy signal (observed)
- S[n] is the true signal (to be estimated)
- h[n] is the noise filter impulse response (to be estimated)
- X[n] is the noise signal (observed)

Supported Methods:
------------------
- Generalized Wiener Filter variants: `gwf_fc`, `gwf_swc`, `gwf_ema`
- Least Mean Squares (LMS): `lms`
- Normalized LMS: `nlms`
- Variable Step-size LMS: `vslms`
- Recursive Least Squares (RLS): `rls`
- Kalman Filter: `kalman`

Usage (Command Line):
---------------------

Basic usage:
    python script.py {real|synthetic} {method} [options]

Examples:
    python script.py real nlms
    python script.py synthetic lms --noise_power=0.5 --filter_type=moving_average --filter_size=100

Required Arguments:
-------------------
- `data`: Type of data to use: `real` or `synthetic`
- `method`: One of the adaptive filtering methods listed above

Real Data Mode:
---------------
- Loads a noisy signal and a reference noise signal from audio files
- Applies the specified adaptive filtering method
- Saves the filtered (cleaned) signal to a WAV file
- Computes MSE (if baseline is available) and optionally plots signals

Synthetic Data Mode:
--------------------
- Requires these additional arguments:
    --noise_power <float>
    --filter_type {"moving_average", "exponential_decay", "mixed"}
    --filter_size <int>

- Optional enhancements to synthetic signal:
    --noise_type {"wgn", "wgn_power_change", "ar", "ar_correlation_change", "mixed"}: Defines the type of synthetic noise signal and any time-varying properties.
    --filter_changing_speed <float>: Controls smooth evolution of filter characteristics

- Synthetic data includes a known true signal for MSE evaluation
- Clipping is applied to provide a clean and visualizable output

Optional Arguments:
-------------------
- `--dont_show_plot`: Disables GUI plots (useful when calling from scripts)
- `--plot_filename`: Save plot to file instead of showing it interactively
- `--print_filter_distances`: Only for synthetic mode + methods that return filter history; prints the L2 distance between the true and estimated filter coefficients at each step
- You can also pass **method-specific hyperparameters** via `--param=value`

Outputs:
--------
- A filtered version of the noisy input signal
- MSE scores between noisy/filtered signals and true signal (when available)
- Optional plots of noisy, filtered, and true signals
- For real data: saves cleaned signal as `clean_<method>_K<filter_size>.wav`

Dependencies:
-------------
- Python 3.x
- NumPy
- Matplotlib (for plotting)
- SoundFile (for reading/writing WAV files)
- Custom modules: `data_loader.py`, `analyzer.py`, and filtering algorithms in `methods/`

Project Structure:
------------------
- data/
  ├── bassLineTalkingNoise.mp3
  └── talkingNoise.mp3
- methods/
  ├── generalized_wiener_filter.py
  ├── lms.py
  ├── vslms.py
  ├── nlms.py
  ├── rls.py
  └── kalman.py
- data_loader.py
- analyzer.py
- <this_script>.py

"""

import sys
import random
import argparse
import numpy as np
import json
import soundfile as sf
import os

from data_loader import load_real_data, generate_synthetic_data
from analyzer import plot_signals, compute_mse

# Import adaptive filtering algorithms
from methods.generalized_wiener_filter import (
    filter_signal_gwf_fc,
    filter_signal_gwf_swc,
    filter_signal_gwf_ema,
)
from methods.lms import filter_signal_lms
from methods.vslms import filter_signal_vslms
from methods.baseline import get_baseline_signal
from methods.nlms import filter_signal_nlms
from methods.rls import filter_signal_rls
from methods.kalman import filter_signal_kalman

# --- HYPERPARAMETERS ---
# Seed for reproducibility of randomness
DEFAULT_SEED = 41
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)

# Paths to real data
DATA_PATH = "data"
NOISY_SIGNAL_PATH = os.path.join(DATA_PATH, "bassLineTalkingNoise.mp3")
NOISE_SIGNAL_PATH = os.path.join(DATA_PATH, "talkingNoise.mp3")

# Best filter size K for real data
BEST_REAL_K = 50
# Signal Type: "sinus" or "musical"
SIGNAL_TYPE = "sinus"
# Number of samples in synthetic signal
NUM_SAMPLES = 10000
# Interval in which the filter parameters abruptly change in synthetic signal
SWITCHING_INTERVAL = 1111  # 1300
# Define synthetic sine
LOW, HIGH = 0, 1000 * np.pi
# After which value the filtered synthetic signal should be clipped
SYNTH_CLIP = 50.0

# All adaptive filtering algorithms used, as well as the cmd line arguments used to select them
methods = {
    "gwf_fc": filter_signal_gwf_fc,
    "gwf_swc": filter_signal_gwf_swc,
    "gwf_ema": filter_signal_gwf_ema,
    "lms": filter_signal_lms,
    "vslms": filter_signal_vslms,
    "nlms": filter_signal_nlms,
    "rls": filter_signal_rls,
    "kalman": filter_signal_kalman,
}

# --- GLOBAL VARIABLES ---

# Baseline signal
true_signal = None
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
    "--print_filter_distances",
    dest="print_filter_distances",
    action="store_true",
    default=False,
    help="Print complete history of filter distances.",
)

parser.add_argument(
    "--plot_filename",
    dest="plot_filename",
    default=None,
    help="Filename of stored plot",
)

# Optional arguments (only needed for synthetic data)
parser.add_argument(
    "--noise_type",
    type=str,
    default="wgn",
    choices=["wgn", "wgn_power_change", "ar", "ar_correlation_change", "mixed"],
)
parser.add_argument(
    "--noise_power",
    type=float,
    help="Power of X, required for synthetic data",
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
# Convert unknown args into a dictionary that will be passed to the filter function
# Allows for method specific parameters to be passed via command line
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

# Validate required arguments for synthetic data
mode = args.data  # "real" or "synthetic"
method = args.method  # e.g. "nlms" or "rls"

if args.data == "synthetic":
    if (
        args.noise_power is None
        or args.noise_type is None
        or args.filter_type is None
        or args.filter_size is None
    ):
        parser.error(
            "When using synthetic data, you must provide --noise_power , --noise_type, --filter_type and --filter_size."
        )
        sys.exit(1)

if mode == "real":  # real data
    # Load data
    res = load_real_data(NOISY_SIGNAL_PATH, NOISE_SIGNAL_PATH)
    if res is None:
        sys.exit(1)
    noisy_signal, noise, sr = res
    # Set filter size that should be tried out
    K = BEST_REAL_K
    # If available, get baseline signal for comparison
    try:
        true_signal = get_baseline_signal(noisy_signal, noise)
    except NotImplementedError:
        print("Warning: Baseline signal extraction not implemented.")
        true_signal = None
else:  # synthetic data
    res = generate_synthetic_data(
        NUM_SAMPLES,
        LOW,
        HIGH,
        SWITCHING_INTERVAL,
        args.filter_size,
        args.filter_type,
        args.filter_changing_speed,
        args.noise_power,
        args.noise_type,
        type_signal=SIGNAL_TYPE,
    )
    if res is None:
        sys.exit(1)
    noisy_signal, true_signal, noise, true_filter_history = res
    # Set filter size that is known, since it was used to generate the synthetic data
    K = args.filter_size

# --- RUN SELECTED FILTERING METHOD ---

if method not in methods:
    print(f"Invalid method '{method}'. Choose one of {list(methods.keys())}.")
    sys.exit(1)
else:
    # Select adaptive filtering method
    filter_function = methods[args.method]

print(f"Running '{args.method}' method on '{args.data}' data...")

# Perform adaptive filtering
filtered_signal = filter_function(noisy_signal, noise, K, extra_args)
# Check for errors in filtering
if filtered_signal is None:
    sys.exit(1)
# If the method returns a tuple, it means it also returns filter history
elif isinstance(filtered_signal, tuple):
    filtered_signal, filter_history = filtered_signal
    if mode == "synthetic" and args.print_filter_distances:
        filter_distances = np.linalg.norm(
            np.array(filter_history) - np.array(true_filter_history),
            axis=1,
        )
        filter_distances = np.clip(filter_distances, -10, 10)
        # Print filter distances if requested
        print("Filter distances:", json.dumps(filter_distances.tolist()))
# For synthetic data, clip result to avoid extreme values and make it easier to visualize
if not mode == "real":
    # If method returns some extreme values, delete them
    filtered_signal = np.clip(filtered_signal, -SYNTH_CLIP, SYNTH_CLIP)

# If real data, save the filtered signal to a file
if mode == "real":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join it with the data filename
    out_path = os.path.join(script_dir, DATA_PATH, f"clean_{method}_K{K}.wav")
    sf.write(out_path, filtered_signal, 44100, subtype="PCM_24")
    print(f"[info] cleaned signal written → {out_path}")

# Visualize results
signals_to_plot = [
    ("Noisy signal", noisy_signal),
    ("Filtered signal", filtered_signal),
]

# Also plot true signal, if available
if true_signal is not None:
    signals_to_plot.append(("True signal", true_signal))
    # Compute MSE
    mse_before_filtering = compute_mse(true_signal, noisy_signal)
    print(f"MSE before filtering: {mse_before_filtering}")
    mse = compute_mse(true_signal, filtered_signal)
    print(f"MSE: {mse}")

plot_signals(
    signals_to_plot,
    filename=args.plot_filename,
    show=not args.dont_show_plot,
)
