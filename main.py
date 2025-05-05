import sys

# Import adaptive filtering algorithms
from methods.simple import filter_signal_simple
from methods.sgd import filter_signal_sgd

from data_loader import load_real_data, generate_synthetic_data
from visualizer import visualize_signals

# Seed for reproducibility of randomness
DEFAULT_SEED = 42
# Paths to real data
DATA_PATH = "data/"
NOISY_SIGNAL_PATH = DATA_PATH + "bassLineTalkingNoise.mp3"
NOISE_SIGNAL_PATH = DATA_PATH + "talkingNoise.mp3"

# All adaptive filtering algorithms used, as well as the cmd line arguments used to select them
methods = {
    "simple": filter_signal_simple,
    "sgd": filter_signal_sgd,
}

# Noisy signal, as well as noise (before it goes through the filter)
noisy_signal = None
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
    res = (
        load_real_data(NOISY_SIGNAL_PATH, NOISE_SIGNAL_PATH)
        if data_arg == "real"
        else generate_synthetic_data(DEFAULT_SEED)
    )
    # Check for error
    if res is None:
        sys.exit(1)
    else:
        noisy_signal, noise = res


if method_arg not in methods.keys():
    print(f"Invalid method. Choose one of {methods.keys()}.")
    sys.exit(1)
else:
    # Select adaptive filtering method
    filter_signal = methods[method_arg]

# Perform adaptive filtering
# TODO: Measure running time
filtered_signal = filter_signal(noisy_signal, noise)
if filtered_signal is None:
    sys.exit(1)

# Visualize results
visualize_signals([noisy_signal, filtered_signal], ["Noisy Signal", "Filtered Signal"])

# TODO: Compute and measure stuff like error, etc...
