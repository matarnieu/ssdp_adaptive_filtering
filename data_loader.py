"""Returns noisy signal and noise signal as a 2-tuple of numpy arrays
Prints error message and returns None when it fails"""


def load_real_data(noisy_signal_path, noise_path):
    raise NotImplementedError


"""Generate synthetic signal. Randomness is based on seed for reproducibility (same seed --> same output)
Returns true signal, noisy signal and noise signal as a 3-tuple of numpy arrays
Prints error message and returns None when it fails"""


def generate_synthetic_data(seed):
    raise NotImplementedError
