import numpy as np
import librosa
from utils import *
from scipy.signal import lfilter

"""Returns noisy signal and noise signal as a 2-tuple of numpy arrays
Prints error message and returns None when it fails"""


def load_real_data(noisy_signal_path, noise_path):
    noisy_signal, srns = librosa.load(noisy_signal_path) #srns = sampling rate noisy signal
    noise_signal, srn = librosa.load(noise_path) # srn = sampling rate noise 
    return noisy_signal, noise_signal, srns, srn         


"""Generate synthetic signal. Returns true signal, noisy signal and noise signal as a 3-tuple of numpy arrays
Prints error message and returns None when it fails"""

def generate_signal(num_samples, low, high, type_signal):
    t = np.linspace(low, high, num_samples)
    if type_signal == 'sinus':
        signal = np.sin(t)
    elif type_signal == 'binary':
        signal = np.zeros(shape=num_samples)
        signal[::2] = 1
    else: 
        raise ValueError
    return signal

def generate_synthetic_data(num_samples, low, high, size_filter, snr=None, type_signal='sinus', type_filter='dirac', type_noise='gaussian'):
    #generate signal
    signal = generate_signal(num_samples=num_samples, low=low, high=high, type_signal=type_signal)
    
    #if snr is None, no noise added, hence no need to return the filter, noisy signal and the noise
    if snr == None:
        return None, signal, None, None
    
    power_noise = compute_power_noise(signal, snr)
    noise = generate_noise(type=type_noise, power_noise=power_noise, size=signal.shape[0])
    h = generate_filter(type=type_filter, size_filter=size_filter)
    #noise_filtered = np.convolve(noise, h, mode='same')
    noise_filtered = lfilter(h, 1, noise)
    noisy_signal = signal + noise_filtered
    return noisy_signal, signal, noise, h

if __name__ == "__main__":
    #begining and end value of the signal we want to generate
    low, high = 0, 1000*np.pi
    #size of the signal
    num_samples = 10_000
    #size of the filter
    size_filter = 10
    #amount of noise we want to have
    snr = 10
    
    type_signal = 'sinus' #can be sinus or binary 
    type_filter='exponential' #can be dirac or exponential
    type_noise='gaussian'
    
    noisy_signal, signal, noise, h = generate_synthetic_data(num_samples=num_samples, low=low, high=high, size_filter=size_filter, snr=snr, type_signal=type_signal, type_filter=type_filter, type_noise=type_noise)
    plot(noisy_signal, signal, h, type_noise, type_signal, type_filter, snr)
