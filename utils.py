import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

def piecewise_std(t, std):
    t = np.arange(t)
    cste = np.ones_like(t) * 1.0
    cste[t > 0.3 * t.max()] = 2.0
    cste[t > 0.6 * t.max()] = 0.5
    return cste * std

def compute_power_signal(signal):
    """
        Compute the average mean power of a signal
    """
    # use abs so we can handle real and complex signal
    power = np.mean(np.abs(signal)**2)
    return power

def compute_power_noise(signal, snr):
    """
        SNR = 10*log_10(P_signal/P_noise) and P_noise = sigma^2 for Gaussian noise
    """
    power_signal = compute_power_signal(signal)
    power_noise = power_signal/(10**(snr/10))
    return power_noise


def filter_exponential_decay(size_filter, timestep):
    """
        generate an exponential decay filter
    """
    t = np.arange(size_filter)
    H = np.zeros((timestep, t.shape[0]))
    alpha = timestep//5
    for time in range(1, timestep+1):
        decay = np.exp(-alpha*t/time)
        H[time-1] = decay/np.sum(decay) #keep the energy
    return H

def filter_moving_average(size_filter, timestep):
    H = np.zeros((timestep, size_filter))
    for t in range(timestep):
        alpha = 1 + 0.5 * np.sin(10 * t / timestep) 
        weights = alpha ** np.arange(size_filter)[::-1]  # more weight on current sample
        H[t] = weights / np.sum(weights)

    return H

def generate_time_varying_filter(type, size_filter, timestep):
    if type=='moving_average':
        return filter_moving_average(size_filter, timestep)
    elif type == 'exponential_decay':
        return filter_exponential_decay(size_filter, timestep)
    else:
        raise ValueError
    

def generate_time_varying_wgn_with_non_smooth_std(mean, std, size, timestep):
    return np.random.normal(loc=mean, scale=piecewise_std(timestep, std), size=size)

def generate_wgn(mean, std, size):
    return np.random.normal(loc=mean, scale=std, size=size)

def generate_noise(power_noise, size, timestep=None):
    if timestep == None:
        return generate_wgn(mean=0, std = np.sqrt(power_noise), size=size)
    else:
        return generate_time_varying_wgn_with_non_smooth_std(mean=0, std = np.sqrt(power_noise), size=size, timestep=timestep)
    
    
def plot(noisy_signal, signal, h, type_signal, type_filter, snr, window_start=0, window_size=500):
    num_sample = signal.shape[0]
    if window_start > num_sample:
        window_start = 0
    if window_size > num_sample:
        window_size = num_sample
        window_start = 0
    # Select a window
    end = window_start + window_size
    if end > num_sample: 
        end = num_sample
        
    x = np.arange(window_start, end)

    _, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # Plot clean signal
    axes[0].plot(x, signal[window_start:end])
    axes[0].set_title(f"{type_signal} signal (window {window_start}-{end})")
    axes[0].set_xlabel("Sample index")
    axes[0].legend()

    # Plot noisy signal
    axes[1].plot(x, noisy_signal[window_start:end], label="Noisy", color='orange')
    axes[1].set_title(f"Noisy signal (SNR={snr} dB)")
    axes[1].set_xlabel("Sample index")
    axes[1].legend()

    # Plot filter
    if h.ndim == 2:  # time-varying filters: shape (num_samples, filter_size)
        # Plot only a few filter snapshots
        step = num_sample//10
        for idx in range(0, num_sample, step):
            #t = window_start + i
            if idx < h.shape[0]:
                axes[2].plot(h[idx], label=f"t={idx}")
        axes[2].set_title(f"{type_filter} filters (sampled over window)")
        axes[2].legend()
    else:
        axes[2].plot(h)
        axes[2].set_title(f"{type_filter} filter (static)")

    axes[2].set_xlabel("Filter tap index")
    plt.tight_layout()
    plt.show()