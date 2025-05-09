import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

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
    cste = timestep//5
    for time in range(1, timestep+1):
        decay = np.exp(-cste*t/time)
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
    t = np.arange(size_filter)
    if type=='moving_average':
        return filter_moving_average(size_filter, timestep)
    elif type == 'exponential_decay':
        return filter_exponential_decay(size_filter, timestep)
    else:
        raise ValueError
    

def generate_wgn(mean, std, size):
    return np.random.normal(loc=mean, scale=std, size=size)

def generate_noise(power_noise, size):
    return generate_wgn(mean=0, std = np.sqrt(power_noise), size=size)
    
    
def plot(noisy_signal, signal, h, type_signal, type_filter, snr):
    if snr == None:
        if type_signal == 'sinus':
            sns.lineplot(signal)    
        else:    
            sns.lineplot(signal, drawstyle='steps-post')
        plt.xlim(0, 100)
        title = f'original signal: {type_signal}'
        if type_signal == 'binary':
            title += " (set xlim = (0, 10) for visibility)"
        plt.title(title)
        plt.xlabel('num_samples')
        plt.show()
        return 
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    if type_signal == 'sinus':
        sns.lineplot(signal, ax=axes[0])    
    else:    
        sns.lineplot(signal, ax=axes[0], drawstyle='steps-post')
    axes[0].set_xlim(0, 100)
    title = f'original signal: ({type_signal})'
    axes[0].set_title(title)
    axes[0].set_xlabel('num_samples')
    
    if type_signal == 'sinus':
        sns.lineplot(noisy_signal, ax=axes[1])    
    else:    
        sns.lineplot(noisy_signal, ax=axes[1], drawstyle='steps-post')
    axes[1].set_xlim(0, 100)
    axes[1].set_title(f'noisy signal, Gaussian noise, SNR = {snr}dB')
    axes[1].set_xlabel('num_samples')

    sns.pointplot(h, ax=axes[2])
    axes[2].set_title(f"{type_filter} filter")
    axes[2].set_xlabel('num_samples')

    plt.show()
    
def plot_2(noisy_signal, signal, h, type_signal, type_filter, snr, window_start=0, window_size=500):
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
    axes[0].plot(x, signal[window_start:end], label="Clean")
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
