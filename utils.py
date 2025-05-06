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
        SNR = 10*log_10(P_signal/P_noise) and P_noise = sigma^2
    """
    power_signal = compute_power_signal(signal)
    power_noise = power_signal/(10**(snr/10))
    return power_noise


def filter_exponential_decay(t, tao = 5):
    """
        generate an exponential decay filter
    """
    h = np.exp(-t/tao)
    return h

def filter_dirac(t, alpha = 0.5):
    """
        generate filter h[n] = δ[n] + α δ[n‑1]+α^2 δ[n‑2] + ...
    """
    return [alpha**n for n in t] 

def generate_filter(type, size_filter):
    t = np.arange(size_filter)
    if type=='dirac':
        return filter_dirac(t)
    elif type == 'exponential':
        return filter_exponential_decay(t)
    else:
        raise ValueError
    

def generate_wgn(mean, std, size):
    return np.random.normal(loc=mean, scale=std, size=size)

def generate_noise(type, power_noise, size):
    
    if type == 'gaussian':
        return generate_wgn(mean=0, std = np.sqrt(power_noise), size=size)
    elif type == 'uniform':
        pass 
    else:
        return ValueError
    
    
def plot(noisy_signal, signal, h, type_noise, type_signal, type_filter, snr):
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
    
    if snr != None:
        if type_signal == 'sinus':
            sns.lineplot(noisy_signal, ax=axes[1])    
        else:    
            sns.lineplot(noisy_signal, ax=axes[1], drawstyle='steps-post')
        axes[1].set_xlim(0, 100)
        axes[1].set_title(f'noisy signal, {type_noise} noise, SNR = {snr}dB')
        axes[1].set_xlabel('num_samples')

        sns.pointplot(h, ax=axes[2])
        axes[2].set_title(f"{type_filter} filter")
        axes[2].set_xlabel('num_samples')

    plt.show()