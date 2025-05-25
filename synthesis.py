import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# FILTER


def _filter_exponential_decay(size_filter, timesteps, changing_speed=0):
    """
    generate an exponential decay filter
    """
    t = np.arange(size_filter)
    H = np.zeros((timesteps, t.shape[0]))
    alpha = timesteps // 5
    for time in range(1, timesteps + 1):
        if changing_speed == 0:
            adjusted_time = 1
        else:
            adjusted_time = max(time * changing_speed, 1e-6)
        decay = np.exp(-alpha * t / adjusted_time)
        H[time - 1] = decay / np.sum(decay)  # keep the energy
    return H


def _filter_moving_average(size_filter, timesteps, changing_speed=0):
    H = np.zeros((timesteps, size_filter))
    for time in range(timesteps):
        alpha = 1 + 0.5 * np.sin(10 * time * changing_speed / timesteps)
        weights = alpha ** np.arange(size_filter)[::-1]  # more weight on current sample
        H[time] = weights / np.sum(weights)

    return H


def generate_mixed_filter(
    size_filter, total_timesteps, switching_interval, changing_speed=0
):
    # Generate both full filter sequences
    H1 = _filter_exponential_decay(size_filter, total_timesteps, changing_speed)
    H2 = _filter_moving_average(size_filter, total_timesteps, changing_speed)

    # Create switch mask
    time = np.arange(total_timesteps)
    is_alt_filter = (time // switching_interval) % 2 == 1

    # Initialize final filter matrix
    H = np.zeros_like(H1)
    H[is_alt_filter] = H2[is_alt_filter]
    H[~is_alt_filter] = H1[~is_alt_filter]

    return H


def generate_filter(type, filter_size, timesteps, switching_interval, changing_speed=0):
    if type == "moving_average":
        return _filter_moving_average(filter_size, timesteps, changing_speed)
    elif type == "exponential_decay":
        return _filter_exponential_decay(filter_size, timesteps, changing_speed)
    elif type == "mixed":
        return generate_mixed_filter(
            filter_size, timesteps, switching_interval, changing_speed
        )
    else:
        raise ValueError


# NOISE


def _piecewise_std(size, std, switching_interval):
    t = np.arange(size)
    interval_idx = (t // switching_interval) % 3  # Cycle through 0, 1, 2

    cste = np.ones_like(t, dtype=float)
    cste[interval_idx == 1] = 2.0
    cste[interval_idx == 2] = 0.5

    return cste * std


def generate_impulse_noise(mean, std, size, impulse_prob=0.01, impulse_scale=10):
    noise = np.random.normal(mean, std, size)
    impulses = np.random.rand(size) < impulse_prob
    noise[impulses] = mean + impulse_scale * std * np.random.choice(
        [-1, 1], size=np.sum(impulses)
    )
    return noise


def generate_pink_noise(mean, std, n_samples):
    # Generate pink (1/f) noise using the Voss-McCartney algorithm.
    n_rows = int(np.ceil(np.log2(n_samples)))
    n_cols = n_samples
    array = np.random.randn(n_rows, n_cols)
    array = np.cumsum(array, axis=0)
    pink = np.sum(array / (2 ** np.arange(n_rows)[:, None]), axis=0)
    pink = pink - np.mean(pink)
    pink = pink / np.std(pink)
    pink = pink * std + mean
    return pink


import numpy as np
from scipy.integrate import solve_ivp


def _generate_chaotic_noise(
    mean, std, size, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0, x0=[1.0, 1.0, 1.0]
):
    """
    Generate chaotic noise using the Lorenz attractor.
    Returns a 1D projection of the 3D trajectory (e.g., the x-component).
    The result is scaled to the desired mean and std.
    """

    def lorenz(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_span = (0, dt * size)
    t_eval = np.linspace(t_span[0], t_span[1], size)

    sol = solve_ivp(lorenz, t_span, x0, t_eval=t_eval, method="RK45")
    x = sol.y[0]  # Take the x-component (or use y[1], y[2] for variation)

    # Normalize to desired mean and std
    x = (x - np.mean(x)) / np.std(x)
    x = x * std + mean
    return x


def _generate_uniform_noise(mean, std, total_timesteps):
    half_range = std * np.sqrt(3)
    uniform_noise = np.random.uniform(
        low=mean - half_range, high=mean + half_range, size=total_timesteps
    )
    return uniform_noise


def _generate_wgn(mean, std, size):
    return np.random.normal(loc=mean, scale=std, size=size)


def _generate_time_varying_wgn_with_non_smooth_std(mean, std, size, switching_interval):
    return np.random.normal(
        loc=mean, scale=_piecewise_std(size, std, switching_interval), size=size
    )


def _generate_mixed_noise(mean, std, total_timesteps, switching_interval):
    # Generate Gaussian white noise

    gaussian_noise = np.random.normal(loc=mean, scale=std, size=total_timesteps)
    # DIFFERENT CHOICES FOR SECOND NOISE
    uniform_noise = _generate_uniform_noise(mean, std, total_timesteps)
    chaotic_noise = _generate_chaotic_noise(mean, std, total_timesteps)
    pink_noise = generate_pink_noise(mean, std, total_timesteps)
    impulse_noise = generate_impulse_noise(mean, std, total_timesteps)
    second_noise = chaotic_noise  # Choose one of the above

    # Create switching mask: True for chaotic intervals [1000, 2000), [3000, 4000), ...
    time = np.arange(total_timesteps)
    is_chaotic = (time // switching_interval) % 2 == 1

    # Combine based on mask
    mixed_noise = np.where(is_chaotic, second_noise, gaussian_noise)

    return mixed_noise


def generate_noise(
    size, noise_power, noise_power_change, noise_distribution_change, switching_interval
):
    std = np.sqrt(noise_power)
    mean = 0.0
    if noise_distribution_change:
        return _generate_mixed_noise(mean, std, size, switching_interval)
    elif noise_power_change:
        return _generate_time_varying_wgn_with_non_smooth_std(
            mean, std, size, switching_interval
        )
    else:
        return _generate_wgn(mean, std, size)


def plot(
    noisy_signal,
    signal,
    h,
    type_signal,
    type_filter,
    snr,
    window_start=0,
    window_size=500,
):
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
    axes[1].plot(x, noisy_signal[window_start:end], label="Noisy", color="orange")
    axes[1].set_title(f"Noisy signal (SNR={snr} dB)")
    axes[1].set_xlabel("Sample index")
    axes[1].legend()

    # Plot filter
    if h.ndim == 2:  # time-varying filters: shape (num_samples, filter_size)
        # Plot only a few filter snapshots
        step = num_sample // 10
        for idx in range(0, num_sample, step):
            # t = window_start + i
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
