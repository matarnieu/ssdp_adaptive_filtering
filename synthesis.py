import numpy as np

# FILTER


def _filter_exponential_decay(size_filter, timesteps, changing_speed=0):
    """
    Generates a time-varying exponential decay filter.

    Each row in the returned matrix corresponds to a time step, and the filter decays more
    slowly or quickly depending on the changing speed.

    Args:
        size_filter (int): The number of taps (length) of the filter.
        timesteps (int): The number of time steps for which the filter is generated.
        changing_speed (float, optional): Controls how rapidly the decay profile changes
            over time. Defaults to 0 (no change). Recommended values are between 0 and 1.

    Returns:
        np.ndarray: A matrix of shape (timesteps, size_filter) representing the filter over time.
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
    """
    Generates a time-varying moving average filter.

    Each row in the returned matrix corresponds to a time step, and the filter decays more
    slowly or quickly depending on the changing speed.

    Args:
        size_filter (int): The number of taps (length) of the filter.
        timesteps (int): The number of time steps for which the filter is generated.
        changing_speed (float, optional): Controls how rapidly the decay profile changes
            over time. Defaults to 0 (no change). Recommended values are between 0 and 1.

    Returns:
        np.ndarray: A matrix of shape (timesteps, size_filter) representing the filter over time.
    """
    H = np.zeros((timesteps, size_filter))
    for time in range(timesteps):
        alpha = 1 + 0.5 * np.sin(10 * time * changing_speed / timesteps)
        weights = alpha ** np.arange(size_filter)[::-1]  # more weight on current sample
        H[time] = weights / np.sum(weights)

    return H


def generate_mixed_filter(
    size_filter, total_timesteps, switching_interval, changing_speed=0
):
    """
    Creates a filter matrix that alternates between exponential decay and moving average filters.

    Args:
        size_filter (int): The number of taps in the filter.
        total_timesteps (int): Total number of time steps.
        switching_interval (int): Number of time steps between switching the filter type.
        changing_speed (float, optional): Speed at which the filter characteristics evolve over time.
                Defaults to 0 (no change). Recommended values are between 0 and 1.

    Returns:
        np.ndarray: A matrix of shape (total_timesteps, size_filter) containing the mixed filter.
    """
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


def _generate_wgn(mean, std, size):
    return np.random.normal(loc=mean, scale=std, size=size)


def _generate_time_varying_wgn_with_non_smooth_std(mean, std, size, switching_interval):
    return np.random.normal(
        loc=mean, scale=_piecewise_std(size, std, switching_interval), size=size
    )


def _generate_ar_noise(mean, std, size, order, coeffs=None):
    if coeffs is None:
        # Generate stable AR coefficients (roots inside unit circle)
        coeffs = np.random.uniform(-0.6, 0.6, order)
        while np.any(np.abs(np.roots(np.r_[1, -coeffs])) >= 1):
            coeffs = np.random.uniform(-0.6, 0.6, order)

    w = np.random.normal(0, 1.0, size)  # unit variance
    x = np.zeros(size)
    for n in range(order, size):
        x[n] = np.dot(coeffs, x[n - order : n][::-1]) + w[n]

    # Rescale to desired mean and std
    x = (x - np.mean(x)) / np.std(x)  # Normalize to zero mean, unit std
    x = x * std + mean
    return x


def _generate_highly_correlated_ar_noise(mean, std, size, order=10):
    # Use strong, slowly decaying coefficients
    coeffs = 0.95 ** np.arange(1, order + 1)
    coeffs = coeffs / np.sum(coeffs) * 0.9  # normalize and scale to stay stable

    # Now generate AR noise with those coefficients
    return _generate_ar_noise(mean, std, size, order=order, coeffs=coeffs)


def _generate_mixed_noise(mean, std, total_timesteps, switching_interval):
    """
    Generate noise that switches between two extremes:
    - White Gaussian Noise (WGN): no temporal correlation
    - AR(10) Noise: strong temporal correlation

    The signal switches every `switching_interval` samples.
    """
    time = np.arange(total_timesteps)
    segment_type = (time // switching_interval) % 2  # 0 or 1

    # Generate WGN (uncorrelated)
    wgn = _generate_wgn(mean, std, total_timesteps)

    # Generate AR(10) with slowly decaying coefficients (high correlation)
    ar_order = 10
    ar_noise = _generate_highly_correlated_ar_noise(
        mean, std, total_timesteps, order=ar_order
    )

    # Merge the two signals
    mixed_noise = np.zeros(total_timesteps)
    mixed_noise[segment_type == 0] = wgn[segment_type == 0]
    mixed_noise[segment_type == 1] = ar_noise[segment_type == 1]

    return mixed_noise


def _generate_morphing_ar_noise(
    mean, std, size, order=3, coeffs_start=None, coeffs_end=None
):
    """
    Generate AR noise with linearly evolving coefficients from coeffs_start to coeffs_end.

    Args:
        mean (float): Desired mean of the output signal.
        std (float): Desired std of the output signal.
        size (int): Number of samples.
        order (int): AR order.
        coeffs_start (array): Starting AR coefficients.
        coeffs_end (array): Ending AR coefficients.

    Returns:
        np.ndarray: Time-varying AR noise signal.
    """
    if coeffs_start is None:
        coeffs_start = np.array([0.6, -0.5, 0.2])[:order]
    if coeffs_end is None:
        coeffs_end = np.array([-0.8, 0.4, 0.3])[:order]

    assert (
        len(coeffs_start) == order and len(coeffs_end) == order
    ), "Coefficient arrays must match the AR order"

    x = np.zeros(size)
    w = np.random.normal(0, 1, size)

    # Linearly interpolate coefficients over time
    coeffs_t = np.linspace(coeffs_start, coeffs_end, num=size)

    # Generate AR process with time-varying coefficients
    for n in range(order, size):
        current_coeffs = coeffs_t[n]
        x[n] = np.dot(current_coeffs, x[n - order : n][::-1]) + w[n]

    # Normalize to target mean and std
    x = (x - np.mean(x)) / np.std(x)
    x = x * std + mean
    return x


def generate_noise(size, noise_power, noise_type, switching_interval):
    std = np.sqrt(noise_power)
    mean = 0.0
    if noise_type == "wgn":
        return _generate_wgn(mean, std, size)
    elif noise_type == "ar":
        return _generate_highly_correlated_ar_noise(mean, std, size, order=10)
    elif noise_type == "mixed":
        return _generate_mixed_noise(mean, std, size, switching_interval)
    elif noise_type == "ar_correlation_change":
        return _generate_morphing_ar_noise(
            mean,
            std,
            size,
            order=10,
            coeffs_start=[0.0] * 10,  # No memory (essentially white noise)
            coeffs_end=[
                0.9,
                -0.75,
                0.6,
                -0.4,
                0.5,
                -0.35,
                0.2,
                -0.15,
                0.1,
                -0.05,
            ],  # Very strong memory, near instability
        )
    elif noise_type == "wgn_power_change":
        return _generate_time_varying_wgn_with_non_smooth_std(
            mean, std, size, switching_interval
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
