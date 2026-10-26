def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    # Write code here
    if not values:
        return []

    ema = [values[0]]

    for i in range(1, len(values)):
        current = alpha * values[i] + (1 - alpha) * ema[-1]
        ema.append(current)

    return ema