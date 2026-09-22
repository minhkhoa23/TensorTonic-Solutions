def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    """
    Returns the smoothed level at every time step.
    """
    # Write code here
    if not series:
        return []

    if len(series) == 1:
        return [series[0]]

    level = series[0]
    trend = series[1] - series[0]

    result = [level]

    for i in range(1, len(series)):
        previous_level = level

        level = (
            alpha * series[i] + (1 - alpha) * (previous_level + trend)
        )

        trend = (beta * (level - previous_level) + (1 - beta) * trend)

        result.append(level)

    return result