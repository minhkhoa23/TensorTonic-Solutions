def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    # Write code here
    result = []
    window_size = len(weights)
    weight_sum = sum(weights)

    for i in range(len(values) - window_size + 1):
        weighted_sum = 0.0

        for j in range(window_size):
            weighted_sum += values[i + j] * weights[j]

        result.append(weighted_sum / weight_sum)
    return result