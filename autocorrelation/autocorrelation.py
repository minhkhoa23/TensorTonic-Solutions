def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    if not series:
        return []

    mean = sum(series) / len(series)

    denominator = sum(
        (x - mean) ** 2
        for x in series
    )

    # Chuỗi hằng số -> variance = 0
    if denominator == 0:
        return [1.0] + [0.0] * max_lag

    result = []

    for lag in range(max_lag + 1):
        numerator = 0

        for i in range(lag, len(series)):
            numerator += (
                (series[i] - mean)
                * (series[i - lag] - mean)
            )

        result.append(numerator / denominator)

    return result