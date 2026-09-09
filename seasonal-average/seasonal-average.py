def seasonal_average(series: list, period: int) -> list:
    """
    Returns the average for each position in the seasonal cycle.
    """
    # Write code here
    result = []

    for i in range(period):
        values = series[i::period]
        average = sum(values) / len(values)
        result.append(average)

    return result