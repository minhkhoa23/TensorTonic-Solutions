def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    # Write code here
    result = []
    max_lag = max(lags)

    for i in range(max_lag, len(series)):
        result.append([series[i - lag] for lag in lags])

    return result