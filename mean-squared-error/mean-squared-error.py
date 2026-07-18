import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n = y_pred.shape[0]
    sum = np.sum((y_pred - y_true) ** 2)
    result = (1 / n) * sum
    return float(result)
