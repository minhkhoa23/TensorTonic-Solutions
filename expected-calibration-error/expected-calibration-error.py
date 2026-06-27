import numpy as np
def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    ece = 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_pred >= left) & (y_pred <= right)
        else:
            mask = (y_pred >= left) & (y_pred < right)

        if np.sum(mask) == 0:
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_pred[mask])

        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)