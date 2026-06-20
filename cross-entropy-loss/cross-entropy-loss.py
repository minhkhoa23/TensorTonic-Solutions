import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rows = np.arange(len(y_true))
    true = y_pred[rows, y_true]
    loss = -np.log(true)

    return float(np.mean(loss))