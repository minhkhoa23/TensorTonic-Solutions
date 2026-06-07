import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    tp = 0
    fp = 0
    fn = 0

    for c in classes:
        tp += np.sum((y_true == c) & (y_pred == c))
        fp += np.sum((y_true != c) & (y_pred == c))
        fn += np.sum((y_true == c) & (y_pred != c))

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return 2 * tp / denom