import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have same length")

    if len(fpr) < 2:
        raise ValueError("Need at least 2 points")

    area = np.trapezoid(tpr, fpr)

    return float(area)