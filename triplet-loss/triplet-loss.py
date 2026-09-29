import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    anchor = np.atleast_2d(anchor)
    positive = np.atleast_2d(positive)
    negative = np.atleast_2d(negative)

    d_ap = np.sum((anchor - positive) ** 2, axis = 1)

    d_an = np.sum((anchor - negative) ** 2, axis = 1)

    loss = np.maximum(d_ap - d_an + margin, 0.0)

    return float(np.mean(loss))