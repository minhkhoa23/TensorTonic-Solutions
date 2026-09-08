import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.asarray(p, dtype = float)
    q = np.asarray(q, dtype = float)

    p = p + eps
    q = q + eps

    kl = np.sum(p * np.log(p / q))

    return float(kl)