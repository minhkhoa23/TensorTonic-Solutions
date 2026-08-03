import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, dtype = float)
    eps = 1e-10

    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm > eps:
            return v / norm
        return v.copy()
    elif v.ndim == 2:
        norms = np.linalg.norm(v, axis = 1, keepdims = True)
        return np.where(norms > eps, v / np.where(norms > eps, norms, 1.0), v)
    else:
        raise ValueError