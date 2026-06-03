import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    clipped = np.asarray(g).copy()
    
    norm = np.linalg.norm(clipped)
    if norm == 0 or max_norm <= 0:
        return clipped
    if norm <= max_norm:
        return clipped
    else:
        scale = clipped * (max_norm / norm)
        return scale