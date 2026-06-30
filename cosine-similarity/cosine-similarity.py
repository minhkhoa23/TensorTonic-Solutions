import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0
    return np.dot(a, b) / norm