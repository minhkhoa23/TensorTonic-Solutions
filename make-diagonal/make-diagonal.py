import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    result = np.zeros((len(v), len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            if i == j:
                result[i][j] = v[i]
    return result
