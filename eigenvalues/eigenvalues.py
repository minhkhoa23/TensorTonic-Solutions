import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix)
    except Exception:
        return None
    if matrix.ndim != 2:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    eigen = np.linalg.eigvals(matrix)

    idx = np.lexsort((eigen.imag, eigen.real))

    return eigen[idx]