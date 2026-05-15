import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    m = len(A)
    n = len(A[0])
    result = np.zeros((n, m))
    for i in range(0, m):
        for j in range(0, n):
            result[j][i] = A[i][j]

    return result