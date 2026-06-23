import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.ndim != 2:
        return None
    N, D = X.shape

    if N < 2:
        return None

    mean = np.mean(X, axis = 0)

    X_centered = X - mean

    cov = (1 / (N - 1)) * np.dot(X_centered.T, X_centered)
    return cov