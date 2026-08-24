import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    indices = np.argsort(eigenvalues)[::-1]

    top_components = eigenvectors[:, indices[:k]]

    projected = X_centered @ top_components

    return projected.tolist()