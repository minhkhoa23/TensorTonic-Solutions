import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    # Step 1: Center each feature
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute numerator matrix
    numerator = X_centered.T @ X_centered

    # Step 3: Compute denominator
    norms = np.sqrt(np.sum(X_centered ** 2, axis=0))
    denominator = np.outer(norms, norms)

    # Step 4: Compute correlation matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denominator

    return corr
