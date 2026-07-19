import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    n_features = X.shape[1]
    I = np.eye(n_features)

    return np.linalg.solve (X.T @ X + lam * I, X.T @ y)