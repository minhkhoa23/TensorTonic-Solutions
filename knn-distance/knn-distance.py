import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]

    distances = np.sqrt(np.sum(diff ** 2, axis = 2))
    sorted_indices = np.argsort(distances, axis = 1) [:, :k]
    
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    result = np.full((n_test, k), -1, dtype=int)

    m = min(k, n_train)
    result[:, :m] = sorted_indices[:, :m]

    return result