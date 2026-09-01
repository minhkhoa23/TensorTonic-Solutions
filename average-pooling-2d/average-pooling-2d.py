import numpy as np
def average_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping average-pooled windows.
    """
    # Write code here
    X = np.asarray(X)
    h, w = X.shape

    out_h = h // pool_size

    out_w = w // pool_size

    output = np.zeros((out_h, out_w))

    for i in range(out_h):

        for j in range(out_w):

            window = X[

                i * pool_size:(i + 1) * pool_size,

                j * pool_size:(j + 1) * pool_size

            ]

            output[i, j] = np.mean(window)

    return output.tolist()