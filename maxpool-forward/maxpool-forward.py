import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    X = np.asarray(X)
    H, W = X.shape

    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            row = i * stride
            col = j * stride

            window = X[row: row + pool_size, col: col + pool_size]

            out[i, j] = np.max(window)
    return out.tolist()