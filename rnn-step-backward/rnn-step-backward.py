import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here
    x_t, h_prev, h_t, W, U, b = cache

    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    h_t = np.asarray(h_t)
    W = np.asarray(W)
    U = np.asarray(U)
    b = np.asarray(b)

    da = dh * (1 - h_t ** 2)

    dx_t = W.T @ da

    dh_prev = U.T @ da

    dW = np.outer(da, x_t)
    dU = np.outer(da, h_prev)

    db = da.copy()

    return dx_t, dh_prev, dW, dU, db