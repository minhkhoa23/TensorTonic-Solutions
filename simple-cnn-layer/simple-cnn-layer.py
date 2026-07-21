import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    x = np.asarray(x)
    W = np.asarray(W)
    b = np.asarray(b)

    N, C_in, H, width = x.shape

    C_out, C_in_w, KH, KW = W.shape

    if C_in != C_in_w:
        raise ValueError('input error')

    H_out = H - KH + 1
    W_out = width - KW + 1

    y = np.zeros((N, C_out, H_out, W_out), dtype = float)

    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    region = x[
                        n,
                        :,
                        i:i + KH,
                        j:j + KW
                    ]

                    y[n, c_out, i, j] = (
                        np.sum(region * W[c_out]) + b[c_out]
                    )
    return y