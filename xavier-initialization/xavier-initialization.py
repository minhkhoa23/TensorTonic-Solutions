import numpy as np
def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    W = np.asarray(W, dtype = float)
    limit = np.sqrt(6 / (fan_in + fan_out))
    W = W * 2 * limit - limit
    return W