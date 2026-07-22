import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # YOUR CODE HERE
    hidtohid = np.dot(h_prev, W_hh.T)
    inputtohid = np.dot(x_t, W_xh.T)
    hiddenstate = np.tanh(inputtohid + hidtohid + b_h)
    return hiddenstate