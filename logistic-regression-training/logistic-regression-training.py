import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    for i in range(0, steps):
        z = np.dot(X, w) + b
        y_hat = _sigmoid(z)

        dz = y_hat - y
        dw = np.dot(X.T, dz) / m
        db = np.sum(dz) / m

        w = w - lr * dw
        b = b - lr * db
    return w ,b