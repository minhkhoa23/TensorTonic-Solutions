import numpy as np


def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)
    
    mt = beta1 * m + (1 - beta1) * grad
    vt = beta2 * v + (1 - beta2) * np.power(grad, 2)
    m_new = mt / (1 - (np.power(beta1, t)))
    v_new = vt / (1 - (np.power(beta2, t)))
    param_new = param - lr * m_new / (np.sqrt(v_new) + eps)

    return param_new, mt, vt