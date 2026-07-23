import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    result = []
    n = len(y_true)
    for i in range(n):
        p = max(eps, min(y_pred[i], 1 - eps))
        if y_true[i] == 1:
            loss = - math.log(p)
            result.append(loss)
        else:
            loss = - math.log(1 - p)
            result.append(loss)
    return result