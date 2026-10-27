import math

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    # Write code here
    losses = []
    eps = 1e-12

    for p, y in zip(predictions, targets):
        p = max(eps, min(1 - eps, p))

        # Xác suất của lớp đúng
        pt = p if y == 1 else 1 - p

        # Focal loss
        loss = -alpha * (1 - pt)**gamma * math.log(pt)

        losses.append(loss)

    return sum(losses) / len(losses)
