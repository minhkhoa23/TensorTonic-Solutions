import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    """
    Returns a dictionary of feature PSI scores and skew flags.
    """
    # Write code here
    result = {}

    for feature in train_dist:
        train = np.asarray(train_dist[feature], dtype=float)
        serving = np.asarray(serving_dist[feature], dtype=float)
        train = train + eps
        serving = serving + eps
        psi = np.sum((serving - train) * np.log(serving / train))
        result[feature] = {
            'psi': float(psi),
            'skewed': bool(psi > threshold)
        }
    return result