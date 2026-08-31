import numpy as np

def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """
    # Write code here
    priorities_arr = np.array(priorities, dtype=np.float64)
    n = len(priorities_arr)

    # 1. Calculate sampling probabilities P(i)
    scaled_priorities = priorities_arr**alpha
    probabilities = scaled_priorities / np.sum(scaled_priorities)

    # 2. Calculate Importance Sampling weights: w_i = (1 / (N * P(i))) ** beta
    weights = (1.0 / (n * probabilities)) ** beta

    # 3. Normalize weights by max(w_i) for stability
    normalized_weights = weights / np.max(weights)

    return [probabilities.tolist(), normalized_weights.tolist()]
    
