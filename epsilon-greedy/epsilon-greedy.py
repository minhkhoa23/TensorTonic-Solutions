import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    # Write code here
    rng = np.random.default_rng(seed)

    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))

    return int(np.argmax(q_values))