import numpy as np

def compute_advantage(states: list, rewards: list, V: list, gamma: float) -> np.ndarray:
    """
    Returns the advantages as a NumPy array.
    """
    # Write code here
    states = np.asarray(states, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    V = np.asarray(V, dtype=float)

    returns = np.zeros(len(rewards), dtype=float)

    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    advantages = returns - V[states]

    return np.round(advantages, 4)