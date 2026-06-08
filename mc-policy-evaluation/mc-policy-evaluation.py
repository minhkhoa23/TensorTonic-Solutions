import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)

    for episode in episodes:
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, reward = episode[t]

            G = reward + gamma * G

            if state not in [episode[i][0] for i in range(t)]:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1

    V = np.zeros(n_states)

    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]
    return V
