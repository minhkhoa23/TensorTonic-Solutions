import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    values = np.asarray(values)
    transitions = np.asarray(transitions)
    rewards = np.asarray(rewards)

    n_states = values.shape[0]
    new_values = np.zeros(n_states)

    for s in range(n_states):
        action_values = []
    
        for a in range(transitions.shape[1]):
            expected_value = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
            action_values.append(expected_value)
        new_values[s] = max(action_values)
    return list(new_values)