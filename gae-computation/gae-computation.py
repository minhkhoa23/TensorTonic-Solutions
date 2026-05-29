def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    # Write code here
    T = len(rewards)
    advantages = [0.0] * T

    gae_accumulated = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < len (values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae_accumulated = delta + gamma * lam * gae_accumulated
        advantages[t] = gae_accumulated
    return advantages