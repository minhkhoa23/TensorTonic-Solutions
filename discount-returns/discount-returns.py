def discount_returns(rewards: list, gamma: float) -> list:
    """
    Returns the discounted return at every timestep.
    """
    # Write code here
    result = [0.0] * len(rewards)
    running_return = 0.0

    for i in range(len(rewards) - 1, -1, -1):
        running_return = rewards[i] + gamma * running_return
        result[i] = running_return

    return result