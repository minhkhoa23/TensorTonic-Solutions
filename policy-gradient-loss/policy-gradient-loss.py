import torch
def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    baseline = sum(returns) / len(returns)

    advantages = [G - baseline for G in returns]

    loss = 0

    for log_prob, advantage in zip(log_probs, advantages):
        loss += log_prob * advantage
        
    loss = -loss / len(log_probs)

    return loss