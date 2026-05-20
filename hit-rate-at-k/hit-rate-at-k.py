def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    n = len(ground_truth)
    if n == 0:
        return 0.0
    hit = 0
    for i in range(0, len(ground_truth)):
        topk = recommendations[i][:k]
        if ground_truth[i][0] in topk:
            hit += 1
    return hit / len(ground_truth)