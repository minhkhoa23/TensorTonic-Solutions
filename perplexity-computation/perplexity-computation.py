def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    total_log_prob = 0.0
    n = len(actual_tokens)

    for i in range(n):
        token_id = actual_tokens[i]
        probability = prob_distributions[i][token_id]

        total_log_prob += math.log(probability)

    average_log_prob = total_log_prob / n

    return math.exp(-average_log_prob)