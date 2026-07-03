def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    result = []
    for i in range(len(x)):
        if x[i] > 0:
            result.append(x[i])
        else:
            result.append(alpha * (math.exp(x[i]) - 1))
    return result