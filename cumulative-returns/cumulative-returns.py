def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    # Write code here
    result = []
    cumulative = 1.0

    for r in returns:
        cumulative *= (1 + r)
        result.append(cumulative - 1)

    return result