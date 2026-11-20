import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    if lam < 0 or k < 0:
        raise ValueError("Invalid parameters")

    # Step 1: Calculate PMF
    pmf = (math.exp(-lam) * lam**k) / math.factorial(k)

    # Step 2: Calculate CDF
    cdf = 0.0

    for i in range(k + 1):
        probability = (
            math.exp(-lam) * lam**i
        ) / math.factorial(i)

        cdf += probability

    # Step 3: Return results
    return {
        "pmf": pmf,
        "cdf": cdf
    }