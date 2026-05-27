import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    pmf = np.exp(-lam) * np.power(lam, k) / math.factorial(k)
    cdf = 0.0
    for i in range(0, k + 1):
        temp = np.exp(-lam) * np.power(lam, i) / math.factorial(i)
        cdf += temp
    return (pmf, cdf)