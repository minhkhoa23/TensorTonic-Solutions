import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    # Write code here
    # Step 1: Convert to NumPy array
    x = np.asarray(x, dtype=float)

    # Step 2: Initialize random generator
    rng = np.random.default_rng(seed)

    n = len(x)

    # Step 3: Generate bootstrap means
    means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(x, size=n, replace=True)
        means.append(np.mean(sample))

    means = np.asarray(means)

    # Step 4: Calculate bootstrap mean
    boot_mean = np.mean(means)

    # Step 5: Calculate confidence interval
    alpha = 1 - ci

    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))

    return {
        "bootstrap_mean": float(boot_mean),
        "lower": float(lower),
        "upper": float(upper)
    }