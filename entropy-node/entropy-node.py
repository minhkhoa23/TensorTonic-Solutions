import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y)
    _, count = np.unique(y, return_counts = True)

    probs = count / y.size

    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))

    return entropy
    