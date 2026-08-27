import numpy as np

def chi2_independence(C: list) -> dict:
    """
    Returns a dictionary with chi2 and expected.
    """
    # Write code here
    C = np.asarray(C, dtype=float)

    row_sums = np.sum(C, axis=1, keepdims=True)

    col_sums = np.sum(C, axis=0, keepdims=True)

    total = np.sum(C)

    expected = row_sums @ col_sums / total

    chi2 = np.sum((C - expected) ** 2 / expected)

    return {
        'chi2': float(chi2),
        'expected': expected
    }