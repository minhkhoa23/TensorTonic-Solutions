import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    # Write code here
    A = np.asarray(A)
    trace = np.trace(A)
    return float(trace)