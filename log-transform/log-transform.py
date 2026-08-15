import math

def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    result = []
    for i in range (0, len(values)):
        temp = math.log(1 + values[i])
        result.append(temp)
    return result