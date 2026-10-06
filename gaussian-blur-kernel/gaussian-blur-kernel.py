import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    # Write code here
    kernel = []

    center = size // 2

    total = 0

    for i in range(size):
        row = []

        for j in range(size):
            x = j - center
            y = i - center

            value = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            row.append(value)
            total += value
        kernel.append(row)

    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total
    return kernel