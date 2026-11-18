def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    # Write code here
    if not data:
        return []

    rows = len(data)
    cols = len(data[0])

    result = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):
        # Step 1: Find min and max of column j
        col_min = min(data[i][j] for i in range(rows))
        col_max = max(data[i][j] for i in range(rows))

        # Step 2: Calculate range
        diff = col_max - col_min

        # Step 3: Scale each element
        for i in range(rows):
            if diff == 0:
                result[i][j] = 0.0
            else:
                result[i][j] = (
                    data[i][j] - col_min
                ) / diff

    return result