def rating_normalization(matrix: list) -> list:
    """
    Returns the mean-centered user-item matrix.
    """
    # Write code here
    result = []

    for row in matrix:
        rated_values = [x for x in row if x != 0]

        # User chưa rating bất kỳ item nào
        if len(rated_values) == 0:
            result.append([0.0 for x in row])
            continue

        mean = sum(rated_values) / len(rated_values)

        normalized_row = [
            x - mean if x != 0 else 0.0
            for x in row
        ]

        result.append(normalized_row)

    return result