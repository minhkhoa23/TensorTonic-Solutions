def polynomial_features(values: list, degree: int) -> list:
    """
    Returns powers from zero through degree for every value.
    """
    # Write code here
    result = []

    for x in values:
        features = []

        for p in range(degree + 1):
            features.append(x ** p)

        result.append(features)

    return result