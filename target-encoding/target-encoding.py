def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    sums = {}
    counts = {}
    for category, target in zip(categories, targets):
        if category not in sums:
            sums[category] = 0
            counts[category] = 0

        sums[category] += target
        counts[category] += 1
    means = {}

    for category in sums:
        means[category] = sums[category] / counts[category]
        
    result = []

    for category in categories:
        result.append(means[category])

    return result