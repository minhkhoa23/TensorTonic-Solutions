def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    """
    Returns the positive-similarity weighted rating prediction.
    """
    # Write code here
    numerator = 0.0
    demoninator = 0.0

    for sim, rating in zip(similarities, ratings):
        if sim > 0:
            numerator += sim * rating
            demoninator += sim
    if demoninator == 0:
        return 0.0
    return numerator / demoninator