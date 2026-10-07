def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    # Write code here
    weighted_sum = 0.0
    similarity_sum = 0.0

    for i in range(len(user_ratings)):
        # Bỏ item target
        if i == target:
            continue

        rating = user_ratings[i]
        similarity = item_similarities[i]

        # Chỉ lấy rating khác 0 và similarity dương
        if rating != 0 and similarity > 0:
            weighted_sum += rating * similarity
            similarity_sum += similarity

    if similarity_sum == 0:
        return 0.0

    return weighted_sum / similarity_sum