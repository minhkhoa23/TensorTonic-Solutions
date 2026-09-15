def adjusted_cosine_similarity(ratings_matrix: list, item_i: int, item_j: int) -> float:
    """
    Returns the adjusted cosine similarity between the requested items.
    """
    # Write code here
    numerator = 0.0
    sum_i_squared = 0.0
    sum_j_squared = 0.0

    for user_ratings in ratings_matrix:

        # Chỉ xét user đã rating cả 2 item
        if user_ratings[item_i] == 0 or user_ratings[item_j] == 0:
            continue

        # Các rating thực sự của user, bỏ rating 0
        rated_items = [
            rating for rating in user_ratings
            if rating != 0
        ]

        # Rating trung bình của user
        user_mean = sum(rated_items) / len(rated_items)

        # Rating sau khi điều chỉnh theo mean
        adjusted_i = user_ratings[item_i] - user_mean
        adjusted_j = user_ratings[item_j] - user_mean

        # Tử số
        numerator += adjusted_i * adjusted_j

        # Hai phần của mẫu số
        sum_i_squared += adjusted_i ** 2
        sum_j_squared += adjusted_j ** 2

    denominator = (sum_i_squared ** 0.5) * (sum_j_squared ** 0.5)

    # Tránh chia cho 0
    if denominator == 0:
        return 0.0

    return numerator / denominator