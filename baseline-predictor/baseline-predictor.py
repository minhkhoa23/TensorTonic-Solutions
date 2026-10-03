def baseline_predict(ratings_matrix: list, target_pairs: list) -> list:
    """
    Returns the baseline predictions for the requested user-item pairs.
    """
    # Write code here
    ratings = []

    for row in ratings_matrix:
        for rating in row:
            if rating != 0:
                ratings.append(rating)

    global_mean = sum(ratings) / len(ratings)

    num_users = len(ratings_matrix)
    num_items = len(ratings_matrix[0])

    # User biases
    user_bias = []

    for row in ratings_matrix:
        rated = [x for x in row if x != 0]

        if len(rated) == 0:
            user_bias.append(0)
        else:
            user_mean = sum(rated) / len(rated)
            user_bias.append(user_mean - global_mean)

    # Item biases
    item_bias = []

    for j in range(num_items):
        rated = []

        for i in range(num_users):
            if ratings_matrix[i][j] != 0:
                rated.append(ratings_matrix[i][j])

        if len(rated) == 0:
            item_bias.append(0)
        else:
            item_mean = sum(rated) / len(rated)
            item_bias.append(item_mean - global_mean)

    # Predictions
    predictions = []

    for user, item in target_pairs:
        pred = global_mean + user_bias[user] + item_bias[item]
        predictions.append(pred)

    return predictions