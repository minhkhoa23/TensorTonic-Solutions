def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    """
    Returns a copy with missing ratings replaced by user or item means.
    """
    # Write code here
    result = [row[:] for row in ratings_matrix]

    if mode == "user":
        for i in range(len(result)):
            ratings = [x for x in result[i] if x != 0]

            if ratings:
                mean = sum(ratings) / len(ratings)

                for j in range(len(result[i])):
                    if result[i][j] == 0:
                        result[i][j] = mean

    elif mode == "item":
        rows = len(result)
        cols = len(result[0])

        for j in range(cols):
            ratings = []

            for i in range(rows):
                if result[i][j] != 0:
                    ratings.append(result[i][j])

            if ratings:
                mean = sum(ratings) / len(ratings)

                for i in range(rows):
                    if result[i][j] == 0:
                        result[i][j] = mean

    return result