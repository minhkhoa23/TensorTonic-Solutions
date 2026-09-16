def rank_transform(values: list) -> list:
    """
    Returns the one-based average rank of every value.
    """
    # Write code here
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])

    ranks = [0.0] * len(values)

    i = 0

    while i < len(sorted_pairs):
        j = i

        # Tìm các phần tử có cùng giá trị
        while (
            j + 1 < len(sorted_pairs)
            and sorted_pairs[j + 1][1] == sorted_pairs[i][1]
        ):
            j += 1

        # Vì rank bắt đầu từ 1:
        # vị trí i -> rank i + 1
        # vị trí j -> rank j + 1
        average_rank = ((i + 1) + (j + 1)) / 2

        # Gán cùng average rank cho các giá trị bằng nhau
        for k in range(i, j + 1):
            original_index = sorted_pairs[k][0]
            ranks[original_index] = average_rank

        i = j + 1

    return ranks