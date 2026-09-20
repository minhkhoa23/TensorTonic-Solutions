def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    """
    Returns a two-dimensional list with shape (new_h, new_w).
    """
    # Write code here
    H = len(image)
    W = len(image[0])

    result = []

    for i in range(new_h):
        row = []

        # Tọa độ y trên ảnh gốc
        if new_h == 1:
            y = 0
        else:
            y = i * (H - 1) / (new_h - 1)

        y0 = int(y)
        y1 = min(y0 + 1, H - 1)
        dy = y - y0

        for j in range(new_w):

            # Tọa độ x trên ảnh gốc
            if new_w == 1:
                x = 0
            else:
                x = j * (W - 1) / (new_w - 1)

            x0 = int(x)
            x1 = min(x0 + 1, W - 1)
            dx = x - x0

            # Nội suy theo chiều ngang
            v0 = image[y0][x0] * (1 - dx) + image[y0][x1] * dx
            v1 = image[y1][x0] * (1 - dx) + image[y1][x1] * dx

            # Nội suy theo chiều dọc
            value = v0 * (1 - dy) + v1 * dy

            row.append(value)

        result.append(row)

    return result