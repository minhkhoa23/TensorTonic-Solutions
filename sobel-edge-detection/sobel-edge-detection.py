import math

def sobel_edges(image: list) -> list:
    """
    Returns the zero-padded Sobel gradient magnitude at every pixel.
    """
    # Write code here
    if not image or not image[0]:
        return []

    h = len(image)
    w = len(image[0])

    gx_kernel = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    gy_kernel = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]

    result = [[0.0 for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):

            gx = 0.0
            gy = 0.0

            # Duyệt kernel 3x3
            for ki in range(3):
                for kj in range(3):

                    # Tọa độ trên ảnh
                    ni = i + ki - 1
                    nj = j + kj - 1

                    # Zero padding:
                    # ngoài ảnh -> pixel = 0
                    if 0 <= ni < h and 0 <= nj < w:
                        pixel = image[ni][nj]
                    else:
                        pixel = 0

                    gx += pixel * gx_kernel[ki][kj]
                    gy += pixel * gy_kernel[ki][kj]

            result[i][j] = math.sqrt(gx * gx + gy * gy)

    return result