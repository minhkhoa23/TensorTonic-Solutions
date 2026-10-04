import math

def rotate_image(image: list, angle_degrees: float) -> list:
    """
    Returns the counterclockwise nearest-neighbor rotation.
    """
    # Write code here
    rows = len(image)
    cols = len(image[0])

    output = [[0 for _ in range(cols)] for _ in range(rows)]

    cx = (cols - 1) / 2
    cy = (rows - 1) / 2

    theta = math.radians(angle_degrees)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    for y in range(rows):
        for x in range(cols):

            dx = x - cx
            dy = y - cy

            # inverse mapping
            src_x = cos_t * dx - sin_t * dy + cx
            src_y = sin_t * dx + cos_t * dy + cy

            src_x = round(src_x)
            src_y = round(src_y)

            if 0 <= src_x < cols and 0 <= src_y < rows:
                output[y][x] = image[src_y][src_x]

    return output