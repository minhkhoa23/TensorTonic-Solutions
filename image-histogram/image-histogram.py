import numpy as np
def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    image = np.asarray(image)
    result = np.zeros(256)

    for i in image.flatten():
        result[i] += 1

    return list(result)
    