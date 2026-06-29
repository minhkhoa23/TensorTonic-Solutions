def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    result = []
    for i in range(len(image)):
        row = []
        for j in range(len(image[0])):
            R = image[i][j][0]
            G = image[i][j][1]
            B = image[i][j][2]
            gray = 0.299 * R + 0.587 * G + 0.114 * B
            row.append(gray)
        result.append(row)
    return result
            
    