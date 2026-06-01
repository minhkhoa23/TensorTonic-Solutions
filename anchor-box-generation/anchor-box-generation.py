import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    if isinstance(feature_size, int):
        feature_h = feature_w = feature_size
    else:
        feature_h, feature_w = feature_size

    if isinstance(image_size, int):
        image_h = image_w = image_size
    else:
        image_h, image_w = image_size

    stride_y = image_h / feature_h
    stride_x = image_w / feature_w

    anchors = []

    for i in range(feature_h):
        for j in range(feature_w):
            center_y = (i + 0.5) * stride_y
            center_x = (j + 0.5) * stride_x

            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)

                    x_min = center_x - w / 2
                    y_min = center_y - h / 2
                    x_max = center_x + w / 2
                    y_max = center_y + h / 2

                    anchors.append([x_min, y_min, x_max, y_max])

    return anchors