import numpy as np

def histogram_equalize(image: list) -> list:
    """
    Returns the histogram-equalized grayscale image.
    """
    # Write code here
    img = np.asarray(image, dtype=np.uint8)

    # 1. Histogram
    hist = np.bincount(img.flatten(), minlength=256)

    # 2. CDF
    cdf = hist.cumsum()

    # 3. First non-zero CDF value
    cdf_min = cdf[cdf > 0][0]

    # 4. Normalize CDF to [0, 255]
    total_pixels = img.size

    mapping = np.round(
        (cdf - cdf_min)
        / (total_pixels - cdf_min)
        * 255
    )

    mapping = np.clip(mapping, 0, 255).astype(np.uint8)

    # 5. Replace old pixels with equalized values
    result = mapping[img]

    return result.tolist()