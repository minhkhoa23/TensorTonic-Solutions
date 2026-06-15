import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.asarray(T, dtype = float)
    points = np.asarray(points, dtype = float)

    if points.ndim == 1:
        p_h = np.append(points, 1.0)
        transformed = T @ p_h
        return transformed[:3]

    ones = np.ones((points.shape[0], 1))
    p_h = np.hstack([points, ones])

    transformed = p_h @ T.T

    return transformed[:, :3]