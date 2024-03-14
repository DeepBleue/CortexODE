import numpy as np


def normalize_points(points):
    """Normalize a set of 3D points."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return points / norms