import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def slerp(v0, v1, t):
    """
    Spherical linear interpolation between two vectors.
    v0, v1: the vectors
    t: interpolation factor between 0 and 1
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    
    dot = np.dot(v0, v1)
    dot = np.clip(dot, -1.0, 1.0)  # Clip to handle potential numerical errors

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    if sin_theta_0 == 0:
        return (1.0 - t) * v0 + t * v1

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * v0 + s1 * v1

def generate_vectors_between(v0, v1, n):
    """
    Generate n vectors between v0 and v1.
    v0, v1: the input vectors
    n: number of vectors to generate
    """
    ts = np.linspace(0, 1, n)
    return [slerp(v0, v1, t) for t in ts]

def generate_multiple_vectors(v0s, v1s, n):
    """
    Generate vectors between multiple v0 and v1 vectors.
    v0s, v1s: lists of vectors
    n: number of vectors to generate for each pair
    """
    all_vectors = []
    for v0, v1 in tqdm(zip(v0s, v1s),total=len(v0s)):
        vectors = generate_vectors_between(v0, v1, n)
        all_vectors.extend(vectors)
    return all_vectors

# def normalize_points(points):
#     """Normalize a set of 3D points."""
#     norms = np.linalg.norm(points, axis=1, keepdims=True)
#     norms[norms == 0] = 1  # Avoid division by zero
#     return points / norms

# v0s = [np.array([1, 0, 0]), np.array([0, 0, 1])]
# v1s = [np.array([0, 1, 0]), np.array([-1, 0, 0])]
# all_vectors = generate_multiple_vectors(v0s, v1s, 5)


# print(v1s[0])
# print(all_vectors[4])


# # Plotting in 3D space
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot each vector as a line from origin
# for vec in all_vectors:
#     ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]])

# # Set the limits and labels for each axis
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()