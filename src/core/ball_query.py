import numpy as np


def ball_query(points, sampled_points, radius):
    groups = []
    for center in sampled_points:
        diff = points - center
        dists = np.linalg.norm(diff, axis=1)
        mask = dists < radius

        group = points[mask]
        groups.append(group)
    return groups
