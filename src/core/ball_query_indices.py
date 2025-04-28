import numpy as np


def ball_query_indices(points, sampled_points, radius):

    groups = []
    for center in sampled_points:
        diff = points - center  # (N, 3)
        dists = np.linalg.norm(diff, axis=1)  # (N,)
        mask = dists <= radius
        group_indices = np.where(mask)[0]  # 条件を満たすインデックスだけ取得
        groups.append(group_indices)
    return groups
