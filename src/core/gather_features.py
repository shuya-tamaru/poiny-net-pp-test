import torch
import numpy as np


def gather_features(features, groups_indices, max_num_points=32):
    M = len(groups_indices)
    C = features.shape[1]

    # 空のバッファを作る
    gathered = torch.zeros((M, C, max_num_points), dtype=features.dtype)

    for i, indices in enumerate(groups_indices):
        n = len(indices)
        if n >= max_num_points:
            selected = indices[:max_num_points]
        else:
            selected = np.random.choice(indices, max_num_points, replace=True)

        gathered[i] = features[selected].T

    return gathered
