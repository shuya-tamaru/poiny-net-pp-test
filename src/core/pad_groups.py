import torch
import numpy as np


def pad_groups(groups, max_num_points):
    M = len(groups)
    padded_groups = np.zeros((M, max_num_points, 3), dtype=np.float32)

    for i, group in enumerate(groups):
        n = group.shape[0]
        if n > max_num_points:
            padded_groups[i] = group[:max_num_points]
        else:
            repeat_indices = np.random.choice(n, max_num_points, replace=True)
            padded_groups[i] = group[repeat_indices]

    padded_groups = np.transpose(padded_groups, (0, 2, 1))

    return torch.from_numpy(padded_groups)
