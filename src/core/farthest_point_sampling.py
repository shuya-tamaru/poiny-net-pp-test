import numpy as np


def farthest_point_sampling(points, num_samples):

    N, _ = points.shape

    sampled_indices = np.zeros(num_samples, dtype=int)
    distances = np.full(N, np.inf)

    first_idx = np.random.randint(0, N)
    sampled_indices[0] = first_idx

    for i in range(1, num_samples):
        latest_point = points[sampled_indices[i - 1]]

        diff = points - latest_point  # (N, 3)
        dists = np.linalg.norm(diff, axis=1)  # (N,)

        # 既存の最短距離より小さい場合だけ更新
        distances = np.minimum(distances, dists)

        # 最も「距離が遠い点」を次の代表点にする
        next_idx = np.argmax(distances)
        sampled_indices[i] = next_idx

    return sampled_indices
