from sklearn.neighbors import NearestNeighbors


def propagate_labels(source_points, source_labels, target_points):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(source_points)
    distances, indices = nbrs.kneighbors(target_points)
    target_labels = source_labels[indices[:, 0]]
    return target_labels
