from plyfile import PlyData, PlyElement
import numpy as np


def save_ply(points, colors, labels, save_path):
    colors = (colors * 255).astype(np.uint8)
    vertex = np.array(
        [(p[0], p[1], p[2], c[0], c[1], c[2], int(l))
         for p, c, l in zip(points, colors, labels)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
               ('green', 'u1'), ('blue', 'u1'), ('label', 'i4')]
    )

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(save_path)
    print(f"ラベル付きPLYを保存しました: {save_path}")
