import os
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
from torch_geometric.data import Data

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POINT_DIM = 3


class BuildingPointCloudDataset(Dataset):
    def __init__(self, data_dir, num_classes, transform=None):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.file_list = [f for f in os.listdir(
            data_dir) if f.endswith('.ply')]

        self.transform = transform

        if len(self.file_list) == 0:
            raise ValueError(f"{data_dir} にPLYファイルが見つかりません")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        ply_path = os.path.join(self.data_dir, self.file_list[idx])
        points, colors, labels = self.read_ply(ply_path)
        points = self.normalize_points(points)
        labels_contiguous = np.copy(labels)

        data = Data(
            pos=torch.from_numpy(points).float(),  # (N,3)
            y=torch.from_numpy(labels_contiguous).long(),  # (N,)
            color=torch.from_numpy(colors).float(),  # (N,)
        )

        if self.transform:
            data = self.transform(data)

        return data

    def read_ply(self, ply_path):
        ply_data = PlyData.read(ply_path)
        vertex = ply_data['vertex']
        points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        colors = np.vstack(
            [vertex['red'], vertex['green'], vertex['blue']]).T / 255.0
        labels = vertex['scalar_Classification']
        return points, colors, labels

    def normalize_points(self, points):
        centroid = np.mean(points, axis=0, keepdims=True)
        points = points - centroid
        furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
        points = points / furthest_distance
        return points
