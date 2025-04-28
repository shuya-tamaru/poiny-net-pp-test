import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from .Dataset import BuildingPointCloudDataset
from .ball_query import ball_query
from .pad_groups import pad_groups
from ..models.MPL import MPL
from .ball_query_indices import ball_query_indices
from .gather_features import gather_features
from .ClassificationHead import ClassificationHead
from .farthest_point_sampling import farthest_point_sampling


def learning(num_classes):
    dataset = BuildingPointCloudDataset(
        data_dir="./data", num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(dataloader))

    classifier = ClassificationHead(input_dim=256, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    EPOCHS = 50

    for epoch in range(EPOCHS):
        classifier.train()
        total_loss = 0.0

        for data in dataloader:
            points = data.pos.squeeze(0).numpy()

            num_samples = 4096
            sampled_indices = farthest_point_sampling(points, num_samples)
            sampled_points = points[sampled_indices]

            radius = 0.1
            groups = ball_query(points, sampled_points, radius)
            padded_groups = pad_groups(groups, max_num_points=32)
            mpl = MPL(channels=[3, 64, 128], batch_norm=True)
            output = mpl(padded_groups)
            pooled_output = torch.max(output, dim=2)[0]

            num_samples_second = 1024
            second_sampled_indices = farthest_point_sampling(
                sampled_points, num_samples_second)
            second_sampled_points = sampled_points[second_sampled_indices]

            new_radius = 0.2
            new_groups_indices = ball_query_indices(
                sampled_points, second_sampled_points, new_radius)
            new_groups_features = gather_features(
                pooled_output, new_groups_indices)
            mpl2 = MPL(channels=[128, 256], batch_norm=True)
            new_output = mpl2(new_groups_features)
            new_pooled_output = torch.max(new_output, dim=2)[0]
            new_pooled_output = new_pooled_output.detach()

            labels = data.y.squeeze(0)[sampled_indices][second_sampled_indices]

            preds = classifier(new_pooled_output)  # (1024, NUM_CLASSES)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    save_dir = "results/learning"
    save_path = os.path.join(save_dir, "classifier.pth")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(classifier.state_dict(), save_path)
    print("モデル保存完了！")
    return
