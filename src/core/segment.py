import torch
from .ClassificationHead import ClassificationHead
from torch_geometric.loader import DataLoader
from .Dataset import BuildingPointCloudDataset
from .ball_query import ball_query
from .pad_groups import pad_groups
from ..models.MPL import MPL
from .ClassificationHead import ClassificationHead
from .save_ply import save_ply


def segment(num_classes):
    model = ClassificationHead(input_dim=256, num_classes=num_classes)
    model.load_state_dict(torch.load("results/learning/classifier.pth"))
    model.eval()

    dataset = BuildingPointCloudDataset(
        data_dir="./data", num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data = next(iter(dataloader))
    points = data.pos.squeeze(0).numpy()  # (N, 3)
    colors = data.color.squeeze(0).numpy()  # (N, 3)
    print(len(points))

    # --- 特徴抽出スタート ---
    groups = ball_query(points, points, radius=0.1)
    print(len(points))
    padded_groups = pad_groups(groups, max_num_points=32)
    mpl = MPL(channels=[3, 64, 128], batch_norm=True)
    output = mpl(padded_groups)
    pooled_output = torch.max(output, dim=2)[0]  # (N, 128)

    mpl2 = MPL(channels=[128, 256], batch_norm=True)
    output2 = mpl2(pooled_output.unsqueeze(-1))  # Conv1Dに合わせるために次元追加
    features = output2.squeeze(-1)  # (N, 256)

    # --- 推論 ---
    with torch.no_grad():
        preds = model(features)  # (N, num_classes)

    pred_labels = preds.argmax(dim=1)

    # --- 保存 ---
    print(len(points))
    save_ply(
        points,
        colors,
        pred_labels.cpu().numpy(),
        "results/learning/predicted.ply"
    )

    print("全点推論完了！")
    return
