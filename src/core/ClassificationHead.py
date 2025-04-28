import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 256 -> 128
        self.fc2 = nn.Linear(128, num_classes)  # 128 -> NUM_CLASSES

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
