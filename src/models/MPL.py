import torch.nn as nn
import torch.nn.functional as F


class MPL(nn.Module):
    def __init__(self, channels, batch_norm=True):
        super(MPL, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.convs.append(nn.Conv1d(channels[i], channels[i + 1], 1))

            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(channels[i + 1]))

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)

            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            if i < len(self.convs) - 1:
                x = F.relu(x)

        return x
