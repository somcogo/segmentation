import torch.nn as  nn

class SampleModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Linear(in_channels, out_channels)
        self.layer2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d
        self._init_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.layer2(x)
        x = self.relu(x)
        out = self.batch_norm(x)

        return out