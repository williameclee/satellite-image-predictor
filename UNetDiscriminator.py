import torch.nn as nn


class UNetDiscriminator(nn.Module):
    def __init__(self, in_channels=5):  # 2 input + 3 target
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
