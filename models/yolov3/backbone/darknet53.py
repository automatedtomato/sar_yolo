import torch
import torch.nn as nn

from ..blocks import CNNBlock, ResidualBlock


class Darknet53(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.layer1 = nn.Sequential(  # Route connection 1
            CNNBlock(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # output: 256x256
            CNNBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            ),  # output: 128x128
            ResidualBlock(channels=64, n_iters=1),  # output: 128x128
            CNNBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),  # output: 64x64
            ResidualBlock(channels=128, n_iters=2),  # output: 64x64
            CNNBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # output: 32x32
            ResidualBlock(channels=256, n_iters=8),  # output: 32x32
        )

        self.layer2 = nn.Sequential(
            # Route connection 2
            CNNBlock(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # output: 16x16
            ResidualBlock(channels=512, n_iters=8),  # output: 16x16
        )

        self.layer3 = nn.Sequential(
            # Route connection 3: deepest features
            CNNBlock(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # output: 8x8
            ResidualBlock(channels=1024, n_iters=4),  # output: 8x8
        )

    def forward(self, x):
        route_connection_1 = self.layer1(x)

        route_connection_2 = self.layer2(route_connection_1)

        deepest_features = self.layer3(route_connection_2)

        return deepest_features, route_connection_2, route_connection_1
