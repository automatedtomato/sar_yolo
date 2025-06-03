import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_batchnorm: bool = True, **kwargs
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=not use_batchnorm,
            **kwargs
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.activation = nn.LeakyReLU(negative_slope=0.1)

        self.use_batchnorm = use_batchnorm

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
            return self.activation(x)
        return x
