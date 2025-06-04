import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels: int, n_iters: int = 1, use_residuals: bool = True):

        super().__init__()

        res_layers = []

        for _ in range(n_iters):
            res_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels, out_channels=channels // 2, kernel_size=1
                    ),
                    nn.BatchNorm2d(num_features=channels // 2),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=channels // 2,
                        out_channels=channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=channels),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            )

        self.layers = nn.ModuleList(res_layers)
        self.use_residuals = use_residuals
        self.n_iters = n_iters

    def forward(self, x):

        for layer in self.layers:
            residual = x
            x = layer(x)

            if self.use_residuals:
                x += residual

        return x
