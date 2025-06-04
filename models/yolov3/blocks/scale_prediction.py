import torch
import torch.nn as nn


class ScalePrediction(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):

        super().__init__()

        # Prediction layer (3 anchors, 5 outputs)
        self.pred = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=2 * in_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=(n_classes + 5) * 3,
                kernel_size=1,
            ),
        )

        self.n_classes = n_classes

    def forward(self, x):
        output = self.pred(x)
        output = output.view(
            x.size(0),  # Batch size
            3,  # Number of anchors
            self.n_classes
            + 5,  # Number of outputs (classes + x, y, w, h, objectness_score)
            x.size(2),  # Width
            x.size(3),  # Height
        )
        output = output.permute(
            0, 1, 3, 4, 2
        )  # (Bach size, Number of anchors, Width, Height, Number of outputs)

        return output
