import torch
import torch.nn as nn

from ..blocks import ScalePrediction
from ..blocks import CNNBlock
from ..blocks import ResidualBlock


class YOLOv3Head(nn.Module):
    def __init__(self, n_classes: int):

        super().__init__()
        self.n_classes = n_classes

        # scale 1 (deepest feature map: 8x8)
        self.predict_1 = nn.Sequential(
            CNNBlock(
                in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0
            ),
            CNNBlock(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(channels=1024, use_residuals=False, n_iters=1),
            CNNBlock(
                in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0
            )
        )
        
        self.scale_pred_1 = ScalePrediction(in_channels=512, n_classes=self.n_classes)

        # Upsampling
        self.upsample_1 = nn.Sequential(
            CNNBlock(
                in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
            ),
            nn.Upsample(scale_factor=2),
        )

        # Scale 2 (intermediate feature map: 16x16)
        self.predict_2 = nn.Sequential(
            # Concatenation
            CNNBlock(
                in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0
            ),
            CNNBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(channels=512, use_residuals=False, n_iters=1),
            CNNBlock(
                in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
            )
        )
        
        self.scale_pred_2 = ScalePrediction(in_channels=256, n_classes=self.n_classes)

        self.upsample_2 = nn.Sequential(
            CNNBlock(
                in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0
            ),
            nn.Upsample(scale_factor=2),
        )

        # Scale 3 (shallowest feature map: 32x32)
        self.predict_3 = nn.Sequential(
            # Concatenation
            CNNBlock(
                in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0
            ),
            CNNBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(channels=256, use_residuals=False, n_iters=1),
            CNNBlock(
                in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0
            )
        )
        
        self.scale_pred_3 = ScalePrediction(in_channels=128, n_classes=self.n_classes)

    def forward(self, deepest_feature_map, route_connection_2, route_connection_1):
        outputs = []
        
        # Predict scale1 (8x8)
        x = deepest_feature_map
        x = self.predict_1(x)
        pred_1 = self.scale_pred_1(x)
        outputs.append(pred_1)

        # Upsample 512 to 256, 8x8 to 16x16
        x_up = self.upsample_1(x)

        # Concatenate
        x_concat = torch.cat((x_up, route_connection_2), dim=1)  # 256 + Darknet53: 512 = 768ch

        # Predict scale2 (16x16)
        x = x_concat
        x = self.predict_2(x_concat)
        pred_2 = self.scale_pred_2(x)
        outputs.append(pred_2)
        
        # Upsample 256 to 128, 16x16 to 32x32
        x_up = self.upsample_2(x)
        
        # Concatenate
        x_concat = torch.cat((x_up, route_connection_1), dim=1)  # 128 + Darknet53: 256 = 384ch

        # Predict scale3 (32x32)
        x = x_concat
        x = self.predict_3(x)
        pred_3 = self.scale_pred_3(x)
        outputs.append(pred_3)

        return outputs
