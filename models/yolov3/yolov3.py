import torch
import torch.nn as nn
from .backbone.darknet53 import Darknet53
from .detection_head.yolov3_head import YOLOv3Head


class YOLOv3(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_channels: int = 3,
    ):

        super().__init__()

        self.backbone = Darknet53(in_channels=in_channels)

        self.head = YOLOv3Head(n_classes=n_classes)

    def forward(self, x):
        deepest_feature_map, route_connection_2, route_connection_1 = self.backbone(x)
        outputs = self.head(deepest_feature_map, route_connection_2, route_connection_1)

        return outputs
