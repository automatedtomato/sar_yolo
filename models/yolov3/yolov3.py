import torch
import torch.nn as nn
from .backbone.darknet53 import Darknet53
from .detection_head.yolov3_head import YOLOv3Head

from typing import Any
from logging import getLogger

logger = getLogger(__name__)

class YOLOv3(nn.Module):
    
    """
    YOLOv3 model

    Args:
        === Either config or n_classes must be provided ===
        n_classes (int, optional): Number of classes 
        in_channels (int): Number of input channels, default to 3
        config (dict, optional): Configuration dictionary
    """
    def __init__(
        self,
        config: dict[str, Any] = None,
        *,
        n_classes: int = None,
        in_channels: int = 3,
    ):


        super().__init__()
        
        if config is not None and n_classes is None:
            self.n_classes = config["model"]["n_classes"]
        elif config is None and n_classes is not None:
            self.n_classes = n_classes
        elif config is not None and n_classes is not None:
            logger.warining("Both config and n_classes are provided. Using n_classes.")
            self.n_classes = n_classes
        else:
            raise ValueError("Either config or n_classes must be provided")

        self.backbone = Darknet53(in_channels=in_channels)

        self.head = YOLOv3Head(n_classes=self.n_classes)
        

    def forward(self, x):
        deepest_feature_map, route_connection_2, route_connection_1 = self.backbone(x)
        outputs = self.head(deepest_feature_map, route_connection_2, route_connection_1)

        return outputs
