import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_stream import DataStream
import torchvision.transforms as transforms
from typing import Any, Optional

from utils import load_config

from logging import getLogger

logger = getLogger(__name__)

class SaRDataset(Dataset):
    """
    Dataset class for the SaR dataset
    
    Args:
        data_stream (DataStream): DataStream object instance
        config (dict[str, Any]): Configuration dictionary
        output_size (int, optional): Output size. Defaults to -1.
        transform (Optional[transforms.Compose], optional): Transformation pipeline. Defaults to None.
    """
    def __init__(
        self,
        data_stream: DataStream,
        config: dict[str, Any],
        output_size: int = -1,
        transform: Optional[transforms.Compose]=None
        ):
        
        self.config = config

        
        self.data_stream = data_stream
        self.image_paths, self.annot_paths = data_stream.generate_data(
            img_extension=self.config['data']['img_ext'],
            annot_extension=self.config['data']['annot_ext'],
            img_prefix=config['data']['img_path'],
            annot_prefix=config['data']['annot_path'],
            output_size=output_size,
            return_sep=True
        )
        
        # Load from config
        self.image_size = config['model']['input_size']
        self.grid_sizes = config['model']['grid_sizes']
        self.n_classes = config['model']['n_classes']
        self.anchors = config['model']['anchors']
        
        
        if transform is None :
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        if len(self.image_paths) != len(self.annot_paths):
            logger.error(f'Number of images and annotations must be the same, but got {len(self.image_paths)} images and {len(self.annot_paths)} annotations')
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Get 1 item from dataset
        
        Args:
            idx (int): Index of the item
        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                image: [channel, height, width]
                targets: [target_scale1, target_scale2, target_scale3]
        """
        # 1. Load image
        image = self._load_and_parse_image(idx)
        
        # 2. Load and parse annotations
        annots = self._load_and_parse_annot(idx)
        
        # 3. Apply YOLOv3 target format
        targets = self._convert_to_target(annots)
        
        return image, targets
        
    def _load_and_parse_image(self, idx: int) -> torch.Tensor:
        try:
            pil_image = self.data_stream.load_image(self.image_paths[idx])
            if pil_image is None:
                return torch.zeros(3, self.image_size, self.image_size)
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
            image_tensor = self.transform(pil_image)
            
            return image_tensor
        
        except Exception as e:
            logger.error(f'Failed to load image from {self.image_paths[idx]}: {e}')
            return torch.zeros(3, self.image_size, self.image_size)
        
    def _load_and_parse_annot(self, idx: int):
        try:
            annot_lines = self.data_stream.load_annot(self.annot_paths[idx], self.config['data']['annot_ext'])
            if annot_lines is None:
                return []
            
            annots = self.parse_annot(annot_lines)
            return annots
            
        except Exception as e:
            logger.error(f'Failed to load annotation from {self.annot_paths[idx]}: {e}')
            return []
        
    def parse_annot(self, annot: str) -> list[tuple[int, float, float, float, float, int]]:
        """
        Parse YOLO annotations
        
        Args:
            annot (str): YOLO annotation string
        Returns:
            list[tuple[int, float, float, float, float, int]]: List of tuples (class_id, x_center, y_center, width, height, label)
        """

        annotations = []
        
        for line in annot:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 6:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    label = int(parts[5])
                    
                    annotations.append((class_id, x_center, y_center, width, height, label))
                    
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    annotations.append((class_id, x_center, y_center, width, height, -1))
                    
                elif len(parts) <= 4 or len(parts) > 6:
                    raise ValueError(f'Invalid annotation format: {line}')
                
        return annotations
    
    def _convert_to_target(self, annots: list[tuple[int, float, float, float, float, int]]) -> list[torch.Tensor]:
        """
        Convert YOLO annotations to target format

        Args:
            annots (list[tuple[int, float, float, float, float, int]]):
                List of tuples (class_id, x_center, y_center, width, height, label)

        Returns:
            list[torch.tensor]: List of tensors, shape - 3 * [batch, anchors, w, h, (n_classes + 5)]
        """
        targets = []
        
        for _, (grid_size, anchors) in enumerate(zip(self.grid_sizes, self.anchors)):
            target = torch.zeros(3, grid_size, grid_size, 5 + self.n_classes)
            
            for annot in annots:
                class_id, x_center, y_center, width, height, label = annot
                
                grid_x = int(x_center * grid_size)
                grid_y = int(y_center * grid_size)
                
                if grid_x >= grid_size or grid_y >= grid_size or grid_x < 0 or grid_y < 0:
                    continue
                
                best_anchor = self._find_best_anchor(width, height, anchors)
                
                if target[best_anchor, grid_y, grid_x, 4] > 0.5:
                    continue
                
                target[best_anchor, grid_y, grid_x, 0] = x_center
                target[best_anchor, grid_y, grid_x, 1] = y_center
                target[best_anchor, grid_y, grid_x, 2] = width
                target[best_anchor, grid_y, grid_x, 3] = height
                target[best_anchor, grid_y, grid_x, 4] = 1.0  # objectness score
                
                final_class = class_id if label == -1 else label
                if final_class >= self.n_classes:
                    target[best_anchor, grid_y, grid_x, 5 + final_class ] = 1.0  # objectness score
                
            
            targets.append(target)
            
        return targets
    
    def _find_best_anchor(
        self,
        gt_width: float,
        gt_height: float,
        anchors: list[tuple[int, int]],
    ) -> int:
        
        """
        Find the best anchor for a given ground truth box

        Args:
            gt_width (float): Width of the ground truth box
            gt_height (float): Height of the ground truth box
            anchors (list[tuple[int, int]]): List of anchor boxes

        Returns:
            int: Index of the best anchor
        """
        
        best_iou = 0
        best_anchor_idx = 0
        
        for i, (anchor_w, anchor_h) in enumerate(anchors):
            
            # Normalize anchor size
            anchor_w /= self.image_size
            anchor_h /= self.image_size
            
            interserction = min(gt_width, anchor_w) * min(gt_height, anchor_h)
            unioin = gt_width * gt_height + anchor_w * anchor_h - interserction
            iou = interserction / (unioin + 1e-10)
            
            if iou > best_iou:
                best_iou = iou
                best_anchor_idx = i
                
        return best_anchor_idx   