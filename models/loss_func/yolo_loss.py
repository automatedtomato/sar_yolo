import torch
import torch.nn as nn
from logging import getLogger
import torch.nn.functional as F

logger = getLogger(__name__)

def calc_iou(self, pred_box: torch.tensor, target_box: torch.tensor, format: str='xywh', epsilon: float=1e-6) -> torch.tensor:
        """
        Calculate IoU between predicted and target boxes.
        
        Args:
            pred_box (torch.tensor): Predicted box coordinates
            target_box (torch.tensor): Target box coordinates
            format (str, optional): Box format. Defaults to 'xywh'; also supports 'xyxy'
            epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """

        # Box coordinates of prediction and ground truth
        if format == 'xywh':
            pb_x1 = pred_box[..., 0:1] - pred_box[..., 2:3] / 2
            pb_y1 = pred_box[..., 1:2] - pred_box[..., 3:4] / 2
            pb_x2 = pred_box[..., 0:1] + pred_box[..., 2:3] / 2
            pb_y2 = pred_box[..., 1:2] + pred_box[..., 3:4] / 2

            tb_x1 = target_box[..., 0:1] - target_box[..., 2:3] / 2
            tb_y1 = target_box[..., 1:2] - target_box[..., 3:4] / 2
            tb_x2 = target_box[..., 0:1] + target_box[..., 2:3] / 2
            tb_y2 = target_box[..., 1:2] + target_box[..., 3:4] / 2
            
        elif format == 'xyxy':
            pb_x1 = pred_box[..., 0:1]
            pb_y1 = pred_box[..., 1:2]
            pb_x2 = pred_box[..., 2:3]
            pb_y2 = pred_box[..., 3:4]

            tb_x1 = target_box[..., 0:1]
            tb_y1 = target_box[..., 1:2]
            tb_x2 = target_box[..., 2:3]
            tb_y2 = target_box[..., 3:4]
            
        else:
            raise ValueError('format must be "xywh" or "xyxy"')
        
        
        # Coordinates of the intersection rectangle
        inter_rect_x1 = torch.max(pb_x1, tb_x1)
        inter_rect_y1 = torch.max(pb_y1, tb_y1)
        inter_rect_x2 = torch.min(pb_x2, tb_x2)
        inter_rect_y2 = torch.min(pb_y2, tb_y2)

        # Intersection area (minimum 0)
        intersection = (inter_rect_x2 - inter_rect_x1).clamp(0) * (inter_rect_y2 - inter_rect_y1).clamp(0)
        
        # Union area
        pred_area = abs((pb_x2 - pb_x1) * (pb_y2 - pb_y1))
        target_area = abs((tb_x2 - tb_x1) * (tb_y2 - tb_y1))
                
        union_area = pred_area + target_area - intersection + epsilon
        
        iou = intersection / union_area
        
        return iou
    
def batch_iou(self, pred_box: torch.tensor, target_box: torch.tensor, format: str='xywh') -> torch.tensor:
    """
    Calculate IoU in batch process
    
    Args:
        pred_box (torch.tensor): Predicted box coordinates
        target_box (torch.tensor): Target box coordinates
        format (str, optional): Box format. Defaults to 'xywh'; also supports 'xyxy'
    """
    return self.calc_iou(pred_box, target_box, format='xywh')
    
    
def yolo_loss(
    pred: list[torch.tensor],
    target: list[torch.tensor],
    lambda_coord: float=5,
    lambda_obj: float=1,
    lambda_noobj: float=0.5,
    obj_threshold: float=0.5,
    lambda_class: float=1.0,
    epsilon: float=1e-6
) -> dict[str, torch.tensor]:
    """
    Calculate YOLOv3 loss function for multi-scale predictions
    
    Args:
        pred (list[torch.tensor]): Predicted outputs - shape: (batch_size, num_anchors, w, h, (num_classes + 5))
        target (list[torch.tensor]): Target outputs
        lambda_coord (float, optional): Weight for coordinate loss. Defaults to 5.
        lambda_obj (float, optional): Weight for object loss. Defaults to 1.
        lambda_noobj (float, optional): Weight for non-object loss. Defaults to 0.5.
        obj_threshold (float, optional): Threshold for objectness. Defaults to 0.5.
        lambda_class (float, optional): Weight for class loss. Defaults to 1.0.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
    """
    
    total_coord_loss = 0
    total_noobj_loss = 0
    total_obj_loss = 0
    total_class_loss = 0
    
    scale_losses = []
    
    for scl_idx, (pred_scale, target_scale) in enumerate(zip(pred, target)):
        scale_loss = _yolo_loss_single(
            pred_scale,
            target_scale,
            lambda_coord,
            lambda_obj,
            lambda_noobj,
            obj_threshold,
            lambda_class,
            epsilon
        )
                
        total_coord_loss += scale_loss['coord_loss']
        total_noobj_loss += scale_loss['noobj_loss']
        total_obj_loss += scale_loss['obj_loss']
        total_class_loss += scale_loss['class_loss']
        
        scale_losses.append({
            f'scale_{scl_idx + 1}_coord_loss': {scale_loss["coord_loss"]}, 
            f'scale_{scl_idx + 1}_obj_loss': {scale_loss["obj_loss"]},
            f'scale_{scl_idx + 1}_noobj_loss': {scale_loss["noobj_loss"]},
            f'scale_{scl_idx + 1}_class_loss': {scale_loss["class_loss"]},
            f'scale_{scl_idx + 1}_total_loss': {scale_loss["total_loss"]}
        })
        
    total_loss = total_coord_loss + total_obj_loss + total_noobj_loss + total_class_loss
    
    result = {
        'coord_loss': total_coord_loss,
        'obj_loss': total_obj_loss,
        'noobj_loss': total_noobj_loss,
        'class_loss': total_class_loss,
        'total_loss': total_loss,
    }
    
    for scale_loss in scale_losses:
        result.update(scale_loss)
        
    return result
            
def _yolo_loss_single(
    pred: torch.tensor,
    target: torch.tensor,
    lambda_coord: float=5,
    lambda_obj: float=1,
    lambda_noobj: float=0.5,
    obj_threshold: float=0.5,
    lambda_class: float=1.0,
    epsilon: float=1e-6
    ) -> dict[str, torch.tensor]:
    
    """
    Calculate YOLOv3 loss function for single scale
    
    Args:
        pred (torch.tensor): Predicted outputs - shape: (batch_size, num_anchors, w, h, (num_classes + 5))
        target (torch.tensor): Target outputs
        lambda_coord (float, optional): Weight for coordinate loss. Defaults to 5.
        lambda_obj (float, optional): Weight for object loss. Defaults to 1.
        lambda_noobj (float, optional): Weight for non-object loss. Defaults to 0.5.
        lambda_class (float, optional): Weight for class loss. Defaults to 1.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        
    Returns:
        dict[str, torch.tensor]: Losses
    """
    
    total_coord_loss = 0
    total_noobj_loss = 0
    total_obj_loss = 0
    total_class_loss = 0
    
    _, num_anchors, _, _, _ = pred.shape
    
    for i in range(num_anchors):
        
        pred_anchor = pred[:, i, :, :, :]
        target_anchor = target[:, i, :, :, :]
        
        coord_diff = pred_anchor[..., 0:4] - target_anchor[..., 0:4]
        coord_loss = torch.sum(torch.square(coord_diff) * target_anchor[..., 4:5])
        
        obj_mask = target_anchor[..., 4:5] >= obj_threshold
        obj_pred = pred_anchor[..., 4:5]
        
        obj_loss = torch.sum(-1 * torch.log(torch.sigmoid(obj_pred) + epsilon) * obj_mask)
        
        noobj_mask = ~obj_mask
        
        noobj_loss = torch.sum(-torch.log(1 - torch.sigmoid(obj_pred) + epsilon) * noobj_mask)
        
        class_mask = obj_mask.expand_as(pred_anchor[..., 5:])
        class_loss = F.binary_cross_entropy_with_logits(
            pred_anchor[..., 5:], target_anchor[..., 5:],
            reduction='none')
        class_loss = torch.sum(class_loss * class_mask)
            
        total_coord_loss += coord_loss
        total_noobj_loss += noobj_loss
        total_obj_loss += obj_loss
        total_class_loss += class_loss
        
    total_loss = (
        lambda_coord * total_coord_loss + 
        lambda_obj * total_obj_loss + 
        lambda_noobj * total_noobj_loss +
        lambda_class * total_class_loss
    )
        
    return {
        'total_loss': total_loss,
        'coord_loss': total_coord_loss,
        'obj_loss': total_obj_loss,
        'noobj_loss': total_noobj_loss,
        'class_loss': total_class_loss
    }