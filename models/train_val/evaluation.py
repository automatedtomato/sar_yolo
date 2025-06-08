import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from logging import getLogger
import torch.nn.functional as F

logger = getLogger(__name__)


class YOLOv3Evaluator:
    """
    YOLOv3 evaluation metrics calculator
    
    This class works like a quality inspector in a factory - it examines 
    the model's predictions and measures how well they match the ground truth
    using various metrics like mAP, Precision, Recall, F1, and IoU distribution.
    """
    
    def __init__(
        self,
        n_classes: int,
        iou_thresholds: List[float] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        """
        Initialize YOLOv3 evaluator
        
        Args:
            n_classes (int): Number of classes
            iou_thresholds (List[float], optional): IoU thresholds for mAP calculation
            confidence_threshold (float): Confidence threshold for predictions
            nms_threshold (float): NMS threshold for post-processing
        """
        self.n_classes = n_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Storage for predictions and targets across batches
        self.all_predictions = []
        self.all_targets = []
        
    def reset(self):
        """Reset accumulated predictions and targets"""
        self.all_predictions = []
        self.all_targets = []
        
    def update(
        self, 
        predictions: List[torch.Tensor], 
        targets: List[torch.Tensor]
    ):
        """
        Update evaluator with batch predictions and targets
        
        Args:
            predictions (List[torch.Tensor]): Model predictions for 3 scales
            targets (List[torch.Tensor]): Ground truth targets for 3 scales
        """
        # Convert multi-scale predictions to single prediction format
        batch_predictions = self._process_predictions(predictions)
        batch_targets = self._process_targets(targets)
        
        self.all_predictions.extend(batch_predictions)
        self.all_targets.extend(batch_targets)
        
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Returns:
            Dict[str, float]: Dictionary containing all metrics
        """
        if not self.all_predictions or not self.all_targets:
            logger.warning("No predictions or targets to evaluate")
            return {}
            
        metrics = {}
        
        # Compute mAP
        map_metrics = self._compute_map()
        metrics.update(map_metrics)
        
        # Compute Precision, Recall, F1
        pr_metrics = self._compute_precision_recall_f1()
        metrics.update(pr_metrics)
        
        # Compute IoU distribution
        iou_metrics = self._compute_iou_distribution()
        metrics.update(iou_metrics)
        
        return metrics
        
    def _process_predictions(
        self, 
        predictions: List[torch.Tensor]
    ) -> List[List[Dict[str, float]]]:
        """
        Process multi-scale predictions into NMS-filtered detection format
        
        Args:
            predictions (List[torch.Tensor]): Raw model predictions for 3 scales
            
        Returns:
            List[List[Dict]]: Processed predictions for each image in batch
        """
        batch_size = predictions[0].shape[0]
        grid_sizes = [13, 26, 52]
        anchors = [
            [[36.0, 24.0], [29.0, 50.0], [55.0, 30.0]],
            [[11.0, 22.0], [25.0, 17.0], [18.0, 33.0]],
            [[4.0, 5.0], [7.0, 12.0], [15.0, 11.0]]
        ]
        
        batch_predictions = []
        
        for batch_idx in range(batch_size):
            image_detections = []
            
            # Process each scale
            for scale_idx, (pred, grid_size, scale_anchors) in enumerate(
                zip(predictions, grid_sizes, anchors)
            ):
                scale_pred = pred[batch_idx]  # [3, grid_size, grid_size, n_classes+5]
                
                # Apply sigmoid to objectness and class predictions
                scale_pred[..., 4] = torch.sigmoid(scale_pred[..., 4])  # objectness
                scale_pred[..., 5:] = torch.sigmoid(scale_pred[..., 5:])  # classes
                
                # Filter by confidence threshold
                obj_mask = scale_pred[..., 4] > self.confidence_threshold
                
                for anchor_idx in range(3):
                    for y in range(grid_size):
                        for x in range(grid_size):
                            if obj_mask[anchor_idx, y, x]:
                                detection = self._decode_detection(
                                    scale_pred[anchor_idx, y, x],
                                    x, y, grid_size, scale_anchors[anchor_idx]
                                )
                                if detection:
                                    image_detections.append(detection)
            
            # Apply NMS
            filtered_detections = self._apply_nms(image_detections)
            batch_predictions.append(filtered_detections)
            
        return batch_predictions
    
    def _decode_detection(
        self, 
        pred: torch.Tensor, 
        grid_x: int, 
        grid_y: int, 
        grid_size: int, 
        anchor: List[float]
    ) -> Dict[str, float]:
        """
        Decode single detection from grid cell
        
        Args:
            pred (torch.Tensor): Prediction tensor for single anchor
            grid_x, grid_y (int): Grid cell coordinates
            grid_size (int): Size of the grid
            anchor (List[float]): Anchor box dimensions
            
        Returns:
            Dict[str, float]: Decoded detection
        """
        # Decode box coordinates
        x_center = (torch.sigmoid(pred[0]) + grid_x) / grid_size
        y_center = (torch.sigmoid(pred[1]) + grid_y) / grid_size
        width = torch.exp(pred[2]) * anchor[0] / 416.0  # Normalize by input size
        height = torch.exp(pred[3]) * anchor[1] / 416.0
        
        # Convert to xyxy format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        objectness = pred[4].item()
        class_probs = pred[5:]
        class_id = torch.argmax(class_probs).item()
        confidence = objectness * class_probs[class_id].item()
        
        return {
            'x1': x1.item(), 'y1': y1.item(), 'x2': x2.item(), 'y2': y2.item(),
            'confidence': confidence, 'class_id': class_id
        }
    
    def _apply_nms(
        self, 
        detections: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Apply Non-Maximum Suppression to detections
        
        Args:
            detections (List[Dict]): List of detections
            
        Returns:
            List[Dict]: Filtered detections after NMS
        """
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._calculate_iou_xyxy(current, det) < self.nms_threshold
                or current['class_id'] != det['class_id']
            ]
            
        return keep
    
    def _process_targets(
        self, 
        targets: List[torch.Tensor]
    ) -> List[List[Dict[str, float]]]:
        """
        Process multi-scale targets into detection format
        
        Args:
            targets (List[torch.Tensor]): Ground truth targets for 3 scales
            
        Returns:
            List[List[Dict]]: Processed targets for each image in batch
        """
        batch_size = targets[0].shape[0]
        batch_targets = []
        
        for batch_idx in range(batch_size):
            image_targets = []
            
            # Process each scale to extract ground truth boxes
            for scale_idx, target in enumerate(targets):
                scale_target = target[batch_idx]  # [3, grid_size, grid_size, n_classes+5]
                grid_size = scale_target.shape[1]
                
                for anchor_idx in range(3):
                    for y in range(grid_size):
                        for x in range(grid_size):
                            if scale_target[anchor_idx, y, x, 4] > 0.5:  # objectness > 0.5
                                # Extract ground truth box
                                x_center = scale_target[anchor_idx, y, x, 0].item()
                                y_center = scale_target[anchor_idx, y, x, 1].item()
                                width = scale_target[anchor_idx, y, x, 2].item()
                                height = scale_target[anchor_idx, y, x, 3].item()
                                
                                # Convert to xyxy format
                                x1 = x_center - width / 2
                                y1 = y_center - height / 2
                                x2 = x_center + width / 2
                                y2 = y_center + height / 2
                                
                                # Find class
                                class_probs = scale_target[anchor_idx, y, x, 5:]
                                class_id = torch.argmax(class_probs).item()
                                
                                image_targets.append({
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'class_id': class_id
                                })
            
            batch_targets.append(image_targets)
            
        return batch_targets
    
    def _calculate_iou_xyxy(
        self, 
        box1: Dict[str, float], 
        box2: Dict[str, float]
    ) -> float:
        """
        Calculate IoU between two boxes in xyxy format
        
        Args:
            box1, box2 (Dict): Boxes with x1, y1, x2, y2 keys
            
        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _compute_map(self) -> Dict[str, float]:
        """
        Compute mean Average Precision (mAP)
        
        Returns:
            Dict[str, float]: mAP metrics
        """
        aps = []
        
        for class_id in range(self.n_classes):
            class_aps = []
            
            for iou_thresh in self.iou_thresholds:
                ap = self._compute_ap_for_class(class_id, iou_thresh)
                class_aps.append(ap)
                
            aps.append(np.mean(class_aps))
            
        return {
            'mAP': np.mean(aps),
            'mAP@0.5': self._compute_ap_for_class(-1, 0.5),  # All classes
            'mAP@0.75': self._compute_ap_for_class(-1, 0.75),
        }
    
    def _compute_ap_for_class(self, class_id: int, iou_threshold: float) -> float:
        """
        Compute Average Precision for a specific class and IoU threshold
        
        Args:
            class_id (int): Class ID (-1 for all classes)
            iou_threshold (float): IoU threshold
            
        Returns:
            float: Average Precision
        """
        # Collect all predictions and targets for the class
        pred_boxes = []
        gt_boxes = []
        
        for img_idx, (preds, targets) in enumerate(zip(self.all_predictions, self.all_targets)):
            # Filter predictions by class
            if class_id == -1:  # All classes
                pred_boxes.extend([(pred, img_idx) for pred in preds])
                gt_boxes.extend([(target, img_idx) for target in targets])
            else:
                pred_boxes.extend([(pred, img_idx) for pred in preds if pred['class_id'] == class_id])
                gt_boxes.extend([(target, img_idx) for target in targets if target['class_id'] == class_id])
        
        if not pred_boxes or not gt_boxes:
            return 0.0
            
        # Sort predictions by confidence
        pred_boxes.sort(key=lambda x: x[0]['confidence'], reverse=True)
        
        # Calculate precision and recall
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        num_gt = len(gt_boxes)
        
        gt_matched = {}  # Track which ground truth boxes have been matched
        
        for i, (pred, img_idx) in enumerate(pred_boxes):
            # Find best matching ground truth box
            best_iou = 0
            best_gt_idx = -1
            
            for j, (gt, gt_img_idx) in enumerate(gt_boxes):
                if img_idx != gt_img_idx:
                    continue
                    
                iou = self._calculate_iou_xyxy(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    
            # Check if prediction matches ground truth
            if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
                
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = precisions[recalls >= t]
            if len(prec_at_recall) > 0:
                ap += np.max(prec_at_recall) / 11
                
        return ap
    
    def _compute_precision_recall_f1(self) -> Dict[str, float]:
        """
        Compute overall Precision, Recall, and F1 score
        
        Returns:
            Dict[str, float]: Precision, Recall, F1 metrics
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for preds, targets in zip(self.all_predictions, self.all_targets):
            # Match predictions to targets
            matched_gt = set()
            
            for pred in preds:
                best_iou = 0
                best_gt_idx = -1
                
                for i, target in enumerate(targets):
                    if pred['class_id'] == target['class_id']:
                        iou = self._calculate_iou_xyxy(pred, target)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                            
                if best_iou >= 0.5 and best_gt_idx not in matched_gt:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
                    
            # Count false negatives (unmatched ground truth boxes)
            total_fn += len(targets) - len(matched_gt)
            
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    def _compute_iou_distribution(self) -> Dict[str, float]:
        """
        Compute IoU distribution statistics
        
        Returns:
            Dict[str, float]: IoU distribution metrics
        """
        ious = []
        
        for preds, targets in zip(self.all_predictions, self.all_targets):
            for pred in preds:
                best_iou = 0
                
                for target in targets:
                    if pred['class_id'] == target['class_id']:
                        iou = self._calculate_iou_xyxy(pred, target)
                        best_iou = max(best_iou, iou)
                        
                ious.append(best_iou)
                
        if not ious:
            return {'IoU_mean': 0.0, 'IoU_std': 0.0}
            
        ious = np.array(ious)
        
        return {
            'IoU_mean': np.mean(ious),
            'IoU_std': np.std(ious),
            'IoU_median': np.median(ious),
            'IoU_min': np.min(ious),
            'IoU_max': np.max(ious)
        }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_classes: int
) -> Dict[str, float]:
    """
    Evaluate YOLOv3 model on a dataset
    
    Args:
        model (torch.nn.Module): YOLOv3 model
        dataloader: DataLoader for evaluation dataset
        device: Device to run evaluation on
        n_classes (int): Number of classes
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    model.eval()
    evaluator = YOLOv3Evaluator(n_classes=n_classes)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            
            # Get model predictions
            predictions = model(images)
            
            # Update evaluator
            evaluator.update(predictions, targets)
            
            if batch_idx % 50 == 0:
                print(f"Evaluated batch {batch_idx}/{len(dataloader)}")
    
    # Compute final metrics
    metrics = evaluator.compute_metrics()
    
    return metrics

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import cv2
from PIL import Image
from logging import getLogger

logger = getLogger(__name__)


class YOLOv3Visualizer:
    """
    YOLOv3 prediction and ground truth visualization class
    
    This class works like an artist's canvas - it takes the model's predictions
    and paints them onto images so we can see what the model is "seeing"
    and how well it's performing.
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        colors: List[Tuple[int, int, int]] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        """
        Initialize YOLOv3 visualizer
        
        Args:
            class_names (List[str], optional): List of class names
            colors (List[Tuple], optional): Colors for each class
            confidence_threshold (float): Confidence threshold for visualization
            nms_threshold (float): NMS threshold for post-processing
        """
        self.class_names = class_names or ['Object']
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Generate colors if not provided
        if colors is None:
            self.colors = self._generate_colors(len(self.class_names))
        else:
            self.colors = colors
            
    def _generate_colors(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """
        Generate distinct colors for classes
        
        Args:
            n_colors (int): Number of colors to generate
            
        Returns:
            List[Tuple[int, int, int]]: RGB color tuples
        """
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            # Convert HSV to RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    def visualize_predictions(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        targets: List[torch.Tensor] = None,
        device: torch.device = None,
        save_path: str = None,
        show_confidence: bool = True,
        show_ground_truth: bool = True,
        max_images: int = 4
    ) -> None:
        """
        Visualize model predictions on a batch of images
        
        Args:
            model: YOLOv3 model
            images (torch.Tensor): Batch of input images [B, C, H, W]
            targets (List[torch.Tensor], optional): Ground truth targets
            device: Device to run inference on
            save_path (str, optional): Path to save visualization
            show_confidence (bool): Whether to show confidence scores
            show_ground_truth (bool): Whether to show ground truth boxes
            max_images (int): Maximum number of images to visualize
        """
        if device is None:
            device = next(model.parameters()).device
            
        model.eval()
        
        with torch.no_grad():
            images = images.to(device)
            predictions = model(images)
            
        # Limit number of images to visualize
        batch_size = min(images.shape[0], max_images)
        
        # Create subplot grid
        cols = 2 if batch_size <= 2 else 3
        rows = (batch_size + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if batch_size == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(batch_size):
            # Process single image
            image = self._denormalize_image(images[i])
            pred_boxes = self._process_single_prediction(predictions, i)
            gt_boxes = self._process_single_target(targets, i) if targets else []
            
            # Create visualization
            ax = axes[i]
            self._plot_single_image(
                ax, image, pred_boxes, gt_boxes,
                show_confidence, show_ground_truth
            )
            
        # Hide extra subplots
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
    def _denormalize_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor for visualization
        
        Args:
            image (torch.Tensor): Normalized image tensor [C, H, W]
            
        Returns:
            np.ndarray: Denormalized image array [H, W, C]
        """
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Denormalize
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        return image
    
    def _process_single_prediction(
        self,
        predictions: List[torch.Tensor],
        image_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Process predictions for a single image
        
        Args:
            predictions (List[torch.Tensor]): Multi-scale predictions
            image_idx (int): Index of image in batch
            
        Returns:
            List[Dict]: Processed detections
        """
        grid_sizes = [13, 26, 52]
        anchors = [
            [[36.0, 24.0], [29.0, 50.0], [55.0, 30.0]],
            [[11.0, 22.0], [25.0, 17.0], [18.0, 33.0]],
            [[4.0, 5.0], [7.0, 12.0], [15.0, 11.0]]
        ]
        
        detections = []
        
        # Process each scale
        for scale_idx, (pred, grid_size, scale_anchors) in enumerate(
            zip(predictions, grid_sizes, anchors)
        ):
            scale_pred = pred[image_idx]  # [3, grid_size, grid_size, n_classes+5]
            
            # Apply sigmoid to objectness and class predictions
            scale_pred = scale_pred.clone()
            scale_pred[..., 4] = torch.sigmoid(scale_pred[..., 4])  # objectness
            scale_pred[..., 5:] = torch.sigmoid(scale_pred[..., 5:])  # classes
            
            # Filter by confidence threshold
            obj_mask = scale_pred[..., 4] > self.confidence_threshold
            
            for anchor_idx in range(3):
                for y in range(grid_size):
                    for x in range(grid_size):
                        if obj_mask[anchor_idx, y, x]:
                            detection = self._decode_detection(
                                scale_pred[anchor_idx, y, x],
                                x, y, grid_size, scale_anchors[anchor_idx]
                            )
                            if detection:
                                detections.append(detection)
        
        # Apply NMS
        filtered_detections = self._apply_nms(detections)
        return filtered_detections
    
    def _decode_detection(
        self,
        pred: torch.Tensor,
        grid_x: int,
        grid_y: int,
        grid_size: int,
        anchor: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Decode single detection from grid cell
        
        Args:
            pred (torch.Tensor): Prediction tensor for single anchor
            grid_x, grid_y (int): Grid cell coordinates
            grid_size (int): Size of the grid
            anchor (List[float]): Anchor box dimensions
            
        Returns:
            Optional[Dict]: Decoded detection or None
        """
        # Decode box coordinates
        x_center = (torch.sigmoid(pred[0]) + grid_x) / grid_size
        y_center = (torch.sigmoid(pred[1]) + grid_y) / grid_size
        width = torch.exp(pred[2]) * anchor[0] / 416.0  # Normalize by input size
        height = torch.exp(pred[3]) * anchor[1] / 416.0
        
        # Convert to xyxy format (normalized coordinates)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Clamp coordinates to [0, 1]
        x1 = max(0, min(1, x1.item()))
        y1 = max(0, min(1, y1.item()))
        x2 = max(0, min(1, x2.item()))
        y2 = max(0, min(1, y2.item()))
        
        objectness = pred[4].item()
        class_probs = pred[5:]
        class_id = torch.argmax(class_probs).item()
        class_prob = class_probs[class_id].item()
        confidence = objectness * class_prob
        
        return {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'confidence': confidence, 'class_id': class_id
        }
    
    def _apply_nms(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression to detections
        
        Args:
            detections (List[Dict]): List of detections
            
        Returns:
            List[Dict]: Filtered detections after NMS
        """
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections of the same class
            detections = [
                det for det in detections
                if self._calculate_iou_xyxy(current, det) < self.nms_threshold
                or current['class_id'] != det['class_id']
            ]
            
        return keep
    
    def _calculate_iou_xyxy(
        self,
        box1: Dict[str, float],
        box2: Dict[str, float]
    ) -> float:
        """
        Calculate IoU between two boxes in xyxy format
        
        Args:
            box1, box2 (Dict): Boxes with x1, y1, x2, y2 keys
            
        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _process_single_target(
        self,
        targets: List[torch.Tensor],
        image_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Process ground truth targets for a single image
        
        Args:
            targets (List[torch.Tensor]): Multi-scale targets
            image_idx (int): Index of image in batch
            
        Returns:
            List[Dict]: Ground truth boxes
        """
        if not targets:
            return []
            
        gt_boxes = []
        
        # Process each scale to extract ground truth boxes
        for scale_idx, target in enumerate(targets):
            scale_target = target[image_idx]  # [3, grid_size, grid_size, n_classes+5]
            grid_size = scale_target.shape[1]
            
            for anchor_idx in range(3):
                for y in range(grid_size):
                    for x in range(grid_size):
                        if scale_target[anchor_idx, y, x, 4] > 0.5:  # objectness > 0.5
                            # Extract ground truth box (already in normalized coordinates)
                            x_center = scale_target[anchor_idx, y, x, 0].item()
                            y_center = scale_target[anchor_idx, y, x, 1].item()
                            width = scale_target[anchor_idx, y, x, 2].item()
                            height = scale_target[anchor_idx, y, x, 3].item()
                            
                            # Convert to xyxy format
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            # Find class
                            class_probs = scale_target[anchor_idx, y, x, 5:]
                            class_id = torch.argmax(class_probs).item()
                            
                            gt_boxes.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'class_id': class_id
                            })
        
        return gt_boxes
    
    def _plot_single_image(
        self,
        ax: plt.Axes,
        image: np.ndarray,
        pred_boxes: List[Dict[str, Any]],
        gt_boxes: List[Dict[str, Any]],
        show_confidence: bool,
        show_ground_truth: bool
    ) -> None:
        """
        Plot single image with predictions and ground truth
        
        Args:
            ax: Matplotlib axes
            image: Image array
            pred_boxes: Predicted bounding boxes
            gt_boxes: Ground truth bounding boxes
            show_confidence: Whether to show confidence scores
            show_ground_truth: Whether to show ground truth boxes
        """
        ax.imshow(image)
        ax.set_title(f'Predictions: {len(pred_boxes)}, GT: {len(gt_boxes)}')
        ax.axis('off')
        
        h, w = image.shape[:2]
        
        # Draw ground truth boxes (green, dashed)
        if show_ground_truth and gt_boxes:
            for gt_box in gt_boxes:
                x1 = gt_box['x1'] * w
                y1 = gt_box['y1'] * h
                x2 = gt_box['x2'] * w
                y2 = gt_box['y2'] * h
                
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='green', facecolor='none',
                    linestyle='--', alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add class label for ground truth
                class_name = self.class_names[gt_box['class_id']] if gt_box['class_id'] < len(self.class_names) else f"Class{gt_box['class_id']}"
                ax.text(
                    x1, y1 - 5, f'GT: {class_name}',
                    color='green', fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
                )
        
        # Draw prediction boxes (colored by class)
        for pred_box in pred_boxes:
            x1 = pred_box['x1'] * w
            y1 = pred_box['y1'] * h
            x2 = pred_box['x2'] * w
            y2 = pred_box['y2'] * h
            
            # Get color for this class
            color_idx = pred_box['class_id'] % len(self.colors)
            color = [c/255.0 for c in self.colors[color_idx]]  # Convert to matplotlib format
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add class label and confidence
            class_name = self.class_names[pred_box['class_id']] if pred_box['class_id'] < len(self.class_names) else f"Class{pred_box['class_id']}"
            
            if show_confidence:
                label = f'{class_name}: {pred_box["confidence"]:.2f}'
            else:
                label = class_name
                
            ax.text(
                x1, y1 - 5, label,
                color=color, fontsize=8, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )
    
    def visualize_confidence_distribution(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = None,
        save_path: str = None,
        num_batches: int = 10
    ) -> None:
        """
        Visualize confidence score distribution across dataset
        
        Args:
            model: YOLOv3 model
            dataloader: DataLoader for evaluation dataset
            device: Device to run inference on
            save_path: Path to save visualization
            num_batches: Number of batches to process
        """
        if device is None:
            device = next(model.parameters()).device
            
        model.eval()
        
        all_confidences = []
        all_objectness = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                    
                images = images.to(device)
                predictions = model(images)
                
                # Extract confidence scores from all predictions
                for pred in predictions:
                    # Apply sigmoid to objectness
                    objectness = torch.sigmoid(pred[..., 4])
                    class_probs = torch.sigmoid(pred[..., 5:])
                    
                    # Get max class probability for each prediction
                    max_class_probs = torch.max(class_probs, dim=-1)[0]
                    confidences = objectness * max_class_probs
                    
                    # Filter by confidence threshold
                    valid_mask = objectness > self.confidence_threshold
                    
                    all_confidences.extend(confidences[valid_mask].cpu().numpy().flatten())
                    all_objectness.extend(objectness[valid_mask].cpu().numpy().flatten())
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence distribution
        ax1.hist(all_confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.axvline(self.confidence_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.confidence_threshold}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Objectness distribution
        ax2.hist(all_objectness, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Objectness Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Objectness Score Distribution')
        ax2.axvline(self.confidence_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.confidence_threshold}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confidence distribution saved to {save_path}")
        else:
            plt.show()
    
    def create_detection_video(
        self,
        model: torch.nn.Module,
        video_path: str,
        output_path: str,
        device: torch.device = None,
        frame_skip: int = 1,
        show_confidence: bool = True
    ) -> None:
        """
        Create video with detection visualization
        
        Args:
            model: YOLOv3 model
            video_path: Path to input video
            output_path: Path to save output video
            device: Device to run inference on
            frame_skip: Process every nth frame
            show_confidence: Whether to show confidence scores
        """
        if device is None:
            device = next(model.parameters()).device
            
        model.eval()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        # Define preprocessing transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % frame_skip != 0:
                    out.write(frame)
                    continue
                
                # Preprocess frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    predictions = model(input_tensor)
                    
                # Process predictions
                pred_boxes = self._process_single_prediction(predictions, 0)
                
                # Draw detections on frame
                for pred_box in pred_boxes:
                    x1 = int(pred_box['x1'] * width)
                    y1 = int(pred_box['y1'] * height)
                    x2 = int(pred_box['x2'] * width)
                    y2 = int(pred_box['y2'] * height)
                    
                    # Get color for this class
                    color_idx = pred_box['class_id'] % len(self.colors)
                    color = self.colors[color_idx]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    class_name = self.class_names[pred_box['class_id']] if pred_box['class_id'] < len(self.class_names) else f"Class{pred_box['class_id']}"
                    
                    if show_confidence:
                        label = f'{class_name}: {pred_box["confidence"]:.2f}'
                    else:
                        label = class_name
                    
                    # Calculate text size and position
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw text background
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - text_height - baseline),
                        (x1 + text_width, y1),
                        color, -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame, label, (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                
                out.write(frame)
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
                    
        finally:
            cap.release()
            out.release()
            logger.info(f"Detection video saved to {output_path}")


def visualize_training_predictions(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    class_names: List[str],
    device: torch.device,
    save_dir: str = './figures/',
    num_samples: int = 4
) -> None:
    """
    Visualize predictions on training and validation samples
    
    Args:
        model: YOLOv3 model
        train_loader: Training data loader
        val_loader: Validation data loader
        class_names: List of class names
        device: Device to run inference on
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = YOLOv3Visualizer(
        class_names=class_names,
        confidence_threshold=0.3
    )
    
    # Visualize training samples
    train_images, train_targets = next(iter(train_loader))
    visualizer.visualize_predictions(
        model=model,
        images=train_images[:num_samples],
        targets=[target[:num_samples] for target in train_targets],
        device=device,
        save_path=os.path.join(save_dir, 'train_predictions.png'),
        max_images=num_samples
    )
    
    # Visualize validation samples
    val_images, val_targets = next(iter(val_loader))
    visualizer.visualize_predictions(
        model=model,
        images=val_images[:num_samples],
        targets=[target[:num_samples] for target in val_targets],
        device=device,
        save_path=os.path.join(save_dir, 'val_predictions.png'),
        max_images=num_samples
    )
    
    # Visualize confidence distribution
    visualizer.visualize_confidence_distribution(
        model=model,
        dataloader=val_loader,
        device=device,
        save_path=os.path.join(save_dir, 'confidence_distribution.png'),
        num_batches=5
    )
    
    logger.info(f"Visualizations saved to {save_dir}")
    
    
def learning_curve(train_history: dict[str, list[float]], val_history: dict[str, list[float]], show_fig: bool=True, save_path: str=None) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for i, key in enumerate(train_history.keys()):
        plt.subplot(2, 3, i + 1)
        plt.plot(train_history[key], label=f'Training {key}')
        plt.plot(val_history[key], label=f'Validation {key}')
        plt.title(f'{key} Over Epochs')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Metric', fontsize=10)
    plt.tight_layout()
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if show_fig:
        plt.show()
    logger.info(f"Learning curve saved to {save_path}")