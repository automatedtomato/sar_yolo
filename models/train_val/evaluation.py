import torch
from typing import List, Any, Optional
from logging import getLogger
from torchvision.ops import nms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from models.loss_func.yolo_loss import yolo_loss

from utils.load_config import load_config

logger = getLogger(__name__)


class YOLOv3Evaluator:
    """
    YOLOv3 evaluation toolkit:
        - IoU calculation
        - Coordinates transformation
        - NMS implementation
        - Postprocess results
        - mAP calculation
        - Precision and Recall calculation
        - Visualize results
        - CSV output

    Args:
        # ===== One of config or config_path must be provided =====
        config (dict[str, Any]): Config dictionary
        config_path (str): Path to config file

        device (torch.device): Device; 'cuda' or 'cpu'
    """

    def __init__(
        self,
        device: torch.device,
        config: Optional[dict[str, Any]] = None,
        config_path: str = None,
    ):

        # ===== One of config or config_path must be provided =====
        if config is not None and config_path is None:
            self.config = config

        elif config is None and config_path is not None:
            self.config = load_config(config_path)

        elif config is not None and config_path is not None:
            logger.warining(
                "Both config and config_path are provided. Using config_path."
            )
            self.config = load_config(config_path)
        else:
            raise ValueError("Either config or config_path must be provided")

        self.device = device
        self.anchors = torch.tensor(config["model"]["anchors"]).to(device)
        self.grid_sizes = config["model"]["grid_sizes"]
        self.img_size = config["model"]["input_size"]

        print(f"\n{__name__}:: YOLOv3 evaluator initialized.")

    # ========== Fundamental utilitiies ==========

    def xywh2xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert xywh to xyxy

        Args:
            boxes (torch.Tensor): Bounding boxes (in xywh format) (N, 4)
        Returns:
            boxes (torch.Tensor): Bounding boxes (in xyxy format) (N, 4)
        """

        x_center, y_center, width, height = boxes.unbind(
            -1
        )  # retruns tuple x_center, y_center, width, height
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def xyxy2xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert xyxy to xywh

        Args:
            boxes (torch.Tensor): Bounding boxes (in xyxy format) (N, 4)
        Returns:
            boxes (torch.Tensor): Bounding boxes (in xywh format) (N, 4)
        """

        x1, y1, x2, y2 = boxes.unbind(-1)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        return torch.stack([x_center, y_center, width, height], dim=-1)

    def calculate_iou(
        self, boxes1: torch.Tensor, boxes2: torch.Tensor, epsilon: float = 1e-10
    ) -> torch.Tensor:
        """
        IoU calculation

        Args:
            boxes1, boxes2 (torch.Tensor): Bounding boxes (in xyxy format)
        Returns:
            iou: (M, M) IoU matrix
        """

        # Calculate areas of boxes
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Left top and right bottom
        left_top = torch.max(boxes1[:, None:2], boxes2[:, :2])
        right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

        # Width and height
        width_height = (right_bottom - left_top).clamp(min=0)

        # Intersection
        intersection = width_height[:, :, 0] * width_height[:, :, 1]

        # Union
        union = area1[:, None] + area2 - intersection

        # IoU (prevent division by zero)
        iou = intersection / union + epsilon

        return iou

    def _decode_ground_truth_grid(
        self, targets_grid: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Decodes ground truth targets from YOLO grid format to a list of raw bounding boxes per image.
        Args:
            targets_grid (List[torch.Tensor]): List of ground truth tensors in YOLO grid format,
                                               e.g., [tensor(B, A, G, G, 5+C), tensor(B, A, G', G', 5+C), ...]
        Returns:
            List[torch.Tensor]: List of tensors, where each tensor corresponds to an image
                                and contains its raw ground truth boxes in (N_objects, 5) format:
                                [class_id, x_center, y_center, width, height]
        """
        decoded_targets_per_image = [
            [] for _ in range(targets_grid[0].shape[0])
        ]  # Initialize list for each image in batch

        for scale_idx, grid_tensor in enumerate(targets_grid):
            # grid_tensor shape: (batch_size, num_anchors, grid_h, grid_w, 5 + num_classes)
            batch_size, num_anchors, grid_h, grid_w, _ = grid_tensor.shape

            # Create grid coordinates for this scale
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_h), torch.arange(grid_w), indexing="ij"
            )
            grid_x = grid_x.to(self.device).float()
            grid_y = grid_y.to(self.device).float()

            for b in range(batch_size):
                # For each image in the batch, iterate through anchors and grid cells
                # Find cells with actual objects (where confidence/objectness is 1.0)
                # Assuming the last element in the 5+C dimension is objectness for GT
                # Or, more robustly, look for class_id != 0 or any non-zero bbox values

                # Reshape to (num_anchors * grid_h * grid_w, 5 + num_classes)
                flat_grid = grid_tensor[b].view(-1, grid_tensor.shape[-1])

                # Filter out "empty" cells (where there's no object, typically indicated by all zeros or a specific value)
                # A simple check: if any of the bbox coords or class_id is non-zero
                # Assuming class_id is the first element, x, y, w, h follow (index 1 to 4)
                # And 1.0 indicates presence of an object at the 5th element (objectness)
                # Let's use the objectness score (index 4 in the first 5 elements for gt_bbox)

                # Find rows where the 5th element (objectness/confidence) is not zero
                # For GT, it's usually 1.0 for valid boxes and 0.0 otherwise
                object_mask = (
                    flat_grid[:, 4] > 0
                ).bool()  # Assuming 5th element (index 4) is objectness/confidence

                if object_mask.sum() == 0:
                    continue  # No objects in this image for this scale

                # Extract relevant data for actual objects
                valid_boxes_raw = flat_grid[
                    object_mask
                ]  # (N_objects_this_scale, 5 + num_classes)

                # Now, we need to convert these grid-relative coordinates back to image-relative (0-1)
                # For GT, x, y are already relative to the grid cell and w, h are relative to the anchor.
                # The format is [class_id, x_grid, y_grid, w_grid, h_grid, objectness_score, ...]
                # Where x_grid, y_grid are relative to the cell, w_grid, h_grid are relative to anchor/grid cell size.

                # This is tricky because your loss function handles this.
                # The simplest assumption is that `gt` in `yolo_loss` are already in [class_id, x_norm, y_norm, w_norm, h_norm]
                # If your SaRDataset already converts annotations to a YOLO-specific grid format,
                # the 5 elements for each cell usually contain:
                # [object_present_score, x_offset, y_offset, w_scale, h_scale, class_probabilities]
                # or [class_id, x_offset, y_offset, w_scale, h_scale] if objectness is implied or handled separately.

                # Let's assume the 5 elements in the targets_grid[b, anchor_idx, grid_y, grid_x, :] are:
                # [x_center_normalized_in_image, y_center_normalized_in_image, w_normalized_in_image, h_normalized_in_image, class_id]
                # This is the most common raw target format.
                # BUT, your print output shows [..., ..., ..., ..., 1.0000] as the 5th element (index 4)
                # and class_id at index 0, and actual coordinates at 1-4.
                # E.g., [0.1037, 0.8873, 0.1743, 0.0533, 1.0000]

                # Let's re-interpret based on your print output:
                # The 5th element (index 4) is the objectness score (1.0 if object, 0.0 if not)
                # The 1st element (index 0) is the class_id
                # The 2nd-5th elements (indices 1-4) are x_center, y_center, width, height (normalized to image size)

                # So, filter by the objectness score (index 4)
                object_mask = (
                    flat_grid[:, 4] == 1.0
                ).bool()  # Only take cells that actually contain an object

                # Extract class_id and bounding box coordinates for valid objects
                # Expected format: [class_id, x_center, y_center, width, height]

                # If targets are structured as [class_id, x_norm, y_norm, w_norm, h_norm] in the first 5 elements for object-containing cells:
                # Then valid_boxes_info will be (N_objects, 5) with actual ground truth bounding boxes.
                if object_mask.sum() > 0:
                    valid_objects_data = flat_grid[object_mask][
                        :, :5
                    ]  # Take first 5 elements: class_id, x, y, w, h
                    decoded_targets_per_image[b].append(valid_objects_data)

        final_decoded_targets = []
        for targets_list_for_image in decoded_targets_per_image:
            if targets_list_for_image:
                final_decoded_targets.append(torch.cat(targets_list_for_image, dim=0))
            else:
                final_decoded_targets.append(
                    torch.empty(0, 5).to(self.device)
                )  # Ensure empty tensors have correct shape and device

        return final_decoded_targets

    # ========== Postprocess results ==========

    def postprocess_raw_outputs(
        self,
        raw_outputs: List[torch.Tensor],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Transform raw outputs to postprocessed results

        Args:
            raw_outputs (list[torch.Tensor]): Raw outputs [(batch_size, num_anchors, w, h, (num_classes + 5))]
            conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.
            nms_threshold (float, optional): NMS threshold. Defaults to 0.4.
        Returns:
            postprocessed_results (list[dict[str, torch.Tensor]]): Results per batch
            [
                {
                    'boxes': (N, 4), # xyxy format
                    'scores': (N, ), # confidence scores
                    'raw_boxes': (N, 4), original xywh format
                },
                ...
            ]
        """

        batch_size = raw_outputs[0].shape[0]
        batch_predictions = []

        for batch_idx in range(batch_size):
            all_boxes = []
            all_scores = []

            # Integrate all 3 sacles
            for scale_idx, output in enumerate(raw_outputs):
                scale_pred = output[batch_idx]  # (num_anchors, w, h, (num_classes + 5))

                # Transform into grid coordinates
                boxes, scores = self._decode_single_scale(scale_pred, scale_idx)

                if len(boxes) > 0:
                    all_boxes.append(boxes)
                    all_scores.append(scores)

            if len(all_boxes) == 0:
                batch_predictions.append(
                    {
                        "boxes": torch.empty(0, 4),
                        "scores": torch.empty(0),
                        "raw_boxes": torch.empty(0, 4),
                    }
                )
                continue

        # Integrate result from all scales
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)

        # Filter by confidence
        mask = all_scores >= conf_threshold
        filtered_boxes = all_boxes[mask]
        filtered_scores = all_scores[mask]

        if len(filtered_boxes) == 0:
            batch_predictions.append(
                {
                    "boxes": torch.empty(0, 4),
                    "scores": torch.empty(0),
                    "raw_boxes": torch.empty(0, 4),
                }
            )

        # Apply NMS (Non-Max Suppresion)
        raw_boxes_xywh = filtered_boxes.clone()
        filtered_boxes_xyxy = self.xywh2xyxy(filtered_boxes)

        keep_indices = nms(filtered_boxes_xyxy, filtered_scores, nms_threshold)

        final_boxes = filtered_boxes_xyxy[keep_indices]
        final_scores = filtered_scores[keep_indices]
        final_raw_boxes = raw_boxes_xywh[keep_indices]

        batch_predictions.append(
            {"boxes": final_boxes, "scores": final_scores, "raw_boxes": final_raw_boxes}
        )

        return batch_predictions

    def _decode_single_scale(
        self, scale_output: torch.Tensor, scale_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode single scale output
        """

        n_anchors, grid_h, grid_w, _ = scale_output.shape

        # Make grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h), torch.arange(grid_w), indexing="ij"
        )
        grid_x = grid_x.to(self.device).float()
        grid_y = grid_y.to(self.device).float()

        boxes = []
        scores = []

        for anchor_idx in range(n_anchors):
            anchor_pred = scale_output[anchor_idx]  # (h, w, (num_classes + 5))

            # Get predictions
            pred_x = torch.sigmoid(anchor_pred[..., 0]) + grid_x
            pred_y = torch.sigmoid(anchor_pred[..., 1]) + grid_y
            pred_w = (
                torch.exp(anchor_pred[..., 2]) * self.anchors[scale_idx][anchor_idx][0]
            )
            pred_h = (
                torch.exp(anchor_pred[..., 3]) * self.anchors[scale_idx][anchor_idx][1]
            )
            pred_conf = torch.sigmoid(anchor_pred[..., 4])

            # Normalize
            pred_x = pred_x / grid_h
            pred_y = pred_y / grid_w
            pred_w = pred_w / self.img_size
            pred_h = pred_h / self.img_size

            # Flatten tensor
            pred_x = pred_x.flatten()
            pred_y = pred_y.flatten()
            pred_w = pred_w.flatten()
            pred_h = pred_h.flatten()
            pred_conf = pred_conf.flatten()

            # Create boxes and scores
            anchor_boxes = torch.stack((pred_x, pred_y, pred_w, pred_h), dim=1)

            boxes.append(anchor_boxes)
            scores.append(pred_conf)

        if len(boxes) > 0:
            boxes = torch.cat(boxes, dim=0)
            scores = torch.cat(scores, dim=0)
        else:
            boxes = torch.empty(0, 4)
            scores = torch.empty(0)

        return boxes, scores

    # ========== Calculate evaluation metrics ==========

    def calculate_metrics(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[torch.Tensor],
        iou_threshold: float = 0.5,
        return_each: bool = False,
    ) -> dict[str, float]:
        """
        Calculate evaluation metrics:

        Args:
            predictions (list[dict[str, torch.Tensor]]): List of predictions
            targets (list[torch.Tensor]): List of targets
            iou_threshold (float): IoU threshold
            return_each (bool): Whether to return each metric; false to return dict; default False

        Returns:
            if return_each:
                mAP, precision, recall, avg_iou (float)

            else:
                metrics {
                    'mAP': float,
                    'precision' :float,
                    'racall': float,
                    'avg_iou': float,
                    'n_tp': int, # true positives
                    'n_fp': int, # false positives
                    'n_fn': int  # false negatives
                }
        """
        all_tp = []
        all_fp = []
        total_gt = 0
        total_iou = 0.0
        n_matched = 0

        for pred, gt in zip(predictions, targets):
            if len(gt) == 0:
                # No ground truth
                if len(pred["boxes"]) > 0:
                    all_fp.extend([1] * len(pred["boxes"]))
                continue

            total_gt += len(gt)

            if len(pred["boxes"]) == 0:
                # No prediction
                continue

            # Calculate IoU
            pred_boxes_xyxy = pred["boxes"]

            gt_boxes_xyxy = self.xywh2xyxy(
                gt[:, 1:5]
            )  # [class, x, y, w, h] -> [x1, y1, x2, y2]
            iou_matrix = self.calculate_iou(pred_boxes_xyxy, gt_boxes_xyxy)

            # Matching
            tp, fp, matched_ious = self._match_predictions(
                pred["scores"], iou_matrix, iou_threshold
            )

            all_tp.extend(tp)
            all_fp.extend(fp)

            if len(matched_ious) > 0:
                total_iou += sum(matched_ious)
                num_matched += len(matched_ious)

        # Calculate metrics
        n_tp = sum(all_tp)
        n_fp = sum(all_fp)
        n_fn = total_gt - n_tp

        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        avg_iou = total_iou / n_matched if n_matched > 0 else 0.0

        # Calculate mAP (simplified version) (11 points interpolation)
        mAP = self._calculate_ap(all_tp, all_fp, total_gt)

        if return_each:
            mAP, precision, recall, avg_iou

        return {
            "mAP": mAP,
            "precision": precision,
            "recall": recall,
            "avg_iou": avg_iou,
            "n_tp": sum(all_tp),
            "n_fp": sum(all_fp),
            "n_fn": total_gt - sum(all_tp),
        }

    def _match_predictions(
        self, scores: torch.Tensor, iou_matrix: torch.Tensor, iou_threshold: float
    ) -> tuple[list[int], list[int], list[float]]:
        """
        Match predictions with ground truth

        Args:
            scores (torch.Tensor): Prediction scores
            iou_matrix (torch.Tensor): IoU matrix
            iou_threshold (float): IoU threshold

        Returns:
            tp (list[int]): List of true positives
            fp (list[int]): List of false positives
            matched_ious (list[float]): List of matched IoU
        """

        # Sort by confidence (desc)
        sorted_indices = torch.argsort(scores, descending=True)

        matched_gt = set()
        tp = []
        fp = []
        matched_ious = []

        for pred_idx in sorted_indices:
            pred_idx = pred_idx.item()

            # Search for optimal GT
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx in range(iou_matrix.shape[1]):
                if gt_idx in matched_gt:
                    continue

                iou = iou_matrix[pred_idx, gt_idx].item()

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check matching
            if best_iou > iou_threshold:
                tp.append(1)
                fp.append(0)
                matched_gt.add(best_gt_idx)
                matched_ious.append(best_iou)
            else:
                tp.append(0)
                fp.append(1)

        return tp, fp, matched_ious

    def _calculate_ap(self, tp: list[int], fp: list[int], total_gt: int) -> float:
        """
        Calculate average precision (simplified version: 11 points interpolation)

        Args:
            tp (list[int]): List of true positives
            fp (list[int]): List of false positives
            total_gt (int): Total number of ground truth objects

        Returns:
            ap (float): Average precision
        """
        if len(tp) == 0:
            return 0.0

        # Calculate accumulated tp and fp
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / total_gt

        # Interpolation
        ap = 0.0
        for threshold in np.arange(0.0, 1.1, 0.1):
            mask = recalls > threshold
            if np.any(mask):
                ap += np.max(precisions[mask])

        return ap / 11

    # ========== Evaluating model ==========
    def eval_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        *,
        model_name: str = None,
        config: dict[str, Any],
        return_each: bool = False,
    ) -> tuple[float, float, float, float, float, float, float] | dict[str, float]:
        """
        Evaluate a model

        Args:
            model (nn.Module): Model to evaluate
            test_loader (DataLoader): DataLoader for test data
            model_name (str): Name of the model
            config (dict[str, Any]): Configuration
            return_each (bool): Whether to return each metric; false to return dict; default False

        Returns:
            if return_each:
                mAP (float): Mean average precision
                mAP_0_3 (float): Mean average precision for IoU < 0.3
                precision (float): Precision
                recall (float): Recall
                recall_0_3 (float): Recall for IoU < 0.3
                avg_iou (float): Average IoU
                f1_score (float): F1 score

            else:
                results (dict[str, float]): Dictionary of metrics
        """
        eval_config = config.get("evaluating", {})
        iou_threshold = eval_config.get("iou_threshold", 0.5)
        conf_threshold = eval_config.get("conf_threshold", 0.5)
        nms_threshold = eval_config.get("nms_threshold", 0.4)

        print(f"\n{__name__}:: Evaluating model...")
        model = model.to(self.device)
        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]

                outputs = model(images)
                losses = yolo_loss(outputs, targets, **self.config["loss"])

                total_loss += losses["total_loss"].item()

                decoded_raw_targets = self._decode_ground_truth_grid(targets)

                # Get predictions
                batch_preds = self.postprocess_raw_outputs(
                    outputs, conf_threshold, nms_threshold
                )

                all_preds.extend(batch_preds)
                all_targets.extend(decoded_raw_targets)

        if return_each:
            mAP, precision, recall, avg_iou = self.calculate_metrics(
                all_preds, all_targets, iou_threshold, return_each
            )

            # Disaster specific metrics
            mAP_0_3, _, recall_0_3, _ = self.calculate_metrics(
                all_preds, all_targets, iou_threshold=0.3, return_each=return_each
            )
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            return mAP, mAP_0_3, precision, recall_0_3, recall, avg_iou, f1_score

        else:
            metrics = self.calculate_metrics(all_preds, all_targets, iou_threshold)
            disaster_metrics = self.calculate_metrics(
                all_preds, all_targets, iou_threshold=0.3
            )

            precision = metrics["precision"]
            recall = metrics["recall"]
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results = {
                "model": (
                    self.config["model"]["name"] if model_name is None else model_name
                ),
                "n_epochs": self.config["training"]["n_epochs"],
                "learning_rate": self.config["optimizer"]["lr"],
                "weight_decay": self.config["optimizer"]["weight_decay"],
                "batch_size": self.config["dataloader"]["batch_size"],
                "accumulation_steps": self.config["training"]["accumulation_steps"],
                "total_loss": round(total_loss / len(test_loader), 4),
                "mAP": round(metrics["mAP"], 4),
                "mAP_0.3": round(disaster_metrics["mAP"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "recall_0.3": round(disaster_metrics["recall"], 4),
                "IoU": round(metrics["avg_iou"], 4),
                "f1_score": round(f1_score, 4),
                "n_preds": sum(len(p["boxes"]) for p in all_preds),
                "n_ground_truths": sum(len(t) for t in all_targets),
            }

            return results


def learning_curve(
    train_history: dict[str, list[float]],
    val_history: dict[str, list[float]],
    show_fig: bool = True,
    fig_path: str = None,
) -> None:
    """
    Plot learning curve

    Args:
        train_history (dict[str, list[float]]): Training history dictionary
        val_history (dict[str, list[float]]): Validation history dictionary
        show_fig (bool, optional): Whether to show the figure. Defaults to True.
        fig_path (str, optional): Path to save the figure. Defaults to None.
    """

    plt.figure(figsize=(12, 6))
    for i, key in enumerate(train_history.keys()):
        plt.subplot(2, 3, i + 1)
        plt.plot(train_history[key], label=f"Training {key}")
        plt.plot(val_history[key], label=f"Validation {key}")
        plt.title(f"{key} Over Epochs")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Metric", fontsize=10)
    plt.tight_layout()
    plt.legend()

    if fig_path:
        plt.savefig(fig_path)
        logger.info(f"Figure saved to {fig_path}")

    if show_fig:
        plt.show()
