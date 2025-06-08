from .data_stream import DataStream
from .load_config import load_config
from typing import Any
from logging import getLogger
import numpy as np
from sklearn.cluster import KMeans
import yaml

logger = getLogger(__name__)


class AnchorOptimizer:
    """
    Optimize anchors for YOLOv3

    Args:
    === Either config or config_path must be provided ===
        data_stream (DataStream): DataStream object instance
        config (dict[str, Any], optional): Configuration dictionary
        config_path (str): Configuration file path
    """

    def __init__(self, data_stream: DataStream, config: dict[str, Any] = None, config_path: str=None):
        self.data_stream = data_stream
        
        if config is None and config_path is not None:
            self.config_path = config_path
            self.config = load_config(config_path)
            
        elif config is not None and config_path is None:
            self.config = config
            
        elif config is not None and config_path is not None:
            logger.warining("Both config and config_path are provided. Using config_path.")
            self.config_path = config_path
            self.config = load_config(config_path)

        else:
            raise ValueError("Either config or config_path must be provided")

    def calc_box_dimensions(
        self, dataset_type: str = "train"
    ) -> list[tuple[float, float]]:
        """
        Calculate box dimensions for YOLOv3

        Args:
            dataset_type (str, optional): Dataset type. Defaults to 'train'.

        Returns:
            list[tuple[float, float]]: List of box dimensions
        """

        data_config = self.config.get("data", {})
        dataset_config = data_config.get(dataset_type, {})

        img_prefix = dataset_config.get("img_path", "")
        annot_prefix = dataset_config.get("annot_path", "")
        img_extension = data_config.get("img_ext", "")
        annot_extension = data_config.get("annot_ext", "")

        try:
            _, annots = self.data_stream.generate_data(
                img_extension,
                annot_extension,
                img_prefix,
                annot_prefix,
                return_sep=True,
            )
        except Exception as e:
            logger.error(f"Failed to generate data: {e}")
            return []

        box_dimensions = []

        for annot_path in annots:
            try:
                annot_lines = self.data_stream.load_annot(annot_path, annot_extension)
                if annot_lines is None:
                    continue

                annotations = self.parse_annot(annot_lines)
                for annotation in annotations:
                    _, _, _, width, height, _ = annotation
                    box_dimensions.append((width, height))

            except Exception as e:
                logger.error(f"Failed to load annotation from {annot_path}: {e}")
                continue

        return box_dimensions

    def parse_annot(
        self, annot: str
    ) -> list[tuple[int, float, float, float, float, int]]:
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

                    annotations.append(
                        (class_id, x_center, y_center, width, height, label)
                    )

                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    annotations.append(
                        (class_id, x_center, y_center, width, height, -1)
                    )

                elif len(parts) <= 4 or len(parts) > 6:
                    raise ValueError(f"Invalid annotation format: {line}")

        return annotations

    def optimize_anchors(
        self, n_anchors: int = 9, input_size: int = None, dataset_type: str = "train"
    ) -> list[list[list[int]]]:
        """
        optimize anchors for YOLOv3 using K-means clustering

        Args:
            n_anchors (int, optional): Number of anchors. Defaults to 9.
            input_size (int, optional): Input size. Defaults to None.
            dataset_type (str, optional): Dataset type. Defaults to 'train'.

        Returns:
            list[list[list[int]]]: Anchors arrays in YOLO format
        """

        # Collect boundign box dimensions
        box_dimensions = self.calc_box_dimensions(dataset_type)

        if not box_dimensions:
            raise ValueError("No bounding box dimensions found")
            return []

        # Get input size
        if input_size is None:
            input_size = self.config.get("model", {}).get("input_size", 416)

        # Translate from normalized dims to pixel values
        px_dims = []
        for width, height in box_dimensions:
            px_dims.append((width * input_size, height * input_size))

        px_dims = np.array(px_dims)

        logger.info(f"Running K-means clustering with {n_anchors} clusters...")
        logger.info(f"Box dimensions statistics:")
        logger.info(f"  Mean: {px_dims.mean(axis=0)}")
        logger.info(f"  STD: {px_dims.std(axis=0)}")
        logger.info(f"  Min: {px_dims.min(axis=0)}")
        logger.info(f"  Max: {px_dims.max(axis=0)}")

        kmeans = KMeans(n_clusters=n_anchors, n_init=10)
        kmeans.fit(px_dims)

        # Get cluster centers as anchors
        anchors = kmeans.cluster_centers_

        # calc surface area and sort (desc)
        areas = anchors[:, 0] * anchors[:, 1]
        sorted_indices = np.argsort(areas)
        sorted_anchors = anchors[sorted_indices]

        # to integer
        sorted_anchors = np.round(sorted_anchors)

        # Translate into YOLO format
        anchors_per_scale = n_anchors // 3
        yolo_anchors = []

        for i in range(3):
            start_idx = (2 - i) * anchors_per_scale
            end_idx = start_idx + anchors_per_scale
            scale_anchors = sorted_anchors[start_idx:end_idx].tolist()
            yolo_anchors.append(scale_anchors)

        logger.info("Calculated anchors:")
        for i, scale_anchors in enumerate(yolo_anchors):
            logger.info(f"  Scale {i}: {scale_anchors}")
        logger.info(yolo_anchors)

        return yolo_anchors

    def update_config(
        self, new_anchors: list[list[list[int]]], backup: bool = True
    ) -> bool:
        """
        Update anchors in the config file

        Args:
            new_anchors (list[list[list[int]]]): New anchors
            backup (bool, optional): Backup the current config file. Defaults to True.
        """
        try:
            if backup:
                backup_path = self.config_path + ".backup"
                import shutil

                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Backed up config file to {backup_path}")

            self.config["model"]["anchors"] = new_anchors

            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Updated anchors in config file: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to update anchors in config file: {e}")
            return False
