from utils.data_stream import DataStream
from logging import getLogger
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

logger = getLogger(__name__)


class YOLOVisualizer:
    def __init__(self, data_stream: DataStream):
        """
        Visualize YOLO annotations

        Args:
            data_stream (DataStream): DataStream object instance
        """

        self.data_stream = data_stream

        # Colors corresponding to each label
        self.colors = [
            "#ff4b00",
            "#03af7a",
            "#005aff",
            "#f6aa00",
            "#990099",
        ]

        self.labels = {0: "Bent", 1: "Kneeling", 2: "Lying", 3: "Sitting", 4: "Upright"}

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

                    annotations.append((class_id, x_center, y_center, width, height, 0))

                elif len(parts) <= 4 or len(parts) > 6:
                    raise ValueError(f"Invalid annotation format: {line}")

        return annotations

    def _annot_to_bbox(
        self,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        img_width: int,
        img_height: int,
    ) -> list[tuple[int, float, float, float, float]]:
        """
        Convert YOLO annotations to bounding boxes

        Args:
            x_center (float): X center coordinate
            y_center (float): Y center coordinate
            width (float): Width of the bounding box
            height (float): Height of the bounding box
            img_width (int): Width of the image
            img_height (int): Height of the image

        Returns:
            list[tuple[int, float, float, float, float]]: List of tuples (class_id, x1, y1, x2, y2) where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
        """

        # image size in pixels
        w_px = img_width * width
        h_px = img_height * height

        # image center coordinates
        x_center_px = img_width * x_center
        y_center_px = img_height * y_center

        # top-left corner coordinates
        x1 = int(x_center_px - w_px / 2)
        y1 = int(y_center_px - h_px / 2)

        # bottom-right corner coordinates
        x2 = int(x_center_px + w_px / 2)
        y2 = int(y_center_px + h_px / 2)

        return x1, y1, x2, y2

    def visualize_bbox(
        self,
        image_paths: str,
        annot_paths: str,
        show_labels: bool = True,
        fig_size: tuple[int, int] = (12, 8),
    ):
        """
        Visualize YOLO annotations

        Args:
            image_paths (str): Image file path
            annot_paths (str): Annotation file path
            show_labels (bool, optional): Whether to show labels. Defaults to True.
            fig_size (tuple[int, int], optional): Figure size. Defaults to (12, 8).
        """

        plt.figure(figsize=fig_size)

        # Load image
        image = self.data_stream.load_image(image_paths)
        if image is None:
            logger.error(f"Failed to load image from {image_paths}")
            return

        # Load annotation
        annot = self.data_stream.load_annot(annot_paths, ".txt")
        if annot is None:
            logger.error(f"Failed to load annotation from {annot_paths}")
            return

        # Parse annotation
        annotations = self.parse_annot(annot)

        plt.imshow(image)

        for class_id, x_center, y_center, width, height, label in annotations:

            x1, y1, x2, y2 = self._annot_to_bbox(
                x_center, y_center, width, height, image.width, image.height
            )

            color = self.colors[label % len(self.colors)]

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
            plt.gca().add_patch(rect)

            if show_labels:
                label_text = f"{class_id}: Pose - {self.labels[label]}"
                plt.text(
                    x1,
                    y1 - 5,
                    label_text,
                    fontsize=6,
                    color="white",
                    bbox=dict(facecolor=color, alpha=0.7, lw=0),
                )

            plt.title(f"{os.path.basename(image_paths)}", fontsize=10)
            plt.axis("off")
            plt.show()

    def batch_vis_bbox(
        self,
        img_ext: str,
        annot_ext: str,
        img_prefix: str,
        annot_prefix: str,
        batch_size: int,
        show_labels: bool = True,
        save_fig: bool = False,
        save_path: str = None,
    ):
        """
        Visualize multiple YOLO annotations in a grid

        Args:
            img_ext (str): Image file extension
            annot_ext (str): Annotation file extension
            prefix (str): File prefix
            batch_size (int): Batch size
            show_labels (bool, optional): Whether to show labels. Defaults to True.
            save_fig (bool, optional): Whether to save the figure. Defaults to False.
            save_path (str, optional): Path to save the figure. Defaults to None.
        """

        image_paths, annot_paths = self.data_stream.generate_data(
            img_ext,
            annot_ext,
            img_prefix,
            annot_prefix,
            output_size=batch_size,
            return_sep=True,
        )

        if image_paths is None or annot_paths is None:
            logger.error("Failed to generate data")
            return

        cols = min(len(image_paths), 3)
        rows = (batch_size + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))

        if cols == 1 and rows == 1:
            self.visualize_bbox(image_paths[0], annot_paths[0], show_labels=show_labels)

        else:
            for i in range(batch_size):
                plt.subplot(rows, cols, i + 1)

                image = self.data_stream.load_image(image_paths[i])
                annot = self.data_stream.load_annot(annot_paths[i], ".txt")

                img_w_boxes = image.copy()

                annotations = self.parse_annot(annot)

                plt.imshow(img_w_boxes)
                plt.title(f"{os.path.basename(image_paths[i])}", fontsize=10)
                plt.axis("off")

                for class_id, x_center, y_center, width, height, label in annotations:

                    x1, y1, x2, y2 = self._annot_to_bbox(
                        x_center, y_center, width, height, image.width, image.height
                    )

                    color = self.colors[label % len(self.colors)]

                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                    )
                    plt.gca().add_patch(rect)

                    if show_labels:
                        label_text = f"{class_id}: Pose - {self.labels[label]}"
                        plt.text(
                            x1,
                            y1 - 5,
                            label_text,
                            fontsize=6,
                            color="white",
                            bbox=dict(facecolor=color, alpha=0.7, lw=0),
                        )

        if save_fig:
            plt.savefig(save_path)

        plt.show()
