from typing import Any
import yaml
from logging import getLogger

logger = getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load yaml config file

    Args:
        config_path (str): Config file path

    Returns:
        dict[str, Any]: Config
    """

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"\n{__name__}:: Loaded config from {config_path}")
        return config

    except Exception as e:
        raise FileNotFoundError(f"Failed to load config from {config_path}: {e}")

    except yaml.YAMLError as e:
        raise ValueError(f"Failed to analyze config yaml from {config_path}: {e}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Check and Validate config: if not valid, raise error and returns default config

    Args:
        config (dict[str, Any]): Config

    Returns:
        dict[str, Any]: Validated config
    """

    deafult_config = {
        "model": {
            "name": "sample",
            "input_size": 416,
            "n_classes": 6,
            "grid_sizes": [13, 26, 52],
            "anchors": [
                [(116, 90), (156, 198), (373, 326)],  # Scale 1: 8x8
                [(30, 61), (62, 45), (59, 119)],  # Scale 2: 16x16
                [(10, 13), (16, 30), (33, 23)],  # Scale 3: 32x32
            ],
        },
        "data": {
            "bucket_name": "sar-dataset",
            "img_ext": ".png",
            "annot_ext": ".txt",
            "test": {
                "img_path": "data/new_dataset3/test/images",
                "annot_path": "data/new_dataset3/All labels with Pose information/labels",
            },
            "train": {
                "img_path": "data/new_dataset3/train/images",
                "annot_path": "data/new_dataset3/All labels with Pose information/labels",
            },
            "val": {
                "img_path": "data/new_dataset3/val/images",
                "annot_path": "data/new_dataset3/All labels with Pose information/labels",
            },
        },
        "dataloader": {"batch_size": 8, "num_workers": 4, "pin_memory": True},
        "training": {
            "log_interval": 100,
            "accumulation_steps": 1,
            "n_epochs": 10,
            "patience": 3,
            "save_path": "sample.pt",
        },
        "loss": {
            "lambda_coord": 5,
            "lambda_obj": 1,
            "lambda_noobj": 0.5,
            "lambda_class": 1.0,
            "obj_threshold": 0.5,
        },
        "optimizer": {"type": "adam", "lr": 1e-3, "weight_decay": 5e-4},
        "evaluating": {
            "iou_threshold": 0.5,
            "nms_threshold": 0.5,
            "conf_threshold": 0.5,
            "fig_path": "figures/sample_lr.png",
            "metrics_path": "metrics/sample_metrics.csv",
        },
    }

    def _merge_config(default: dict, user: dict) -> dict:
        result = default.copy()
        for key, value in user.items():
            if (
                key in result
                and isinstance(value, dict)
                and isinstance(result[key], dict)
            ):
                result[key] = _merge_config(default[key], value)
            else:
                result[key] = value
        return result

    valid_config = _merge_config(deafult_config, config)

    if not valid_config["data"]["bucket_name"]:
        raise ValueError("Bucket name is required")

    if valid_config["model"]["n_classes"] < 0 or not isinstance(
        valid_config["model"]["n_classes"], int
    ):
        raise ValueError("Number of classes is required / should be positive integer")

    print(f"\n{__name__}:: Validated config: {valid_config}")
    return valid_config
