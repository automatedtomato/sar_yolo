import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from utils.data_stream import DataStream
from .dataset import SaRDataset
from typing import Any, Optional
from logging import getLogger

logger = getLogger(__name__)


def create_dataloader(
    data_stream: DataStream,
    config: dict[str, Any],
    output_size: int = -1,
    load_ratio: float = 1.0,
    transform: Optional[transforms.Compose] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create a dataloader for the dataset.

    Args:
        data_stream (DataStream): DataStream object instance
        config (dict[str, Any]): Configuration dictionary
        output_size (int, optional): Output size. Defaults to -1.
        load_ratio (float, optional): Load ratio (0.0 to 1.0) to split the dataset. Defaults to 1.0.
        transform (Optional[transforms.Compose], optional): Transformation pipeline. Defaults to None.

    Returns:
        DataLoader: DataLoader objects (train, val, test)
    """
    train_config = config["data"]["train"]
    val_config = config["data"]["val"]
    test_config = config["data"]["test"]
    
    if load_ratio > 1.0 or load_ratio <= 0.0:
        raise ValueError("Load ratio must be between 0.0(not included) and 1.0")

    train_dataset = SaRDataset(
        data_stream=data_stream,
        config={
            **config,
            "data": {
                **config["data"],
                "img_path": train_config["img_path"],
                "annot_path": train_config["annot_path"],
            },
        },
        output_size=output_size,
        transform=transform,
    )

    train_dataset = Subset(train_dataset, range(int(len(train_dataset) * load_ratio)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=config["dataloader"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=config["dataloader"]["pin_memory"],
    )

    val_dataset = SaRDataset(
        data_stream=data_stream,
        config={
            **config,
            "data": {
                **config["data"],
                "img_path": val_config["img_path"],
                "annot_path": val_config["annot_path"],
            },
        },
        output_size=output_size,
        transform=transform,
    )

    val_dataset = Subset(val_dataset, range(int(len(val_dataset) * load_ratio)))

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=config["dataloader"]["pin_memory"],
    )

    test_dataset = SaRDataset(
        data_stream=data_stream,
        config={
            **config,
            "data": {
                **config["data"],
                "img_path": test_config["img_path"],
                "annot_path": test_config["annot_path"],
            },
        },
        output_size=output_size,
        transform=transform,
    )

    test_dataset = Subset(test_dataset, range(int(len(test_dataset) * load_ratio)))

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=config["dataloader"]["pin_memory"],
    )

    print(
        f"\nTrain Samples: {train_dataset.__len__()}, Val Samples: {val_dataset.__len__()}, Test Samples: {test_dataset.__len__()}"
    )

    return train_loader, val_loader, test_loader


def collate_fn(
    batch: list[tuple[torch.Tensor, list[torch.Tensor]]],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Custom collate function for DataLoader

    Args:
        batch (list[tuple[torch.Tensor, list[torch.Tensor]]]): List of tuples (image, targets)

    Returns:
        images: [batch_size, channel, height, width]
        targets: [target_scale1, target_scale2, target_scale3]
    """

    images = []
    targets_batch = [[], [], []]  # for 3 scales

    for image, targets in batch:
        images.append(image)
        for i, target in enumerate(targets):
            targets_batch[i].append(target)

    # Concatenate in batch dimension
    images_batch = torch.stack(images)
    for i in range(3):
        if targets_batch[i]:
            targets_batch[i] = torch.stack(targets_batch[i])
        else:  # if no targets, create zero tensor
            grid_size = [13, 26, 52][i]
            targets_batch[i] = torch.zeros(
                1, grid_size, grid_size, 5 + batch[0][1][i].shaep[-1] - 5
            )

    return images_batch, targets_batch
