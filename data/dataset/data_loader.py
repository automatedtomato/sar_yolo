import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.data_stream import DataStream
from .dataset import SaRDataset
from typing import Any, Optional
from logging import getLogger

logger = getLogger(__name__)

def create_dataloader(
    data_stream: DataStream,
    config: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    output_size: int = -1,
    num_workers: int = 0,
    transform: Optional[transforms.Compose]=None
    ) -> DataLoader:
    
    """
    Create a dataloader for the dataset.
    
    Args:
        data_stream (DataStream): DataStream object instance
        config (dict[str, Any]): Configuration dictionary
        batch_size (int): Batch size
        shuffle (bool): Shuffle the dataset
        output_size (int, optional): Output size. Defaults to -1.
        num_workers (int, optional): Number of workers. Defaults to 0.
        
    Returns:
        DataLoader: DataLoader object
    """
    
    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 16)
        
    # Create dataset
    dataset = SaRDataset(
        data_stream,
        config,
        output_size,
        transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f'Dataloader created: dataset length={len(dataset)}, batch_size={batch_size}, total_batches={len(dataloader)}')
    
    return dataloader

def collate_fn(batch: list[tuple[torch.Tensor, list[torch.Tensor]]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Custom collate function for DataLoader
    
    Args:
        batch (list[tuple[torch.Tensor, list[torch.Tensor]]]): List of tuples (image, targets)
        
    Returns:
        images: [batch_size, channel, height, width]
        targets: [target_scale1, target_scale2, target_scale3]
    """
    
    images = []
    targets_batch = [[], [], []] # for 3 scales
    
    for image, targets in batch:
        images.append(image)
        for i, target in enumerate(targets):
            targets_batch[i].append(target)
            
    # Concatenate in batch dimension
    images_batch = torch.stack(images)
    for i in range(3):
        if targets_batch[i]:
            targets_batch[i] = torch.stack(targets_batch[i])
        else: # if no targets, create zero tensor
            grid_size = [13, 26, 52][i]
            targets_batch[i] = torch.zeros(1, grid_size, grid_size, 5 + batch[0][1][i].shaep[-1] - 5)

    return images_batch, targets_batch