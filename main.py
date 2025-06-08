from data.dataset import create_dataloader
from utils import load_config, validate_config, AnchorOptimizer
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from models.yolov3 import YOLOv3
from models.train_val.train_val import train_model
from models.train_val.evaluation import evaluate_model, learning_curve
from utils.data_stream import DataStream

import argparse
import sys
import os
from dotenv import load_dotenv

from logging import getLogger

logger = getLogger(__name__)

def main():
    """
    Main function: Training and evaluation pipeline
    
    Command line arguments:
        config (str): Path to configuration file
        --bucket (str): Name of the bucket where the data is stored: default to "sar-dataset"
        --optim_anchor (bool): Optimize anchors with K-means clustering: default to False
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLOv3 training pipelin")
    
    parser.add_argument(
        "config",
        type=str,
        # required=True,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--bucket",
        type=str,
        default="sar-dataset",
        help="Name of the bucket where the data is stored",
    )

    parser.add_argument(
        "--optim_anchor",
        action='store_true',
        help="Optimize anchors with K-means clustering",
    )

    args = parser.parse_args()

    # Get command line arguments
    config_path = args.config
    bucket_name = args.bucket
    optim_anchor = args.optim_anchor
    
    # Load and validateconfig file
    config = load_config(config_path)
    config = validate_config(config)
    
    # Initialize data stream
    data_stream = DataStream(bucket_name)

    # Optimize anchors with K-means clustering
    if optim_anchor:
        anch_opt = AnchorOptimizer(config)
        optimized = anch_opt.optimize_anchors(9)
        anch_opt.update_config(optimized, backup=False)
        config = load_config(config_path)


    train_loader, val_loader, _ = create_dataloader(
        data_stream=data_stream,
        config=config,
        transform=transforms.Compose([
            transforms.Resize((config['model']['input_size'], config['model']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.25), # diversify perspective
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # diversify lighting condition
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), # simulate noise
            transforms.RandomResizedCrop(size=416, scale=(0.8, 1.0)), # simulate altitude change
            transforms.RandomRotation(degrees=3) # diversify orientation
        ])
    )

    model = YOLOv3(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nUsing device: {device}")

    model = model.to(device)

    model, train_history, val_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    learning_curve(train_history, val_history, save_fig = False, save_path=config['evaluating']['fig_path'])
    

        
        
if __name__ == "__main__":
    main()