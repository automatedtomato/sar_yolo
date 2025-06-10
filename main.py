import argparse

from pipeline import Pipeline

from logging import getLogger

logger = getLogger(__name__)

if __name__ == "__main__":
    
    """
    Main function: Training and evaluation pipeline
    
    Command line arguments:
        config (str): Path to configuration file
        --bucket (str): Name of the bucket where the data is stored: default to "sar-dataset"
        --optim_anchor (bool): Optimize anchors with K-means clustering: default to False
        --no-transform (bool): If this flag is set, data transforms will not be applied
        
    Usage:
        python main.py config.yaml --bucket sar-dataset --optim_anchor
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
    
    parser.add_argument(
        '--no-transform',
        action='store_false',
        help='Apply data transforms'
    )
    
    parser.add_argument(
        '--schedule-lr',
        action='store_true',
        help='Schedule learning rate'
    )

    args = parser.parse_args()

    # Get command line arguments
    config_path = args.config
    optim_anchor = args.optim_anchor
    no_transform = args.no_transform
    schedul_lr = args.schedule_lr
    
    
    pl = Pipeline(
        config_path=config_path,
    )
    
    pl.train_val_pipeline(
        optim_anchor=optim_anchor,
        apply_transforms=not no_transform,
        show_lc=False,
        schedule_lr=schedul_lr
    )