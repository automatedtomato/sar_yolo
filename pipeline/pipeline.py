from data.dataset import create_dataloader
from utils import load_config, validate_config, AnchorOptimizer
import torch
import torchvision.transforms as transforms

from models.yolov3 import YOLOv3
from models.train_val.train import train_model
from models.train_val.evaluation import learning_curve, YOLOv3Evaluator
from utils.data_stream import DataStream

import pandas as pd
from logging import getLogger

logger = getLogger(__name__)

class Pipeline:
    """
    Pipeline class for training and evaluating YOLOv3 model.
    
    Args:
        config_path (str): Path to configuration file
        load_ratio (float): Ratio of data to load
    
    Attributes:
        config (dict): Configuration dictionary
        data_stream (DataStream): DataStream object instance
    """
    
    def __init__(
        self,
        config_path: str,
        ):
        
        self.config_path = config_path
        
        # Load configuration file
        config = load_config(config_path)
        self.config = validate_config(config)
        
        # Initialize data stream
        source = config['data']['source']
        if source == 'gcs':
            bucket_name = config['data']['bucket_name']
            self.data_stream = DataStream(bucket_name)
        if source == 'file':
            data_dir = config['data']['data_dir']
            self.data_stream = DataStream(dir_path=data_dir)
        else:
            raise ValueError("Data source must be either 'gcs' or 'file'.")
                    

    def train_val_pipeline(
        self,
        load_ratio: float = 1.0,
        optim_anchor: bool = False,
        apply_transforms: bool = True,
        schedule_lr: bool = False,
        show_lc: bool = True
        ):
        
        torch.cuda.empty_cache()
        
        # Optimize anchors with K-means clustering
        if optim_anchor:
            anch_opt = AnchorOptimizer(data_stream=self.data_stream, config_path=self.config_path)
            optimized = anch_opt.optimize_anchors(9)
            anch_opt.update_config(optimized, backup=False)
            self.config = load_config(self.config_path)

    
        """ アノテーションとのズレが生じるものは削除 """
        transform = transforms.Compose([
            transforms.Resize((self.config['model']['input_size'], self.config['model']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.RandomHorizontalFlip(p=0.25), # diversify perspective
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # diversify lighting condition
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), # simulate noise
            # transforms.RandomResizedCrop(size=416, scale=(0.8, 1.0)), # simulate altitude change
            # transforms.RandomRotation(degrees=3) # diversify orientation
        ])
            
        train_loader, val_loader, test_loader = create_dataloader(
            data_stream=self.data_stream,
            config=self.config,
            transform=transform if apply_transforms else None,
            load_ratio=load_ratio
        )
        if apply_transforms:
            print(f"\n{__name__}:: Data transforms are applied to the dataset. Total {len(train_loader.dataset)} images are used for training.")

        model = YOLOv3(self.config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{__name__}:: Using device: {device}")

        model = model.to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['optimizer']['lr'], weight_decay=self.config['optimizer']['weight_decay'])
        
        if schedule_lr:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.config['scheduler']['milestones'],
                gamma=self.config['scheduler']['gamma']
                )
        else:
            scheduler = None
        
        model, train_history, val_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )

        learning_curve(train_history, val_history, show_fig = show_lc, fig_path=self.config['evaluating']['fig_path'])
        
        evaluator = YOLOv3Evaluator(device=device, config=self.config)
        metrics = evaluator.eval_model(
            model=model,
            test_loader=test_loader,
            config=self.config,
        )
        
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(self.config['evaluating']['metrics_path'], index=False)
        
        print(metrics_df)
        return metrics, model