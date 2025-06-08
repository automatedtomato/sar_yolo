from data.dataset import create_dataloader
from utils import load_config, validate_config, AnchorOptimizer
import torch
import torchvision.transforms as transforms
import torchvision

from models.yolov3 import YOLOv3
from models.train_val.train import train_model
from models.train_val.evaluation import learning_curve, YOLOv3Evaluator
from utils.data_stream import DataStream

import pandas as pd
from logging import getLogger

logger = getLogger(__name__)

class Pipeline:
    def __init__(
        self,
        config_path: str,
        ):
        
        # Load configuration file
        config = load_config(config_path)
        self.config = validate_config(config)
        
        # Initialize data stream
        self.bucket_name = config['data']['bucket_name']
        self.data_stream = DataStream(self.bucket_name)

    def train_val_pipeline(
        self,
        optim_anchor: bool = False,
        apply_transforms: bool = True,
        show_lc: bool = True
        ):
        
        # Optimize anchors with K-means clustering
        if optim_anchor:
            anch_opt = AnchorOptimizer(self.config)
            optimized = anch_opt.optimize_anchors(9)
            anch_opt.update_config(optimized, backup=False)
            self.config = load_config(self.config_path)

    
        transform = transforms.Compose([
            transforms.Resize((self.config['model']['input_size'], self.config['model']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.25), # diversify perspective
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # diversify lighting condition
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), # simulate noise
            transforms.RandomResizedCrop(size=416, scale=(0.8, 1.0)), # simulate altitude change
            transforms.RandomRotation(degrees=3) # diversify orientation
        ])
            
        train_loader, val_loader, test_loader = create_dataloader(
            data_stream=self.data_stream,
            config=self.config,
            transform=transform if apply_transforms else None
        )

        model = YOLOv3(self.config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")

        model = model.to(device)

        model, train_history, val_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=device
        )

        learning_curve(train_history, val_history, show_fig = show_lc, save_path=self.config['evaluating']['fig_path'])
        
        evaluator = YOLOv3Evaluator(device, self.config)
        metrics = evaluator.eval_model(
            model=model,
            test_loader=test_loader,
            config=self.config,
        )
        
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(self.config['evaluating']['metrics_path'], index=False)
        
        print(metrics_df)
        return metrics, model