import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.loss_func import yolo_loss

from typing import Any, Optional

import os


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    optimizer: torch.optim=None,
    criterion=yolo_loss,
    device: torch.device = torch.device("cpu"),
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, list[float]]]:
    """
    Train and validate model

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        config (dict[str, Any]): Configuration dictionary
        optimizer (torch.optim): Optimizer
        criterion (function): Loss function
        device (torch.device, optional): Device to use. Defaults to 'cpu'.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler for learning rate. Defaults to None.

    Returns:
        nn.Module: Trained model
        tuple[dict[str, list[float]], dict[str, list[float]]]:
            Training and validation history:
                {
                    'total_loss': [],
                    'coor_lossd': [],
                    'obj_loss': [],
                    'noobj_loss': [],
                    'class_loss': []
                }
    """

    n_epochs = config["training"]["n_epochs"]
    patience = config["training"].get("patience", 5)
    save_path = config["training"]["save_path"]
    log_interval = config["training"].get("log_interval", 10)
    accumulation_steps = config["training"].get("accumulation_steps", 1)
    
    # Initialize optimizer
    if optimizer is None:
        
        lr = config['optimizer']['lr']
        weight_decay = config['optimizer']['weight_decay']
        
        if config['optimizer']['type'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        else:
            # TODO: Add other optimizers
            pass

    # Create save directory
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialize history
    train_history = {
        "total_loss": [],
        "coord_loss": [],
        "obj_loss": [],
        "noobj_loss": [],
        "class_loss": [],
    }

    val_history = {
        "total_loss": [],
        "coord_loss": [],
        "obj_loss": [],
        "noobj_loss": [],
        "class_loss": [],
    }

    # Early stopping variables
    best_val_loss = float("inf")
    counter = 0

    # Move model to device
    model = model.to(device)

    print("Training model...")
    print(f"  Number of epochs: {n_epochs}")
    print(f"  Device: {device}")
    print(f"  Patience: {patience}")
    print(f"  Save path: {save_path}\n")

    # Training loop
    for epoch in range(n_epochs):
        # =========== Training ===========
        model.train()
        train_losses = {
            "total": 0.0,
            "coord": 0.0,
            "obj": 0.0,
            "noobj": 0.0,
            "class": 0.0,
        }

        train_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):

            # Move data to device
            images, targets = images.to(device), [t.to(device) for t in targets]

            # Set gradients to zero
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            losses = criterion(outputs, targets, **config["loss"])
            
            # Accumulate gradients
            loss = losses['total_loss'] / accumulation_steps
            loss.backward()
            
            # Update loss
            train_losses["total"] += losses["total_loss"].item()
            train_losses["coord"] += losses["coord_loss"].item()
            train_losses["obj"] += losses["obj_loss"].item()
            train_losses["noobj"] += losses["noobj_loss"].item()
            train_losses["class"] += losses["class_loss"].item()
            train_batches += 1
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % log_interval == 0:
                print(
                    f"  Epoch: {epoch + 1}/{n_epochs} - Batch: {batch_idx}/{len(train_loader)} - Loss: {losses['total_loss'].item():.4f}"
                )
                print(
                    f"      (Coord: {losses['coord_loss'].item():.4f}, Obj: {losses['obj_loss'].item():.4f}, NoObj: {losses['noobj_loss'].item():.4f}, Class: {losses['class_loss'].item():.4f})"
                )

        # ========== Validation ===========
        model.eval()
        val_losses = {
            "total": 0.0,
            "coord": 0.0,
            "obj": 0.0,
            "noobj": 0.0,
            "class": 0.0,
        }
        val_batches = 0

        with torch.no_grad():
            for images, targets in val_loader:

                # Move data to device
                images, targets = images.to(device), [t.to(device) for t in targets]

                # Forward pass
                outputs = model(images)
                losses = criterion(outputs, targets, **config["loss"])

                # Update loss
                val_losses["total"] += losses["total_loss"].item()
                val_losses["coord"] += losses["coord_loss"].item()
                val_losses["obj"] += losses["obj_loss"].item()
                val_losses["noobj"] += losses["noobj_loss"].item()
                val_losses["class"] += losses["class_loss"].item()
                val_batches += 1

        # Calculate average loss
        avg_train_loss = train_losses["total"] / train_batches
        avg_val_loss = val_losses["total"] / val_batches

        # Update history
        train_history["total_loss"].append(avg_train_loss)
        train_history["coord_loss"].append(train_losses["coord"] / train_batches)
        train_history["obj_loss"].append(train_losses["obj"] / train_batches)
        train_history["noobj_loss"].append(train_losses["noobj"] / train_batches)
        train_history["class_loss"].append(train_losses["class"] / train_batches)

        val_history["total_loss"].append(avg_val_loss)
        val_history["coord_loss"].append(val_losses["coord"] / val_batches)
        val_history["obj_loss"].append(val_losses["obj"] / val_batches)
        val_history["noobj_loss"].append(val_losses["noobj"] / val_batches)
        val_history["class_loss"].append(val_losses["class"] / val_batches)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{n_epochs} finished.")
        print(f"\tLearning Rate: {current_lr:.6f}")
        print(f"\tTrain Loss: {avg_train_loss:.4f}")
        print(f"\tVal Loss: {avg_val_loss:.4f}")
        print(f"\tLoss Details:")
        print(
            f'\t\tCoord Loss: T={train_losses["coord"] / train_batches:.4f}, V={val_losses["coord"] / val_batches:.4f}'
        )
        print(
            f'\t\tObj Loss: T={train_losses["obj"] / train_batches:.4f}, V={val_losses["obj"] / val_batches:.4f}'
        )
        print(
            f'\t\tClass Loss: T={train_losses["class"] / train_batches:.4f}, V={val_losses["class"] / val_batches:.4f}\n'
        )

        # Early stopping
        if patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config": config,
                    },
                    save_path,
                )

            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        print("-" * 60)

    print(f"\nTraining finished. Best val loss = {best_val_loss:.4f}")
    print(f"Saving model to {save_path}")

    return model, train_history, val_history
