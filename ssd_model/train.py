"""
Training script for SSD300-VGG16 model for animal detection.
Uses transfer learning with pre-trained COCO weights and modifies the classification head for 6 classes.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

from dataset import AnimalDataset, get_ssd_transforms


# ============================================================================
# ======================== HYPERPARAMETERS ==================================
# ============================================================================

NUM_CLASSES = 8  # 6 animal classes + 1 background class (IMPORTANT: must match COCO JSON categories)
EPOCHS = 100
BATCH_SIZE = 4  # RTX 3050 has 4GB VRAM; use 4 to avoid OOM. Max 8 if memory allows.
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
PRINT_INTERVAL = 50  # Print loss every N batches

# Device: Explicitly use CUDA if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths (update these to point to your COCO dataset)
TRAIN_IMG_DIR = './animal_dataset/train'
TRAIN_ANN_FILE = './animal_dataset/train/_annotations.coco.json'
# Support both common validation folder names ('val' and 'valid') - prefer 'valid' if present
VAL_IMG_DIR = './animal_dataset/valid'  # Optional: can be same as train if no validation split
VAL_ANN_FILE = './animal_dataset/valid/_annotations.coco.json'

# Output
CHECKPOINT_DIR = './checkpoints'
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_ssd.pth')

# ============================================================================
# ======================== UTILITY FUNCTIONS ================================
# ============================================================================


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles variable number of annotations per image.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def _get_ssd_head_specs(model: nn.Module) -> tuple[list[int], list[int]]:
    """
    Extract the channel layout required to rebuild the SSD classification head.

    torchvision's SSD implementation exposes the per-feature-map conv layers on
    ``classification_head.module_list`` and the anchor counts on the model's
    ``anchor_generator``. This is much more stable than relying on internal
    attributes that vary across torchvision versions.
    """
    cls_head = model.head.classification_head

    if hasattr(cls_head, "module_list"):
        in_channels = [layer.in_channels for layer in cls_head.module_list]
    elif hasattr(cls_head, "extra_layers"):
        # Fallback for older/custom torchvision variants.
        in_channels = [layer[0].in_channels for layer in cls_head.extra_layers]
    else:
        raise AttributeError(
            "Unable to infer SSD classification head layout from the current torchvision version."
        )

    if hasattr(model.anchor_generator, "num_anchors_per_location"):
        num_anchors = list(model.anchor_generator.num_anchors_per_location())
    else:
        # Last-resort fallback: infer anchors from the existing head.
        num_anchors = [layer.out_channels // cls_head.num_columns for layer in cls_head.module_list]

    return in_channels, num_anchors


def modify_ssd_head(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Modify the classification head of SSD300 to match the number of classes.
    
    Args:
        model: Pre-trained SSD300 model
        num_classes: Target number of classes
    
    Returns:
        Modified SSD300 model
    """
    in_channels, num_anchors = _get_ssd_head_specs(model)

    # Create new classification head
    new_classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    
    model.head.classification_head = new_classification_head
    return model


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train for one epoch.
    
    **CRITICAL:** Explicitly map images and targets to the device (CUDA/CPU).
    
    Args:
        model: SSD300 model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device (cuda or cpu)
        epoch: Current epoch number
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # **CRITICAL FOR GPU:** Explicitly move images and targets to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        
        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # Print progress
        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f'  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {losses.item():.4f} (Avg: {avg_loss:.4f})')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate model on validation set.
    
    **CRITICAL:** Explicitly map images and targets to the device.
    
    Args:
        model: SSD300 model
        val_loader: Validation data loader
        device: Device (cuda or cpu)
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    for images, targets in val_loader:
        # **CRITICAL FOR GPU:** Explicitly move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass: torchvision detection models return a dict of losses
        # only when in training mode. In eval() they return detections (a list).
        # To compute validation loss we temporarily enable train mode for the
        # forward pass while keeping gradients disabled via @torch.no_grad().
        was_training = model.training
        # set to train mode to get loss dict
        model.train()
        try:
            loss_dict = model(images, targets)
        finally:
            # restore previous mode (eval)
            model.train(was_training)

        if isinstance(loss_dict, list):
            # Safety: if model still returned detections, we cannot compute loss
            # so skip this batch.
            continue

        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main(args):
    """Main training loop."""
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Num Classes (with background): {NUM_CLASSES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    # ========================================================================
    # Load datasets
    # ========================================================================
    print("\n[1/4] Loading datasets...")
    print(f"Device: {DEVICE}")
    
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(TRAIN_ANN_FILE):
        print(f"Error: Training data not found!")
        print(f"  IMG DIR: {TRAIN_IMG_DIR}")
        print(f"  ANN FILE: {TRAIN_ANN_FILE}")
        print("Please update TRAIN_IMG_DIR and TRAIN_ANN_FILE in train.py")
        sys.exit(1)
    
    train_dataset = AnimalDataset(
        img_dir=TRAIN_IMG_DIR,
        ann_file=TRAIN_ANN_FILE,
        annotation_format='coco',
        transform=get_ssd_transforms(is_train=True),
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Animal classes found: {len(train_dataset.class_name_to_idx)}")
    
    # Validation dataset (if available)
    val_dataset = None
    if os.path.exists(VAL_IMG_DIR) and os.path.exists(VAL_ANN_FILE):
        val_dataset = AnimalDataset(
            img_dir=VAL_IMG_DIR,
            ann_file=VAL_ANN_FILE,
            annotation_format='coco',
            transform=get_ssd_transforms(is_train=False),
        )
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print("Note: Validation data not found. Skipping validation.")
    
    # ========================================================================
    # Create data loaders
    # ========================================================================
    print("\n[2/4] Creating data loaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set > 0 for faster data loading on multi-core systems
        collate_fn=collate_fn,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    
    # ========================================================================
    # Load pre-trained SSD300 model and modify head
    # ========================================================================
    print("\n[3/4] Loading pre-trained SSD300-VGG16 model...")
    
    model = ssd300_vgg16(weights='DEFAULT')  # COCO pre-trained weights
    print("Loaded SSD300-VGG16 with COCO weights")
    
    # Modify classification head for our number of classes
    print(f"Modifying classification head for {NUM_CLASSES} classes...")
    model = modify_ssd_head(model, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print("Model ready for training")
    
    # ========================================================================
    # Set up optimizer and learning rate scheduler
    # ========================================================================
    print("\n[4/4] Setting up optimizer and scheduler...")
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,  # Decay LR every 30 epochs
        gamma=0.1,
    )
    
    # ========================================================================
    # Training loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, DEVICE)
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  ✓ Best model saved to {BEST_MODEL_PATH}")
        else:
            # If no validation, save the model every 10 epochs or if train loss improves
            if epoch % 10 == 0:
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  ✓ Model checkpoint saved to {BEST_MODEL_PATH}")
        
        # Update learning rate
        scheduler.step()
    
    # ========================================================================
    # Training complete
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SSD300 for animal detection')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    
    main(args)
