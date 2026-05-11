# Francesco Chiumento, 2024

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import shutil
from sklearn.model_selection import train_test_split
from slice_extraction_nifti import run_slice_extraction

from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders,  
    check_accuracy, 
    save_predictions_as_imgs, 
)

# Hyperparameters etc.
best_dice_score = 0.0
LEARNING_RATE = 0.00001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 
NUM_EPOCHS = 50
NUM_WORKERS = 20 
IMAGE_HEIGHT = 384 
IMAGE_WIDTH = 384 
PIN_MEMORY = True 
LOAD_MODEL = True 

PRETRAINED_CHECKPOINT = "YOUR_PATH_HERE/checkpoints_sag_dp_cube/iteration_11_checkpoint_SAG_DP_CUBE_LESIONED_AND_NOT_LESIONED.pth.tar" 
FINETUNE_MODE = True 

TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"

SAVED_IMAGES_PATH = "saved_images"

# Fine-tuning specific settings
FREEZE_ENCODER = True
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 0.001
CHECKPOINT_DIR = "finetuned_checkpoints_frozen_encoder"    
start_scheduler_after = 10 

#os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAVED_IMAGES_PATH, exist_ok=True)

def prepare_dess_data(train_patients_list=None, val_patients_list=None, source_images_path=None, source_masks_path=None):
    """Automatically prepare DESS data with train/val split"""
    
    if source_images_path is None:
        source_images = "YOUR_PATH_HERE/patients/images"
    else:
        source_images = source_images_path
        
    if source_masks_path is None:
        source_masks = "YOUR_PATH_HERE/patients/masks"
    else:
        source_masks = source_masks_path

    for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_MASK_DIR, exist_ok=True)
    
    temp_dir = "temp_split"
    os.makedirs(temp_dir, exist_ok=True)

    if train_patients_list is None or val_patients_list is None:
        patients = sorted([f for f in os.listdir(source_images) if f.endswith('.mha')])
        train_patients, val_patients = train_test_split(patients, test_size=0.25, random_state=42)
    else:
        train_patients = train_patients_list
        val_patients = val_patients_list
    
    print(f"Train: {train_patients}")
    print(f"Val: {val_patients}")
    
    for folder in ['train_images', 'train_masks', 'val_images', 'val_masks']:
        os.makedirs(os.path.join(temp_dir, folder), exist_ok=True)
    
    for patient in train_patients:
        shutil.copy(os.path.join(source_images, patient), 
                   os.path.join(temp_dir, 'train_images', patient))
        mask_file = patient.replace('.mha', '.nii.gz')
        shutil.copy(os.path.join(source_masks, mask_file), 
                   os.path.join(temp_dir, 'train_masks', mask_file))
    
    for patient in val_patients:
        shutil.copy(os.path.join(source_images, patient), 
                   os.path.join(temp_dir, 'val_images', patient))
        mask_file = patient.replace('.mha', '.nii.gz')
        shutil.copy(os.path.join(source_masks, mask_file), 
                   os.path.join(temp_dir, 'val_masks', mask_file))
    
    run_slice_extraction(
        os.path.join(temp_dir, 'train_images'),
        os.path.join(temp_dir, 'train_masks'),
        os.path.join(temp_dir, 'val_images'),
        os.path.join(temp_dir, 'val_masks'),
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR
    )
    
    shutil.rmtree(temp_dir)
    print("Slices extracted")

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop): 
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE) 

        #forward
        with torch.amp.autocast('cuda'):
            predictions = model(data) 
            loss = loss_fn(predictions,targets) 

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward() 
        scaler.step(optimizer) 

        scaler.update() 

        loop.set_postfix(loss=loss.item())

def to_float32(image, **kwargs):
    return image.astype(np.float32)

def main(train_patients=None, val_patients=None, num_epochs=None, checkpoint_dir=None, source_images_path=None, source_masks_path=None):

    global best_dice_score, NUM_EPOCHS, CHECKPOINT_DIR
    
    if num_epochs is not None:
        NUM_EPOCHS = num_epochs
    if checkpoint_dir is not None:
        CHECKPOINT_DIR = checkpoint_dir
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("Preparing DESS data")
    prepare_dess_data(train_patients, val_patients, source_images_path, source_masks_path)
    train_transform = A.Compose(
        [
            A.Resize(height=384, width=384),
            A.Rotate(limit=16, p=0.5),
            #A.HorizontalFlip(p=0.5),
            A.ElasticTransform(alpha=23.59237990289956, sigma=5.615320305420207, p=0.5),
            A.RandomBrightnessContrast(p=0.12765906010940886),
            A.GaussianBlur(blur_limit=(7, 11), p=0.1),
            A.Lambda(image=to_float32, mask=to_float32), 
            A.Normalize(
                mean=[0.0, 0.0, 0.0],  
                std=[1.0, 1.0, 1.0],  
                max_pixel_value=255.0,
                always_apply=True
            ),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=384, width=384),
            A.Lambda(image=to_float32, mask=to_float32),  
            A.Normalize(
                mean=[0.0, 0.0, 0.0],  
                std=[1.0, 1.0, 1.0],   
                max_pixel_value=255.0,
                always_apply=True
            ),
            ToTensorV2(),
        ]
    )
    model = UNET(in_channels=3, out_channels=1, dropout_rate=0.22818128569325818).to(DEVICE) 
    loss_fn = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    train_loader, val_loader = get_loaders( 
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    if LOAD_MODEL and FINETUNE_MODE:
        print("\n" + "="*50)
        print("FINE-TUNING MODE")
        print("="*50)
        print(f"Loading checkpoint from: {PRETRAINED_CHECKPOINT}")
        pretrained_checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(pretrained_checkpoint["state_dict"])
        print("Pretrained weights loaded successfully")
        if "epoch" in pretrained_checkpoint:
            print(f"Original checkpoint epoch: {pretrained_checkpoint['epoch']}")
        if "dice_score" in pretrained_checkpoint:
            print(f"Original performance: Dice = {pretrained_checkpoint['dice_score']:.4f}")
        if FREEZE_ENCODER:
            frozen_params = 0
            for param in model.downs.parameters():
                param.requires_grad = False
                frozen_params += param.numel() 
            print(f"Encoder frozen ({frozen_params:,} parameters)")
 
    print("Initial validation")
    check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.amp.GradScaler('cuda') 

    #Early stopping
    best_val_dice = 0.0
    epochs_no_improve = 0
    early_stop = False
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        if early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best model at epoch {best_epoch} with dice: {best_val_dice:.4f}")
            break
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        current_dice_score = check_accuracy(val_loader, model, device=DEVICE)
        if current_dice_score > best_val_dice + MIN_DELTA:
            improvement = current_dice_score - best_val_dice
            best_val_dice = current_dice_score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "dice_score": current_dice_score,
                "is_finetuned": True,
                "original_checkpoint": PRETRAINED_CHECKPOINT
            }
            checkpoint_name = f"best_finetuned_dice_{current_dice_score:.4f}_epoch_{epoch+1}.pth.tar"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            save_checkpoint(checkpoint, checkpoint_path)
            print(f"New best model with Dice: {current_dice_score:.4f} (+{improvement:.4f})")
            print(f"Saved to: {checkpoint_path}")
            best_path = os.path.join(CHECKPOINT_DIR, "best_finetuned_model.pth.tar")
            save_checkpoint(checkpoint, best_path)
        else:
            epochs_no_improve += 1
            print(f"Dice: {current_dice_score:.4f} (no improvement for {epochs_no_improve} epochs)")
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            early_stop = True
        if epoch >= start_scheduler_after:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(current_dice_score)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        if (epoch + 1) % 5 == 0:
            print("Saving sample predictions")
            epoch_folder = os.path.join(SAVED_IMAGES_PATH, f"epoch_{epoch+1}")
            os.makedirs(epoch_folder, exist_ok=True)
            save_predictions_as_imgs(
                val_loader, model,
                folder=epoch_folder,
                device=DEVICE
            )

    print("\n" + "="*50)
    print("FINE-TUNING COMPLETED")
    print("="*50)
    print(f"Best Dice Score: {best_val_dice:.4f}")
    print(f"Achieved at epoch: {best_epoch}")
    print(f"Best model saved to: {os.path.join(CHECKPOINT_DIR, 'best_finetuned_model.pth.tar')}")
    print("="*50)
    return os.path.join(CHECKPOINT_DIR, 'best_finetuned_model.pth.tar') 

if __name__ == "__main__":
    main()
