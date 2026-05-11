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

from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders,  
    check_accuracy, 
    save_predictions_as_imgs, 
)

# Hyperparameters etc.
best_dice_score = 0.0
LEARNING_RATE = 0.00006207941891566812
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 7 
NUM_EPOCHS = 100
NUM_WORKERS = 20 
IMAGE_HEIGHT = 384 
IMAGE_WIDTH = 384 
PIN_MEMORY = True 
LOAD_MODEL = False 
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"
SAVED_IMAGES_PATH = "saved_images"
start_scheduler_after = 50 

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

def main(): 
    global best_dice_score

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

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

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

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"),model)

    check_accuracy(val_loader, model, device = DEVICE)

    scaler = torch.amp.GradScaler('cuda') 

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler) 

        current_dice_score = check_accuracy(val_loader, model, device=DEVICE)
        if current_dice_score > best_dice_score:
            best_dice_score = current_dice_score
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, CHECKPOINT_PATH)
            print(f"Checkpoint saved with improved Dice score: {current_dice_score}")

        if epoch >= start_scheduler_after:
            scheduler.step(current_dice_score)

        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        save_predictions_as_imgs(
            val_loader, model, folder=SAVED_IMAGES_PATH, device=DEVICE
        )

if __name__ == "__main__":
    main()
