# Francesco Chiumento, 2024

from model import UNET
from utils import load_checkpoint
from albumentations.pytorch import ToTensorV2
from dataset import CartilageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import albumentations as A
import numpy as np
import torchvision
import os
import nibabel as nib

class NewDataDataset(CartilageDataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        super().__init__(image_dir, mask_dir, transform)
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')], key=lambda x: x.lower())
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')], key=lambda x: x.lower()) if mask_dir is not None else None
        
        if self.masks is not None:
            assert len(self.images) == len(self.masks), "Number of images and masks do not match"
            for img, mask in zip(self.images, self.masks):
                assert os.path.splitext(os.path.splitext(img)[0])[0].lower() == os.path.splitext(os.path.splitext(mask)[0])[0].lower(), f"Image and mask do not match: {img} != {mask}"
                
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = nib.load(img_path).get_fdata()
        image = np.stack((image,) * 3, axis=-1)  # Convert to 3 channels if necessary

        if self.masks is not None:
            mask_path = os.path.join(self.mask_dir, self.masks[index])
            mask = nib.load(mask_path).get_fdata()
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32) 

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

def load_model(checkpoint_path, device):
    model = UNET(in_channels=3, out_channels=1).to(device)
    load_checkpoint(torch.load(checkpoint_path, map_location=device), model)
    model.eval()
    return model

def predict(model, loader, device, use_masks=True):
    model.eval()
    saved_predictions = []
    dice_scores_messages = []

    pbar = tqdm(total=len(loader), desc="Processing slices", unit="slice")
    with torch.no_grad():
        for idx, (image, mask) in enumerate(loader):
            image = image.to(device)
            preds = torch.sigmoid(model(image))
            preds_bin = (preds > 0.5).float()
            saved_predictions.append(preds_bin.cpu().numpy())
            
            if use_masks:
                mask = mask.to(device).float()
                mask = (mask > 0.5).float()
                preds_flat = preds_bin.view(-1)
                mask_flat = mask.view(-1)
                dice_score = dice_coefficient(preds_flat, mask_flat)
                dice_scores_messages.append(f"Slice {idx}: Dice score: {dice_score.item():.4f}")
            
            pbar.update(1)
    pbar.close()

    for message in dice_scores_messages:
        print(message)
    
    return saved_predictions

def dice_coefficient(pred, target):
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum())
    
    return dice

def run_segmentation(checkpoint_path, images_dir, masks_dir, output_folder, device, use_masks=True):
    print("Loading the model...")
    model = load_model(checkpoint_path, device)

    transform = A.Compose([
        A.Resize(height=384, width=384),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    print("Preparing dataset...")
    dataset = NewDataDataset(images_dir, masks_dir if use_masks else None, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    print("Starting the prediction process...")
    predictions = predict(model, loader, device, use_masks=use_masks)

    if not os.path.exists(output_folder):
        print(f"Creating the output directory at {output_folder}")
        os.makedirs(output_folder)
    
    print("Saving the predicted images...")
    
    for idx, pred in enumerate(predictions):
        pred_numpy = pred.squeeze()
        # Apply inverse flipping
        pred_numpy = np.flipud(pred_numpy)
        pred_numpy = np.fliplr(pred_numpy)
        pred_numpy = np.rot90(pred_numpy, -3)  # Rotate 90 degrees (equivalent to -270 degrees)

        formatted_idx = str(idx).zfill(4)
        save_path = os.path.join(output_folder, f"prediction_{formatted_idx}.nii.gz")
        pred_nii = nib.Nifti1Image(pred_numpy, affine=np.eye(4))
        nib.save(pred_nii, save_path)
