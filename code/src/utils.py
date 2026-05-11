# Francesco Chiumento, 2024

import torch
import torchvision
from dataset import CartilageDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename): 
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=20, pin_memory=True):
    train_ds = CartilageDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_ds = CartilageDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def calculate_dice(pred, target, smooth=1e-6):
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_per_image = (2 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return dice_per_image

def process_batch(x_batch, y_batch, model, device):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    with torch.amp.autocast('cuda'):
        preds_batch = model(x_batch)
    preds_batch = torch.sigmoid(preds_batch)
    preds_batch = (preds_batch > 0.5).float()
    dice_scores = calculate_dice(preds_batch, y_batch)
    return dice_scores

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for x, y in loader:
            batch_dice_scores = process_batch(x, y, model, device)
            dice_scores.append(batch_dice_scores)

    dice_scores = torch.cat(dice_scores)
    mean_dice_score = dice_scores.mean().item()
    print(f"Dice score: {mean_dice_score:.4f}")

    model.train()
    return mean_dice_score

def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{idx}.png")

    model.train()
