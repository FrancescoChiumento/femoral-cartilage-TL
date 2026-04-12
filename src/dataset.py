# Francesco Chiumento, 2024

import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class CartilageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)

        # Ensure the image is 3-channel (RGB)
        if len(image.shape) == 2:  # If the image is grayscale, expand to 3 channels
            image = np.stack([image] * 3, axis=-1)

        mask[mask == 255.0] = 1.0   # Assuming the mask uses 255 for the object of interest

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image, mask
