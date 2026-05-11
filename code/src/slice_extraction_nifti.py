# Francesco Chiumento, 2024

import SimpleITK as sitk
import os
import numpy as np
import nibabel as nib

def extract_slices(mha_path, output_dir, prefix, is_mask=False, desired_size=(384, 384)):
    try:
        itk_image = sitk.ReadImage(mha_path)
        img_array = sitk.GetArrayFromImage(itk_image)
    except Exception as e:
        print(f"Error reading image {mha_path}: {e}")
        return
    for i in range(img_array.shape[2]):
        slice = img_array[:, :, i]

        if is_mask:
            slice = slice * 255

        slice_padded = pad_image(slice, desired_size)
        slice_padded = np.rot90(slice_padded, 3)  # Rotate 270 degrees (equivalent to -90 degrees)
        slice_padded = np.fliplr(slice_padded)   # Mirror the image
        slice_padded = np.flipud(slice_padded)

        nii_img = nib.Nifti1Image(slice_padded, affine=np.eye(4))
        img_filename = f"{prefix}_{str(i).zfill(4)}.nii"
        nib.save(nii_img, os.path.join(output_dir, img_filename))

def pad_image(array, desired_size):
    delta_width = desired_size[1] - array.shape[1]
    delta_height = desired_size[0] - array.shape[0]
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)

    return np.pad(array, ((top, bottom), (left, right)), 'constant', constant_values=0)

def run_slice_extraction(train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, output_train_image_dir, output_train_mask_dir, output_val_image_dir, output_val_mask_dir):
    val_image_files = sorted([f for f in os.listdir(val_images_dir) if f.endswith('.mha')], key=lambda x: x.lower())
    val_mask_files = sorted([f for f in os.listdir(val_masks_dir) if f.endswith('.nii.gz') or f.endswith('.mha')], key=lambda x: x.lower())

    assert len(val_image_files) == len(val_mask_files), "Number of images and masks does not match."

    for img_file, mask_file in zip(val_image_files, val_mask_files):
        img_name = os.path.splitext(img_file)[0]
        mask_name = os.path.splitext(os.path.splitext(mask_file)[0])[0] if mask_file.endswith('.nii.gz') else os.path.splitext(mask_file)[0]

        print(f"Processing validation image: {img_name}")
        print(f"Corresponding validation mask: {mask_name}")

        assert img_name == mask_name, f"Image and mask do not match: {img_name}, {mask_name}"

        img_mha_path = os.path.join(val_images_dir, img_file)
        print(f"Reading validation image file: {img_mha_path}")
        extract_slices(img_mha_path, output_val_image_dir, img_name)

        mask_mha_path = os.path.join(val_masks_dir, mask_file)
        print(f"Reading validation mask file: {mask_mha_path}")
        extract_slices(mask_mha_path, output_val_mask_dir, mask_name, is_mask=True)

    train_image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.mha')], key=lambda x: x.lower())
    train_mask_files = sorted([f for f in os.listdir(train_masks_dir) if f.endswith('.nii.gz') or f.endswith('.mha')], key=lambda x: x.lower())

    assert len(train_image_files) == len(train_mask_files), "Number of images and masks does not match."

    for img_file, mask_file in zip(train_image_files, train_mask_files):
        img_name = os.path.splitext(img_file)[0]
        mask_name = os.path.splitext(os.path.splitext(mask_file)[0])[0] if mask_file.endswith('.nii.gz') else os.path.splitext(mask_file)[0]

        print(f"Processing training image: {img_name}")
        print(f"Corresponding training mask: {mask_name}")

        assert img_name == mask_name, f"Image and mask do not match: {img_name}, {mask_name}"

        img_mha_path = os.path.join(train_images_dir, img_file)
        print(f"Reading training image file: {img_mha_path}")
        extract_slices(img_mha_path, output_train_image_dir, img_name)

        mask_mha_path = os.path.join(train_masks_dir, mask_file)
        print(f"Reading training mask file: {mask_mha_path}")
        extract_slices(mask_mha_path, output_train_mask_dir, mask_name, is_mask=True)
