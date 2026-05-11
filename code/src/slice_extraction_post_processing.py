# Francesco Chiumento, 2024

import SimpleITK as sitk
import os
import numpy as np
import shutil
import nibabel as nib

def extract_slices(mha_path, output_dir, prefix, is_mask=False, desired_size=(384, 384)):
    itk_image = sitk.ReadImage(mha_path)
    img_array = sitk.GetArrayFromImage(itk_image)
    affine = itk_image.GetDirection()

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
        
    return affine

def pad_image(array, desired_size):
    delta_width = desired_size[1] - array.shape[1]
    delta_height = desired_size[0] - array.shape[0]
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)

    return np.pad(array, ((top, bottom), (left, right)), 'constant', constant_values=0)

def process_directory(image_mha_path, mask_nii_path, output_image_dir, output_mask_dir, use_masks=True):
    # Create the directory for images if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Processing image: {image_mha_path}")
    image_affine = extract_slices(image_mha_path, output_image_dir, os.path.splitext(os.path.basename(image_mha_path))[0], is_mask=False)

    if use_masks:
        os.makedirs(output_mask_dir, exist_ok=True)
        print(f"Processing mask: {mask_nii_path}")
        # Ensure that extract_slices function can handle both .nii.gz and .mha files
        extract_slices(mask_nii_path, output_mask_dir, os.path.splitext(os.path.basename(mask_nii_path))[0].split('.')[0], is_mask=True)

    print("Processing completed.")

    return image_affine

def clean_folder(folder_path):
    for filename in sorted(os.listdir(folder_path), key=lambda x: x.lower()):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
