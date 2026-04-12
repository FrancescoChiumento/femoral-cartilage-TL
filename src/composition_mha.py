# Francesco Chiumento, 2024

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import os

def crop_to_target(input_img_3d, target_shape):
    current_z, current_y, current_x = input_img_3d.shape
    print(f"Initial 3D image shape: (z, y, x) = ({current_z}, {current_y}, {current_x})")
    target_z, target_y = target_shape
    print(f"Target cropping dimensions: (y, x) = ({target_z}, {target_y})")

    start_z = max((current_z - target_z) // 2, 0)
    end_z = min(start_z + target_z, current_z)
    start_y = max((current_y - target_y) // 2, 0)
    end_y = min(start_y + target_y, current_y)

    print(f"Starting points for cropping: (start_z, start_x) = ({start_z}, {start_y})")
    cropped_img = input_img_3d[start_z:end_z, start_y:end_y, :]
    cropped_z, cropped_y, cropped_x = cropped_img.shape
    print(f"Cropped 3D image shape: (z, y, x) = ({cropped_z}, {cropped_y}, {cropped_x})")
    print(f"Range of intensity values after cropping: {cropped_img.min(), cropped_img.max()}")
    return cropped_img

def combine_slices_to_mha(input_dir, output_file, ground_truth_nii_path, affine, should_flip=False):
    nii_img = nib.load(ground_truth_nii_path)
    if np.linalg.det(affine) == 0:
        raise ValueError("Direction matrix has a determinant of zero, not valid")
    nii_header = nii_img.header

    target_shape = (nii_header['dim'][2], nii_header['dim'][3])
    print("Dimensions extracted from header for target_shape (y, x):", target_shape)

    target_spacing = (nii_header['pixdim'][3], nii_header['pixdim'][2], nii_header['pixdim'][1])
    
    print(f"Target shape: {target_shape}")
    print(f"Target spacing: {target_spacing}")
    
    nii_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')], key=lambda x: x.lower())
    print(f"Found {len(nii_files)} NIfTI files in {input_dir}")
    
    if not nii_files:
        raise ValueError("No NIfTI files found in the input directory.")
    
    nii_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    slices = []
    for nii in nii_files:
        img_path = os.path.join(input_dir, nii)
        img = nib.load(img_path).get_fdata()
        #if should_flip:
            #img = np.flipud(img) 
            #img = np.fliplr(img)
        slices.append(img.squeeze())

    if not slices:
        raise ValueError("No slices were loaded; the list 'slices' is empty.")

    img_3d = np.stack(slices, axis=-1)
    
    cropped_img_3d = crop_to_target(img_3d, target_shape)
    print(f"Cropped image shape: {cropped_img_3d.shape}")
    final_img_3d = np.transpose(cropped_img_3d, (0, 1, 2))
    print(f"Final image shape: {final_img_3d.shape}")
    print(f"Range of intensity values before saving: {final_img_3d.min(), final_img_3d.max()}")
    
    final_img_3d_uint8 = (final_img_3d * 255).astype(np.uint8)
    
    final_sitk_img = sitk.GetImageFromArray(final_img_3d_uint8)
    final_sitk_img.SetSpacing((float(target_spacing[2]), float(target_spacing[1]), float(target_spacing[0])))
    final_sitk_img.SetDirection(affine.flatten().tolist())
    sitk.WriteImage(final_sitk_img, output_file)
