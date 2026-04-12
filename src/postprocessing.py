# Francesco Chiumento, 2024

import os
import SimpleITK as sitk
import numpy as np

def levelset2binary(mask_LS_itk):
    mask_LS_np = sitk.GetArrayFromImage(mask_LS_itk)
    mask_B_np = mask_LS_np > 0.0
    mask_B_np = mask_B_np.astype(np.uint8)
    mask_B_itk = sitk.GetImageFromArray(mask_B_np)
    mask_B_itk.SetSpacing(mask_LS_itk.GetSpacing())
    mask_B_itk.SetOrigin(mask_LS_itk.GetOrigin())
    mask_B_itk.SetDirection(mask_LS_itk.GetDirection())
    return mask_B_itk

def process_and_save_image(file_path, output_folder, override_label=None):
    image = sitk.ReadImage(file_path)
    labels = sitk.ConnectedComponent(image)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labels)
    label_sizes = {l: stats.GetNumberOfPixels(l) for l in stats.GetLabels() if l != 0}
    print(f"Label sizes: {label_sizes}")
    selected_label = override_label if override_label in label_sizes else max(label_sizes, key=label_sizes.get)
    print(f"Selected label: {selected_label} with {label_sizes[selected_label]} pixels")
    binary_image = sitk.BinaryThreshold(labels, lowerThreshold=selected_label, upperThreshold=selected_label, insideValue=255, outsideValue=0)
    mask = levelset2binary(binary_image)
    mask = sitk.Cast(mask, sitk.sitkInt16)

    # Save the image directly without modifying dimensions or orientation
    modified_file_path = os.path.join(output_folder, os.path.basename(file_path).replace('.mha', '_modified.mha'))
    sitk.WriteImage(mask, modified_file_path)  # Save the mask as is

    print(f"Saved modified image to {modified_file_path}")
