# Francesco Chiumento, 2024

from slice_extraction_post_processing import process_directory, clean_folder
from unet_segmentation import run_segmentation
from composition_mha import combine_slices_to_mha
from postprocessing import process_and_save_image
import numpy as np
import torch
import os

def test_main(images_dir, masks_dir, images_slices, masks_slices, checkpoint_path, predictions, segmentations, postprocessed_dir, dicom_headers_dir):
    # Define paths and settings
    if not os.path.exists(postprocessed_dir):
        os.makedirs(postprocessed_dir)
    
    should_flip = True # Option to flip the images if needed

    # Patient Directory Listing
    patient_dirs = sorted(os.listdir(images_dir), key=lambda x: x.lower())

    # Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Patient Data Processing Loop
    for patient in patient_dirs:
        base_patient = os.path.splitext(patient)[0]
        patient_id = base_patient.split('_')[-1]

        image_mha_path = os.path.join(images_dir, f"{base_patient}.mha")
        mask_nii_path = os.path.join(masks_dir, f"{base_patient}.nii.gz")

        # Cleaning the folders for slices and predictions before processing each patient
        clean_folder(images_slices)
        clean_folder(masks_slices)
        clean_folder(predictions)

        # Calling processing functions
        affine = process_directory(image_mha_path, mask_nii_path, images_slices, masks_slices, use_masks=True)
        affine = np.array(affine).reshape(3, 3)
        run_segmentation(checkpoint_path, images_slices, masks_slices, predictions, device, use_masks=True)

        # Combining slices into a .mha file using the NIfTI file as a reference for metadata
        output_mha_file = os.path.join(segmentations, f"{patient}")
        combine_slices_to_mha(predictions, output_mha_file, mask_nii_path, affine, should_flip=True)

        # Post-processing and saving the image
        process_and_save_image(str(output_mha_file), str(postprocessed_dir), None)

        # Confirmation printout for completion of the patient's process
        print(f"The process has been completed for the patient {patient}.")
