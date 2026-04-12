# Francesco Chiumento, 2024

import os
import SimpleITK as sitk

# Path to the folders
input_folder = r'your_path_here'  # folder with segmentation files
output_folder = r'your_path_here' # folder to save files with only label 1

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to extract only label 1
def extract_label_1(nifti_file):
    img = sitk.ReadImage(nifti_file)
    data = sitk.GetArrayFromImage(img)
    
    # Create a mask with only label 1
    label_1_data = (data == 1).astype(int)
    
    # Create a new SimpleITK image
    new_img = sitk.GetImageFromArray(label_1_data)
    new_img.CopyInformation(img)
    
    return new_img

# Process all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        filepath = os.path.join(input_folder, filename)
        new_img = extract_label_1(filepath)
        
        # Save the new image in the output folder
        output_path = os.path.join(output_folder, filename)
        sitk.WriteImage(new_img, output_path)

print("Extraction and saving completed.")
