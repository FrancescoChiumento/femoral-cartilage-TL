# Francesco Chiumento, 2024

import os
import SimpleITK as sitk

source_folder = r'your_path_here'

destination_folder = r'your_path_here'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for filename in os.listdir(source_folder):
    if filename.endswith('.nii'):
        nii_file = os.path.join(source_folder, filename)
        
        image = sitk.ReadImage(nii_file)
        
        mha_file = os.path.join(destination_folder, filename.replace('.nii', '.mha'))
        
        sitk.WriteImage(image, mha_file)
        
        print(f"Conversion completed: {nii_file} -> {mha_file}")
