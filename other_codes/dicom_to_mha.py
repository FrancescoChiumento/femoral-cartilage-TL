# Francesco Chiumento, 2024

import SimpleITK as sitk
import os

def convert_dicom_to_mha(input_folder, output_folder):
    for patient_folder in os.listdir(input_folder):
        patient_path = os.path.join(input_folder, patient_folder)
        
        if os.path.isdir(patient_path):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(patient_path)
            reader.SetFileNames(dicom_names)
            
            try:
                image = reader.Execute()
                output_file = os.path.join(output_folder, f'{patient_folder}.mha')
                sitk.WriteImage(image, output_file)
                print(f"MHA file saved: {output_file}")
            except Exception as e:
                print(f"Error converting folder {patient_path}: {e}")

# Specify the input folder and output file
input_folder = 'put your input path here'
output_folder = 'put your output path here'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

convert_dicom_to_mha(input_folder, output_folder)
