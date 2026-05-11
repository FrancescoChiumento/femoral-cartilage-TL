# Francesco Chiumento, 2024

import pydicom
import os

directory_path = r"your_path_here"

for subdir in os.listdir(directory_path):
    subdir_path = os.path.join(directory_path, subdir)
    if os.path.isdir(subdir_path): 
        print(f"Processing directory: {subdir}")
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            try:
                dataset = pydicom.dcmread(file_path, force=True)  # Try to read each file as DICOM
                print(f"Available attributes in file {file}:")
                print(dir(dataset))  # Print all available DICOM attributes in the file

                # Look for the 'Laterality' tag in the DICOM dataset
                laterality = dataset.get((0x0008, 0x2218), None)
                if laterality:
                    print(f"Laterality of the file {subdir}/{file}: {laterality}")
                else:
                    print(f"No laterality information available for {subdir}/{file}")

            except Exception as e:
                print(f"Impossible to read {file} as DICOM in {subdir}: {e}")
