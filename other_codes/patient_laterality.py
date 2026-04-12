# Francesco Chiumento, 2024

import pydicom
import os

directory_path = r"your_path_here"
visited_directories = set()

for subdir, dirs, files in os.walk(directory_path):
    if subdir not in visited_directories:
        visited_directories.add(subdir)
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                dataset = pydicom.dcmread(file_path, force=True)
                if 'StudyDescription' in dir(dataset):
                    series_desc = dataset.StudyDescription.upper()
                    if 'LEFT' in series_desc:
                        laterality = 'left'
                    elif 'RIGHT' in series_desc:
                        laterality = 'right'
                    else:
                        laterality = "Laterality not specified"
                else:
                    laterality = "Laterality not specified"
                folder_code = os.path.basename(subdir)
                print(f"01\{folder_code}\n{laterality}")
                break 
            except Exception as e:
                print(f"Impossible to read DICOM file {file_path}: {e}")
            break
