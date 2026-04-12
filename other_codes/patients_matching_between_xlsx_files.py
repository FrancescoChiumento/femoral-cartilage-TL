# Francesco Chiumento, 2024

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances

# Load the Excel files
file_path_OAI = 'xlsx_file'
file_path_12 = 'xlsx_file'
data_OAI = pd.read_excel(file_path_OAI)
patients_12 = pd.read_excel(file_path_12)

# Clean the dataset by removing rows with NaN values
cleaned_data = data_OAI.dropna().copy()

# Exclude patients with KLG = 3 or 4
cleaned_data = cleaned_data[~cleaned_data['KLG'].isin([3, 4])]

# Ensure column names are in uppercase for consistency
cleaned_data.columns = map(str.upper, cleaned_data.columns)
patients_12.columns = map(str.upper, patients_12.columns)

# Initialize label encoders for categorical variables with known values
label_encoders = {
    'SEX': LabelEncoder(),
    'SIDE': LabelEncoder()
}

# Fit the encoders with all possible values
label_encoders['SEX'].fit(['Male', 'Female'])
label_encoders['SIDE'].fit(['LEFT', 'RIGHT'])

# Convert categorical variables to numeric values in the cleaned dataset
cleaned_data.loc[:, 'SEX'] = label_encoders['SEX'].transform(cleaned_data['SEX'])
cleaned_data.loc[:, 'SIDE'] = label_encoders['SIDE'].transform(cleaned_data['SIDE'])

# Convert categorical variables to numeric values in the DataFrame of 12 patients
patients_12.loc[:, 'SEX'] = label_encoders['SEX'].transform(patients_12['SEX'])
patients_12.loc[:, 'SIDE'] = label_encoders['SIDE'].transform(patients_12['SIDE'])

# Select relevant columns for matching
columns_for_matching = ['SEX', 'BMI', 'AGE', 'SIDE']
data_matching = cleaned_data[columns_for_matching].copy()
patients_12_matching = patients_12[columns_for_matching].copy()

# Calculate Euclidean distances ensuring the data is of type float
distances = pairwise_distances(patients_12_matching.astype(float), data_matching.astype(float), metric='euclidean')

# Find the indices of the closest matches without duplicates
closest_matches_indices = []
used_indices = set()

for i in range(distances.shape[0]):
    row_distances = distances[i]
    for idx in row_distances.argsort():
        if idx not in used_indices:
            closest_matches_indices.append(idx)
            used_indices.add(idx)
            break

# Retrieve the most similar patients from the main dataset
closest_matches = cleaned_data.iloc[closest_matches_indices]

# Display the matched patients
print(closest_matches)

# Save the matched patients to an Excel file
output_file_path = 'xlsx_file'
closest_matches.to_excel(output_file_path, index=False)
