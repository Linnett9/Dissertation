import os
import shutil
import csv
import re
# Define paths
csv_file_path = os.path.expanduser('~/CS5099_final_code/data/CMMD/CMMD_clinicaldata_revision.csv')
training_dir = '/data/bl70/CMMD/CMMD/augmented/CMMD/CL2.0_TG16x16_WN0.01'
benign_dir = os.path.join(training_dir, 'Benign')
malignant_dir = os.path.join(training_dir, 'Malignant')

# Ensure Benign and Malignant directories exist
os.makedirs(benign_dir, exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)

# Create a mapping from patient IDs to classifications
patient_classification = {}

with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        patient_id = row['ID1']
        classification = row['classification'].strip().capitalize()
        patient_classification[patient_id] = classification

# Create a mapping from patient IDs to classifications
patient_classification = {}

with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        patient_id = row['ID1']
        classification = row['classification'].strip().capitalize()
        patient_classification[patient_id] = classification

# Regular expression to extract the patient ID (Dx-xxxx)
patient_id_pattern = re.compile(r'D\d-\d{4}')

# Process the images in the training directory
for filename in os.listdir(training_dir):
    if filename.endswith('.png'):
        # Search for the patient ID in the filename
        match = patient_id_pattern.search(filename)
        if match:
            patient_id = match.group(0)
            if patient_id in patient_classification:
                classification = patient_classification[patient_id]
                if classification == 'Benign':
                    target_dir = benign_dir
                elif classification == 'Malignant':
                    target_dir = malignant_dir
                else:
                    print(f"Unknown classification '{classification}' for patient ID '{patient_id}'. Skipping.")
                    continue

                source_path = os.path.join(training_dir, filename)
                destination_path = os.path.join(target_dir, filename)

                # Move the file to the appropriate directory
                shutil.move(source_path, destination_path)
                print(f"Moved {filename} to {target_dir}")
            else:
                print(f"Patient ID '{patient_id}' not found in classification data. Skipping.")
        else:
            print(f"No valid patient ID found in filename '{filename}'. Skipping.")

print("Processing complete.")