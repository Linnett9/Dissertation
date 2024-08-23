import os
import csv
import shutil

# Define paths
output_base_path = '/data/bl70/CBIS-DDSM'
cbis_base_path = '/data/bl70/CBIS-DDSM/CBIS-DDSM/augmented/CBIS-DDSM'
cbis_csv_path = os.path.expanduser('~/CS5099_final_code/data/CBIS-DDSM')  # Path to the CSV files

# Create output directories if they don't exist
for dataset in ['Training', 'Testing']:
    for category in ['Benign', 'Malignant']:
        os.makedirs(os.path.join(output_base_path, dataset, category), exist_ok=True)

def process_cbis_csv(csv_file, is_test=False):
    patient_data = {}
    csv_full_path = os.path.join(cbis_csv_path, csv_file)
    if not os.path.exists(csv_full_path):
        print(f"CSV file not found: {csv_full_path}")
        return patient_data
    
    with open(csv_full_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row['patient_id']
            pathology = row['pathology']
            patient_data[patient_id] = pathology.upper() == 'MALIGNANT'
    return patient_data

def copy_cbis_files(src_dir, dest_dir, csv_file, is_test=False):
    patient_data = process_cbis_csv(csv_file, is_test)
    if not patient_data:
        print(f"No patient data found for {csv_file}. Skipping...")
        return
    
    for patient_id, is_malignant in patient_data.items():
        category = 'Malignant' if is_malignant else 'Benign'
        matching_files = [f for f in os.listdir(src_dir) if patient_id in f]
        for filename in matching_files:
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, category, filename)
            if not os.path.exists(src_path):
                print(f"Source file not found: {src_path}")
                continue
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} to {dest_path}")

def process_cbis_data():
    # Loop through each subdirectory under the base path
    for subdir in os.listdir(cbis_base_path):
        subdir_path = os.path.join(cbis_base_path, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir_path}")
            
            # Copy CBIS-DDSM training data
            for csv_file in ['calc_case_description_train_set.csv', 'mass_case_description_train_set.csv']:
                copy_cbis_files(subdir_path, os.path.join(output_base_path, 'Training'), csv_file)

            # Copy CBIS-DDSM testing data
            for csv_file in ['calc_case_description_test_set.csv', 'mass_case_description_test_set.csv']:
                copy_cbis_files(subdir_path, os.path.join(output_base_path, 'Testing'), csv_file, is_test=True)

# Execute the function
process_cbis_data()

print("Processing complete.")
