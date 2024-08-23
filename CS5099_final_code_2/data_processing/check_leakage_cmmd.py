import os

def get_patient_ids(directories):
    patient_ids = set()
    for directory in directories:
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    patient_id = filename[:7]
                    patient_ids.add(patient_id)
    return patient_ids

def check_unique_patient_ids(train_dirs, val_dirs, test_dirs):
    train_patient_ids = get_patient_ids(train_dirs)
    val_patient_ids = get_patient_ids(val_dirs)
    test_patient_ids = get_patient_ids(test_dirs)
    
    # Find duplicates across directories
    train_val_overlap = train_patient_ids.intersection(val_patient_ids)
    train_test_overlap = train_patient_ids.intersection(test_patient_ids)
    val_test_overlap = val_patient_ids.intersection(test_patient_ids)

    # Print results
    if train_val_overlap:
        print(f"Overlap between Training and Validation: {train_val_overlap}")
    if train_test_overlap:
        print(f"Overlap between Training and Test: {train_test_overlap}")
    if val_test_overlap:
        print(f"Overlap between Validation and Test: {val_test_overlap}")
    
    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        print("All Patient IDs are unique across directories.")
    else:
        print("There are duplicate Patient IDs across directories.")

if __name__ == "__main__":
    base_dir = '/data/bl70'
    train_dirs = [
        os.path.join(base_dir, 'TrainingImages'),
       #  os.path.join(base_dir, 'TrainingImages2'),
         #  os.path.join(base_dir, 'TrainingImages3')
    ]
    val_dirs = [
        os.path.join(base_dir, 'ValidationImages'),
       #    os.path.join(base_dir, 'ValidationImages2'),
        #   os.path.join(base_dir, 'ValidationImages3')
    ]
    test_dirs = [
        os.path.join(base_dir, 'TestImages'),
       #    os.path.join(base_dir, 'TestImages2'),
         # os.path.join(base_dir, 'TestImages3')
    ]

    check_unique_patient_ids(train_dirs, val_dirs, test_dirs)
