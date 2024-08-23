import pandas as pd

def load_patient_ids_from_csv(csv_file_path):
    """
    Load patient IDs from a CSV file and split them into training and validation sets.
    """
    df = pd.read_csv(csv_file_path)
    train_patient_ids = df[df['set'] == 'train']['patients_id'].values
    val_patient_ids = df[df['set'] == 'validation']['patients_id'].values
    return train_patient_ids, val_patient_ids
