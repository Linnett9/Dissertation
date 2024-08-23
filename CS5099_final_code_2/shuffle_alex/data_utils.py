import os
import numpy as np
from sklearn.model_selection import GroupKFold
from torchvision import datasets, transforms
import torch
import pandas as pd

print('data_utils.py loaded', flush=True)

def extract_patient_id(filename):
    print(f'Extracting patient ID from filename: {filename}', flush=True)
    basename = os.path.basename(filename)
    if basename.startswith("D"):  # Assuming this is a CMMD file
        patient_id = basename[:7]
    elif basename.startswith("P"):  # Assuming this is a CBIS-DDSM file
        start_index = basename.find("P_")
        end_index = start_index + 7  # Assuming 'P_xxxxx' format
        patient_id = basename[start_index:end_index]
    else:
        raise ValueError("Filename does not match expected patterns")
    print(f'Extracted patient ID: {patient_id}', flush=True)
    return patient_id

def save_fold_patient_ids(output_dir, fold_index, train_patient_ids, val_patient_ids):
    """
    Save the training and validation patient IDs for a given fold to a CSV file.
    """
    print(f'Saving fold patient IDs for fold {fold_index + 1}', flush=True)
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.DataFrame({'patients_id': train_patient_ids, 'set': 'train'})
    val_df = pd.DataFrame({'patients_id': val_patient_ids, 'set': 'validation'})
    
    fold_df = pd.concat([train_df, val_df])
    
    output_file = os.path.join(output_dir, f'fold_{fold_index + 1}_patients_ids2.csv')
    fold_df.to_csv(output_file, index=False)
    print(f"Patients IDs for fold {fold_index + 1} saved to {output_file}", flush=True)

def create_datasets_kfold(directories, fold_index, n_splits=2, seed=42, output_dir=None):
    print('Creating datasets with KFold', flush=True)
    # Transformations for training/validation datasets
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the datasets
    datasets_list = []
    filenames = []
    labels = []

    for directory in directories:
        print(f'Loading dataset from directory: {directory}', flush=True)
        dataset = datasets.ImageFolder(root=directory, transform=transform)
        datasets_list.append(dataset)
        filenames += [os.path.join(directory, fname) for fname, _ in dataset.imgs]
        labels += [label for _, label in dataset.imgs]

    if not datasets_list:
        raise ValueError("No valid datasets found")

    combined_dataset = torch.utils.data.ConcatDataset(datasets_list)

    # Extract patient IDs
    print('Extracting patient IDs', flush=True)
    patient_ids = np.array([extract_patient_id(fname) for fname in filenames])

    # Perform GroupKFold
    print('Performing GroupKFold', flush=True)
    kf = GroupKFold(n_splits=n_splits)
    train_indices, val_indices = list(kf.split(filenames, labels, groups=patient_ids))[fold_index]

    # Create subset datasets for train and validation
    train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)

    # Save patient IDs to CSV if output_dir is provided
    if output_dir:
        print(f'Saving patient IDs to {output_dir}', flush=True)
        train_patient_ids = patient_ids[train_indices]
        val_patient_ids = patient_ids[val_indices]
        save_fold_patient_ids(output_dir, fold_index, train_patient_ids, val_patient_ids)

    print('Datasets created', flush=True)
    return train_dataset, val_dataset

def load_test_dataset(test_dirs):
    """
    Load and normalise the test dataset from multiple directories using the same transformations
    as used in the training/validation datasets.
    """
    print('Loading test dataset', flush=True)
    # Transformations for the test dataset (same as for train/val)
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the datasets from multiple directories
    datasets_list = []
    for directory in test_dirs:
        print(f'Loading test dataset from directory: {directory}', flush=True)
        dataset = datasets.ImageFolder(root=directory, transform=transform)
        datasets_list.append(dataset)
    print('datasets loaded', flush=True)
    # Combine the datasets if there are multiple directories
    if len(datasets_list) > 1:
        test_dataset = torch.utils.data.ConcatDataset(datasets_list)
    else:
        test_dataset = datasets_list[0]
    
    print('Test dataset loaded', flush=True)
    return test_dataset