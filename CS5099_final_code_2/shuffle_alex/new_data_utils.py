import os
import numpy as np
from torchvision import datasets, transforms
import torch

def extract_patient_id(filename):
    basename = os.path.basename(filename)
    if basename.startswith("D"):  # Assuming this is a CMMD file
        return basename[:7]
    elif basename.startswith("P"):  # Assuming this is a CBIS-DDSM file
        start_index = basename.find("P_")
        end_index = start_index + 7  # Assuming 'P_xxxxx' format
        return basename[start_index:end_index]
    else:
        raise ValueError("Filename does not match expected patterns")

def create_datasets_from_patient_ids(directories, patient_ids_df):
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
        dataset = datasets.ImageFolder(root=directory, transform=transform)
        datasets_list.append(dataset)
        filenames += [os.path.join(directory, fname) for fname, _ in dataset.imgs]
        labels += [label for _, label in dataset.imgs]

    if not datasets_list:
        raise ValueError("No valid datasets found")

    combined_dataset = torch.utils.data.ConcatDataset(datasets_list)

    # Extract patient IDs
    patient_ids = np.array([extract_patient_id(fname) for fname in filenames])

    # Filter indices based on the patient IDs from the CSV
    train_patient_ids = patient_ids_df[patient_ids_df['set'] == 'train']['patients_id'].values
    val_patient_ids = patient_ids_df[patient_ids_df['set'] == 'validation']['patients_id'].values

    train_indices = np.where(np.isin(patient_ids, train_patient_ids))[0]
    val_indices = np.where(np.isin(patient_ids, val_patient_ids))[0]

    # Create subset datasets for train and validation
    train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)

    return train_dataset, val_dataset

def load_test_dataset(test_dirs):
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    datasets_list = []
    for directory in test_dirs:
        dataset = datasets.ImageFolder(root=directory, transform=transform)
        datasets_list.append(dataset)
    
    if len(datasets_list) > 1:
        test_dataset = torch.utils.data.ConcatDataset(datasets_list)
    else:
        test_dataset = datasets_list[0]
    
    return test_dataset
