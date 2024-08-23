import os
import torch
from hyperparameters import Hyperparameters
from manual_final_training import final_training_and_evaluation
from utils import set_seed
import pandas as pd
import multiprocessing as mp
from new_data_utils import create_datasets_from_patient_ids, load_test_dataset
from plot_call import generate_plots
from test_plots import generate_test_plots
from test_best_model import test_model_on_test_set
# Define paths
train_dir = ['/raid/bl70/FinalCBIS2/Training2']
test_dir = ['/raid/bl70/FinalCBIS/Testing']
output_dir = '/raid/bl70/Data3'
local_model_dir = '/raid/bl70/Model3'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(local_model_dir, exist_ok=True)

def load_patient_ids_from_csv(csv_file):
    """
    Load patient IDs and their respective split (train/validation) from the provided CSV file.
    """
    patient_ids_df = pd.read_csv(csv_file)
    return patient_ids_df

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # Load the manually found best hyperparameters
    best_params = {
        'seed': 61819,
        'dropout_rate': 0.39834496994226454,
        'l2_strength': 2.6073512456230194e-05,
        'l1_strength': 0.00033731281533309616,
        'batch_size': 32,
        'learning_rate': 0.0002111949992007706,
        'num_dense_units': 458,
        'activation': 'relu',
        'optimizer': 'adamw',
        'filters': 35,
        'classification_threshold': 0.7572913033112828,
        'cond_activation_threshold': 0.7126080574153795,
        'cond_activation_weight': 0.6327548785043808
    }

    best_hp = Hyperparameters()
    for key, value in best_params.items():
        setattr(best_hp, key, value)

    set_seed(best_hp.seed)

    # Load the patient IDs and their train/validation split from the CSV
    patient_ids_df = load_patient_ids_from_csv('/raid/bl70/Data3/fold_5_patients_ids.csv')

    # Run the final training and evaluation using the specified splits
    results = final_training_and_evaluation(
        train_dir, test_dir, best_hp, output_dir, local_model_dir, patient_ids_df=patient_ids_df
    )

    if results:
        history, val_targets, val_preds, epoch_times, val_loss, val_accuracy, final_model, class_labels = results

        generate_plots(history, val_targets, val_preds, class_labels, best_hp, output_dir, epoch_times, val_loss, val_accuracy, final_model)

        test_loss, test_accuracy, test_preds, test_targets = test_model_on_test_set(
            test_dir, best_hp, output_dir, local_model_dir, class_labels
        )
        generate_test_plots(test_targets, test_preds, class_labels, output_dir, test_loss, test_accuracy)
    else:
        print("Final training and evaluation did not return valid results. Skipping plot generation and testing.")
