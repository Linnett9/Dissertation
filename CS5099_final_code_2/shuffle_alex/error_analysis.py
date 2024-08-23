import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch

def extract_patient_id(filename):
    basename = os.path.basename(filename)
    if basename.startswith("D1-"):  # Assuming this is a CMMD file
        return basename[:7]
    elif basename.startswith("Segmented"):  # Assuming this is a CBIS-DDSM file
        start_index = basename.find("P_")
        end_index = start_index + 7  # Assuming 'P_xxxxx' format
        return basename[start_index:end_index]
    else:
        raise ValueError("Filename does not match expected patterns")

def evaluate_and_save_errors(model, test_loader, hp, device, fold_output_dir):
    model.eval()
    test_preds = []
    test_targets = []
    patient_ids = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            outputs, _ = model(inputs)
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            
            batch_patient_ids = [extract_patient_id(test_loader.dataset.dataset.samples[i][0]) 
                                 for i in test_loader.dataset.indices]
            patient_ids.extend(batch_patient_ids)

    test_preds = np.array(test_preds)
    test_preds_binary = (test_preds > hp.classification_threshold).astype(int)
    test_targets = np.array(test_targets)

    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': test_targets.flatten(),
        'predicted_label': test_preds_binary.flatten(),
        'prediction_probability': test_preds.flatten()
    })

    misclassified_df = results_df[results_df['true_label'] != results_df['predicted_label']]

    misclassified_csv_path = os.path.join(fold_output_dir, 'misclassified_samples.csv')
    misclassified_df.to_csv(misclassified_csv_path, index=False)

    print(f"Misclassified samples saved to {misclassified_csv_path}")

    test_accuracy = accuracy_score(test_targets, test_preds_binary)
    return test_accuracy, test_targets, test_preds, test_preds_binary