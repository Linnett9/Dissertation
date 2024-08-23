print("Starting Optuna Shuffle", flush=True)

import os
import torch
import optuna
from optuna.pruners import MedianPruner
from hyperparameters import Hyperparameters
from final_training_and_evaluation import final_training_and_evaluation
from train_ddp import train_single_fold  # Import the single fold training function
from utils import set_seed
from data_utils import create_datasets_kfold
from plot_call import generate_plots 
from test_plots import generate_test_plots
from test_best_model import test_model_on_test_set
import multiprocessing as mp

# Define paths
train_dir = '/data/bl70/CBIS-DDSM/Training'
test_dir = '/data/bl70/CBIS-DDSM/Testing'
output_dir = '/data/bl70/CBIS-DDSM/Data' 
local_model_dir = 'data/bl70/CBIS-DDSM/Models'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(local_model_dir, exist_ok=True)
print("imported all modules")
def objective(trial):
    hp = Hyperparameters(trial)
    n_splits = 2  # Number of splits for k-fold cross-validation
    best_val_accuracy = 0.0
    best_fold_index = 0

    for fold_index in range(n_splits):
        train_dataset, val_dataset = create_datasets_kfold(train_dir, fold_index, n_splits=n_splits, output_dir=output_dir)

        try:
            val_accuracy = train_single_fold(hp, train_dataset, val_dataset, output_dir, trial)

            trial.report(val_accuracy, fold_index)

            if trial.should_prune():
                print(f"Trial pruned at fold {fold_index + 1}. Stopping all folds for this trial.")
                raise optuna.exceptions.TrialPruned()

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_fold_index = fold_index

        except optuna.exceptions.TrialPruned:
            print(f"Trial pruned at fold {fold_index + 1}. Stopping all folds for this trial.")
            raise optuna.exceptions.TrialPruned()

    # Return both the best validation accuracy and the corresponding fold index
    return best_val_accuracy, best_fold_index


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    study = optuna.create_study(
        study_name="breast_cancer_classification",
        storage="sqlite:////data/bl70/CBIS-DDSM/Data/Final_study2.db",
        load_if_exists=True,
        direction='maximize',
        pruner=MedianPruner()
    )
    print("starting optimization")
    study.optimize(objective, n_trials=1)
    
    # After Optuna completes its trials:
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_val_accuracy, best_fold_index = trial.value

    best_hp = Hyperparameters()
    for key, value in trial.params.items():
        setattr(best_hp, key, value)

    set_seed(best_hp.seed)

    # Use the best fold for final training and evaluation
    results = final_training_and_evaluation(
        train_dir, test_dir, best_hp, output_dir, local_model_dir, best_fold_index=best_fold_index
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
