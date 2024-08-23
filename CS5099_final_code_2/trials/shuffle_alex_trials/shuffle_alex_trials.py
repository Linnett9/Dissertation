import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets, models
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import timm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader, Subset
import optuna
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns
import json
import subprocess
import sys

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Custom functions and modules
from plots import (plot_learning_curves, plot_confusion_matrix, plot_roc_curve, 
                   plot_precision_recall_curve, plot_class_wise_accuracy, 
                   plot_train_vs_val_accuracy, plot_train_vs_val_loss, 
                   save_summary_and_report, save_best_configuration)
from transfer_model import (transfer_model_to_gpu, transfer_tuner_results_to_gpu, 
                             REMOTE_TUNER_RESULTS_DIR, REMOTE_MODEL_SAVE_DIR)

# Define paths
train_dir='/data/bl70/CNNTesting/NewTrainingImages/NewTrainingImages'
output_dir = os.path.expanduser('~/Documents/ShuffleAlexPlotted2/Data')
local_model_dir = os.path.expanduser('~/Downloads/ShuffleAlexPlotted2/Data')
tuner_results_dir = os.path.expanduser('~/Downloads/TestingShuffleAlexPlotted2')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(local_model_dir, exist_ok=True)
os.makedirs(tuner_results_dir, exist_ok=True)

# Check if the directories are writable
if not os.access(output_dir, os.W_OK) or not os.access(local_model_dir, os.W_OK) or not os.access(tuner_results_dir, os.W_OK):
    raise Exception("Output or local model directory is not writable")

# Define hyperparameters
class Hyperparameters:
    def __init__(self, trial=None):
        if trial:
            self.seed = trial.suggest_int('seed', 0, 10000)
            self.dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7, step=0.1)
            self.l2_strength = trial.suggest_float('l2_strength', 1e-6, 1e-2, log=True)
            self.l1_strength = trial.suggest_float('l1_strength', 1e-6, 1e-2, log=True)
            self.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            self.num_dense_units = trial.suggest_int('num_dense_units', 128, 1024, step=128)
            self.activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
            self.optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            self.filters = trial.suggest_int('filters', 32, 128, step=32)
            self.classification_threshold = trial.suggest_float('classification_threshold', 0.1, 0.9, step=0.1)
            self.cond_activation_threshold = trial.suggest_float('cond_activation_threshold', 0.5, 0.9, step=0.1)
            self.cond_activation_weight = trial.suggest_float('cond_activation_weight', 0.1, 0.9, step=0.1)
        else:
            self.seed = 42
            self.dropout_rate = 0.5
            self.l2_strength = 1e-4
            self.l1_strength = 1e-4
            self.learning_rate = 1e-3
            self.num_dense_units = 512
            self.activation = 'relu'
            self.optimizer = 'adam'
            self.batch_size = 32
            self.filters = 64
            self.classification_threshold = 0.5
            self.cond_activation_threshold = 0.7
            self.cond_activation_weight = 0.5

# This function ensures the reproducibility of the code by setting the seed for different libraries
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_datasets(train_dir):
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    return dataset

class CombinedModel(nn.Module):
    def __init__(self, hp):
        super(CombinedModel, self).__init__()
        self.hp = hp
        self.shufflenet = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        self.shufflenet.fc = nn.Identity()  # Remove the final fully connected layer
        self.shufflenet_features = self.shufflenet.conv5[0].out_channels
        self.alexnet = models.alexnet(weights='IMAGENET1K_V1')
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])  # Remove the final fully connected layer

        # The second to last linear layer in AlexNet before the output
        alexnet_features = self.alexnet.classifier[-1].in_features if isinstance(self.alexnet.classifier[-1], nn.Linear) else self.alexnet.classifier[-2].in_features

        self.first_classifier = nn.Linear(self.shufflenet_features, 1)

        self.cond_activation_threshold = hp.cond_activation_threshold
        self.cond_activation_weight = hp.cond_activation_weight

        combined_features = self.shufflenet_features + alexnet_features
        self.fc1 = nn.Linear(combined_features, hp.filters)
        self.dropout = nn.Dropout(hp.dropout_rate)
        self.fc2 = nn.Linear(hp.filters, hp.num_dense_units)
        self.final_classifier = nn.Linear(hp.num_dense_units, 1)

    def forward(self, x):
        shufflenet_features = self.shufflenet(x)
        alexnet_features = self.alexnet(x)
        
        first_classifier_output = torch.sigmoid(self.first_classifier(shufflenet_features))
        
        modified_alexnet_features = alexnet_features.clone()  # Clone to avoid in-place modification
        for i in range(first_classifier_output.size(0)):
            if first_classifier_output[i].item() < self.cond_activation_threshold:
                modified_alexnet_features[i] = alexnet_features[i] * self.cond_activation_weight
        
        combined_features = torch.cat((shufflenet_features, modified_alexnet_features), dim=1)
        
        x = self.fc1(combined_features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        final_output = torch.sigmoid(self.final_classifier(x))
        
        return final_output, first_classifier_output

class OptunaCallback:
    def __init__(self, tuner_results_dir):
        self.trial_count = 0
        self.tuner_results_dir = tuner_results_dir

    def __call__(self, study, trial):
        self.trial_count += 1
        print(f"Starting trial {self.trial_count}")
        transfer_tuner_results_to_gpu(self.tuner_results_dir)
        print(f"Finished trial {self.trial_count}")

# Define class labels globally
class_labels = ['Benign', 'Malignant']

# Define the device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    hp = Hyperparameters(trial)
    set_seed(hp.seed)
    criterion = nn.BCELoss()
    
    if hp.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif hp.optimizer == 'rmsprop':
        optimizer_class = optim.RMSprop
    else:
        optimizer_class = optim.SGD

    dataset = create_datasets(train_dir)
    
    # Extracting patient IDs (assuming filenames contain patient IDs)
    filenames = np.array([item[0] for item in dataset.samples])
    labels = np.array(dataset.targets)
    patient_ids = np.array([os.path.basename(fname)[:7] for fname in filenames])

    best_val_accuracy = 0
    best_model_state = None
    best_epoch_duration = 0
    best_val_loss = float('inf')
    best_report_str = ""

    kf = GroupKFold(n_splits=5)
    for fold, (train_indices, val_indices) in enumerate(kf.split(filenames, labels, groups=patient_ids)):
        train_patient_ids = set(patient_ids[train_indices])
        val_patient_ids = set(patient_ids[val_indices])
        if not train_patient_ids.isdisjoint(val_patient_ids):
            raise ValueError(f"Data leakage detected: Train and validation sets contain the same patients.")

        print(f"Fold {fold + 1} Train Patient IDs: {train_patient_ids}")
        print(f"Fold {fold + 1} Validation Patient IDs: {val_patient_ids}")
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=hp.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=hp.batch_size, shuffle=False, num_workers=4)

        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

        # Initialize a new model for each fold
        model = CombinedModel(hp).to(device)
        optimizer = optimizer_class(model.parameters(), lr=hp.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        for epoch in range(10):
            epoch_start_time = time.time()
            
            model.train()
            train_loss = 0
            correct_train_preds = 0
            total_train_preds = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                
                preds = (outputs > hp.classification_threshold).int()
                correct_train_preds += (preds == targets.int()).sum().item()
                total_train_preds += targets.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_accuracy = correct_train_preds / total_train_preds
            history['accuracy'].append(train_accuracy)
            history['loss'].append(train_loss)
            
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            val_preds = np.array(val_preds)
            val_preds = (val_preds > hp.classification_threshold).astype(int)
            val_accuracy = accuracy_score(val_targets, val_preds)
            history['val_accuracy'].append(val_accuracy)
            history['val_loss'].append(val_loss)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict()
                best_epoch_duration = time.time() - epoch_start_time
                best_val_loss = val_loss
                true_classes = val_targets
                predicted_classes = val_preds
                best_report_str = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0)
            
            scheduler.step(val_loss)
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Fold {fold + 1}, Epoch {epoch+1}, Duration: {epoch_duration:.2f} seconds, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Plot learning curves
        plot_learning_curves(history, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        plot_train_vs_val_accuracy(history, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        plot_train_vs_val_loss(history, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)

    save_summary_and_report(
        hp.batch_size, hp.seed, best_epoch_duration, best_val_loss, best_val_accuracy,
        true_classes, predicted_classes, best_report_str, model, hp.dropout_rate, hp.learning_rate, 
        hp.num_dense_units, hp.activation, hp.optimizer, hp.filters, hp.classification_threshold, 
        hp.cond_activation_threshold, hp.cond_activation_weight, hp.l1_strength, hp.l2_strength, 
        output_dir, class_labels
    )

    trial.set_user_attr('best_model_state', best_model_state)
    trial.set_user_attr('best_config', hp.__dict__)
    return best_val_accuracy


study = optuna.create_study(direction='maximize', pruner=MedianPruner())
optuna_callback = OptunaCallback(tuner_results_dir)

try:
    study.optimize(objective, n_trials=50, callbacks=[optuna_callback])
except Exception as e:
    print(f"An error occurred during optimization: {e}")

best_trial = study.best_trial
best_model_state = best_trial.user_attrs['best_model_state']
best_config = best_trial.user_attrs['best_config']

best_model = CombinedModel(Hyperparameters())
state_dict = best_model_state
model_state_dict = best_model.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
model_state_dict.update(state_dict)
best_model.load_state_dict(model_state_dict)

best_model_path = os.path.join(local_model_dir, 'best_model.pth')
torch.save(best_model.state_dict(), best_model_path)

transfer_model_to_gpu(best_model_path, os.path.join(REMOTE_MODEL_SAVE_DIR, 'best_model.pth'))
transfer_tuner_results_to_gpu(tuner_results_dir)

save_best_configuration(best_config, study.best_value, output_dir)

# For final evaluation
dataset = create_datasets(train_dir)
labels = np.array([sample[1] for sample in dataset.samples])
filenames = np.array([sample[0] for sample in dataset.samples])
patient_ids = np.array([os.path.basename(fname)[:7] for fname in filenames])

if len(filenames) != len(labels) or len(filenames) != len(patient_ids):
    raise ValueError(f"Inconsistent lengths: filenames ({len(filenames)}), labels ({len(labels)}), patient_ids ({len(patient_ids)})")

kf = GroupKFold(n_splits=5)

true_classes = []
predicted_classes = []

for fold, (train_indices, val_indices) in enumerate(kf.split(filenames, labels, groups=patient_ids)):
    val_subset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=best_config['batch_size'], shuffle=False, num_workers=4)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            outputs, _ = best_model(inputs)
            true_classes.extend(targets.cpu().numpy())
            predicted_classes.extend((outputs.cpu().numpy() > best_config['classification_threshold']).astype(int))

true_classes = np.array(true_classes)
predicted_classes = np.array(predicted_classes)

# Save classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0)
save_summary_and_report(
    best_config['batch_size'], best_config['seed'], best_epoch_duration, best_val_loss, best_val_accuracy,
    true_classes, predicted_classes, report, best_model, best_config['dropout_rate'], best_config['learning_rate'], 
    best_config['num_dense_units'], best_config['activation'], best_config['optimizer'], best_config['filters'], best_config['classification_threshold'], 
    best_config['cond_activation_threshold'], best_config['cond_activation_weight'], best_config['l1_strength'], best_config['l2_strength'], 
    output_dir, class_labels
)

# Plot final evaluation results
plot_confusion_matrix(true_classes, predicted_classes, best_config["batch_size"], best_config["seed"], 0, best_config["l1_strength"], best_config["l2_strength"], output_dir)
plot_roc_curve(true_classes, predicted_classes, best_config["batch_size"], best_config["seed"], 0, best_config["l1_strength"], best_config["l2_strength"], output_dir)
plot_precision_recall_curve(true_classes, predicted_classes, best_config["batch_size"], best_config["seed"], 0, best_config["l1_strength"], best_config["l2_strength"], output_dir)
plot_class_wise_accuracy(report, best_config["batch_size"], best_config["seed"], 0, best_config["l1_strength"], best_config["l2_strength"], output_dir)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Save environment information
with open(os.path.join(output_dir, 'requirements.txt'), 'w') as f:
    f.write(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode())
