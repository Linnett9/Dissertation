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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from torch.utils.data import DataLoader, Subset
import optuna
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns

# Custom functions and modules
from plots import (plot_learning_curves, plot_confusion_matrix, plot_roc_curve, 
                   plot_precision_recall_curve, plot_class_wise_accuracy, 
                   plot_train_vs_val_accuracy, plot_train_vs_val_loss, 
                   save_summary_and_report, save_best_configuration)
from transfer_model import (transfer_model_to_gpu, transfer_tuner_results_to_gpu, 
                             REMOTE_TUNER_RESULTS_DIR, REMOTE_MODEL_SAVE_DIR)

# Define paths
train_dir='/data/bl70/CNNTesting/NewTrainingImages/NewTrainingImages'
output_dir = os.path.expanduser('~/Documents/ShuffleResHardAlt/Data')
local_model_dir = os.path.expanduser('~/Downloads/ShuffleNetHardAlt/Data')
tuner_results_dir = os.path.expanduser('~/Downloads/ShuffleNetHardAlt')

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
            self.dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7, step=0.1)
            self.l2_strength = trial.suggest_float('l2_strength', 1e-5, 1e-2, log=True)
            self.l1_strength = trial.suggest_float('l1_strength', 1e-5, 1e-2, log=True)
            self.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            self.num_dense_units = trial.suggest_int('num_dense_units', 128, 1024, step=128)
            self.activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
            self.optimizer = trial.suggest_categorical('optimizer', ['adam', 'RMSprop'])
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

def create_datasets(train_dir, batch_size):
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
        self.resnet = timm.create_model('resnet18', pretrained=True, num_classes=0)

        self.first_classifier = nn.Linear(self.shufflenet_features, 1)

        self.cond_activation_threshold = hp.cond_activation_threshold
        self.cond_activation_weight = hp.cond_activation_weight

        combined_features = self.shufflenet_features + self.resnet.num_features
        self.fc1 = nn.Linear(combined_features, hp.filters)
        self.dropout = nn.Dropout(hp.dropout_rate)
        self.fc2 = nn.Linear(hp.filters, hp.num_dense_units)
        self.final_classifier = nn.Linear(hp.num_dense_units, 1)

    def forward(self, x):
        shufflenet_features = self.shufflenet(x)
        resnet_features = self.resnet(x)
        
        first_classifier_output = torch.sigmoid(self.first_classifier(shufflenet_features))
        
        # Process each element in the batch
        for i in range(first_classifier_output.size(0)):
            if first_classifier_output[i].item() < self.cond_activation_threshold:
                resnet_features[i] = resnet_features[i] * self.cond_activation_weight
        
        combined_features = torch.cat((shufflenet_features, resnet_features), dim=1)
        
        x = self.fc1(combined_features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        final_output = torch.sigmoid(self.final_classifier(x))
        
        return final_output, first_classifier_output

    def get_all_layers(self):
        return [self.shufflenet, self.resnet, self.first_classifier, self.fc1, self.dropout, self.fc2, self.final_classifier]

class OptunaCallback:
    def __init__(self, tuner_results_dir):
        self.trial_count = 0
        self.tuner_results_dir = tuner_results_dir

    def __call__(self, study, trial):
        self.trial_count += 1
        print(f"Starting trial {self.trial_count}")
        transfer_tuner_results_to_gpu(self.tuner_results_dir)
        print(f"Finished trial {self.trial_count}")

# Define class labels before the objective function
class_labels = ['Benign', 'Malignant']

def adaptive_sampling(model, dataloader, threshold):
    hard_examples = []
    easy_examples = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to('cpu'), targets.to('cpu').float().view(-1, 1)
        with torch.no_grad():
            outputs, first_classifier_output = model(inputs)
        for i in range(len(inputs)):
            confidence = first_classifier_output[i].item()
            if confidence < threshold:
                hard_examples.append((inputs[i], targets[i]))
            else:
                easy_examples.append((inputs[i], targets[i]))
    return hard_examples, easy_examples

def objective(trial):
    hp = Hyperparameters(trial)
    set_seed(hp.seed)

    criterion = nn.BCELoss()

    # Mapping optimizer names to their corresponding classes
    optimizer_mapping = {
        'adam': optim.Adam,
        'RMSprop': optim.RMSprop
    }

    # Select the optimizer class based on the hyperparameter
    optimizer_class = optimizer_mapping[hp.optimizer]
    
    train_dataset = create_datasets(train_dir, hp.batch_size)
    kf = GroupKFold(n_splits=5)

    filenames = np.array([item[0] for item in train_dataset.samples])
    labels = np.array(train_dataset.targets)
    patient_ids = np.array([os.path.basename(fname)[:7] for fname in filenames])

    best_val_accuracy = 0
    best_model_state = None

    for fold, (train_indices, val_indices) in enumerate(kf.split(filenames, labels, groups=patient_ids)):
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=hp.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=hp.batch_size, shuffle=False)
        
        model = CombinedModel(hp).to('cpu')
        optimizer = optimizer_class(model.parameters(), lr=hp.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(20):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0
            train_correct = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu').float().view(-1, 1)
                optimizer.zero_grad()
                outputs, first_classifier_output = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_correct += (outputs.round() == targets).sum().item()
            
            train_loss /= len(train_loader.dataset)
            train_accuracy = train_correct / len(train_loader.dataset)
            
            hard_examples, easy_examples = adaptive_sampling(model, train_loader, hp.cond_activation_threshold)
            
            if hard_examples:
                hard_loader = DataLoader(hard_examples, batch_size=hp.batch_size, shuffle=True)
                for inputs, targets in hard_loader:
                    inputs, targets = inputs.to('cpu'), targets.to('cpu').float().view(-1, 1)
                    optimizer.zero_grad()
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_loss = 0
            val_correct = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to('cpu'), targets.to('cpu').float().view(-1, 1)
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (outputs.round() == targets).sum().item()
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            val_accuracy = val_correct / len(val_loader.dataset)
            val_preds = np.array(val_preds)
            val_preds = (val_preds > hp.classification_threshold).astype(int)
          
            # Update history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict()
            
            scheduler.step(val_loss)
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Plotting
        print(f"Plotting learning curves for fold {fold + 1}")
        plot_learning_curves(history, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        print(f"Saved learning curves plot for fold {fold + 1}")

        print(f"Plotting confusion matrix for fold {fold + 1}")
        plot_confusion_matrix(val_targets, val_preds, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        print(f"Saved confusion matrix plot for fold {fold + 1}")

        print(f"Plotting ROC curve for fold {fold + 1}")
        plot_roc_curve(val_targets, val_preds, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        print(f"Saved ROC curve plot for fold {fold + 1}")

        print(f"Plotting precision-recall curve for fold {fold + 1}")
        plot_precision_recall_curve(val_targets, val_preds, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        print(f"Saved precision-recall curve plot for fold {fold + 1}")

        report = classification_report(val_targets, val_preds, target_names=class_labels, output_dict=True, zero_division=0)

        print(f"Plotting class-wise accuracy for fold {fold + 1}")
        plot_class_wise_accuracy(report, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        print(f"Saved class-wise accuracy plot for fold {fold + 1}")

        print(f"Plotting train vs val accuracy for fold {fold + 1}")
        plot_train_vs_val_accuracy(history, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        print(f"Saved train vs val accuracy plot for fold {fold + 1}")

        print(f"Plotting train vs val loss for fold {fold + 1}")
        plot_train_vs_val_loss(history, hp.batch_size, hp.seed, fold, hp.l1_strength, hp.l2_strength, output_dir)
        print(f"Saved train vs val loss plot for fold {fold + 1}")

        print(f"Saving summary and report for fold {fold + 1}")
        save_summary_and_report(hp.batch_size, hp.seed, fold, epoch_duration, val_loss, val_accuracy, best_val_accuracy, val_targets, val_preds, report, model, hp.l1_strength, hp.l2_strength, hp, output_dir)
        print(f"Saved summary and report for fold {fold + 1}")

    trial.set_user_attr('best_model_state', best_model_state)  # Attach the best model state to the trial
    return best_val_accuracy

# Verify the output directory
if not os.path.exists(output_dir):
    print(f"Output directory {output_dir} does not exist. Creating it.")
    os.makedirs(output_dir)

if not os.access(output_dir, os.W_OK):
    raise Exception(f"Output directory {output_dir} is not writable")
else:
    print(f"Output directory {output_dir} is writable")

study = optuna.create_study(direction='maximize', pruner=MedianPruner())
optuna_callback = OptunaCallback(tuner_results_dir)
study.optimize(objective, n_trials=50, callbacks=[optuna_callback])

# Retrieve the best model state
best_trial = study.best_trial
best_model_state = best_trial.user_attrs['best_model_state']

# Create the best model and load the best state
best_model = CombinedModel(Hyperparameters())
best_model.load_state_dict(best_model_state)

# Save the best model
best_model_path = os.path.join(local_model_dir, 'best_model.pth')
torch.save(best_model.state_dict(), best_model_path)

# Transfer best model and results
transfer_model_to_gpu(best_model_path, os.path.join(REMOTE_MODEL_SAVE_DIR, 'best_model.pth'))
transfer_tuner_results_to_gpu(tuner_results_dir)