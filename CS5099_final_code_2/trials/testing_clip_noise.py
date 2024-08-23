import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets, models
from sklearn.model_selection import GroupKFold
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import json
import pandas as pd 

# Define paths
base_dir = '/data/bl70/validate/ProcessedImages/CMMD/DuplicateBenignSplit'
output_dir = '/data/bl70/TestingClipNoise/Data'
local_model_dir = '/data/bl70/TestingClipNoise/Data'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(local_model_dir, exist_ok=True)

# Function to get all subdirectories
def get_subdirectories(base_dir):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


# Define best hyperparameters
class Hyperparameters:
    def __init__(self):
        self.seed = 9559
        self.dropout_rate = 0.30000000000000004
        self.l2_strength = 0.0005905699029039175
        self.l1_strength = 1.0275630784763877e-05
        self.learning_rate = 2.2079536885355455e-05
        self.num_dense_units = 768
        self.activation = 'tanh'
        self.optimizer = 'adam'
        self.batch_size = 32
        self.filters = 96
        self.classification_threshold = 0.4
        self.cond_activation_threshold = 0.8
        self.cond_activation_weight = 0.4

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create datasets
def create_datasets(train_dir):
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    return dataset

# Define the combined model
class CombinedModel(nn.Module):
    def __init__(self, hp):
        super(CombinedModel, self).__init__()
        self.hp = hp
        self.shufflenet = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        self.shufflenet.fc = nn.Identity()
        self.shufflenet_features = self.shufflenet.conv5[0].out_channels
        self.alexnet = models.alexnet(weights='IMAGENET1K_V1')
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])

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
        
        modified_alexnet_features = alexnet_features.clone()
        for i in range(first_classifier_output.size(0)):
            if first_classifier_output[i].item() < self.cond_activation_threshold:
                modified_alexnet_features[i] = alexnet_features[i] * self.cond_activation_weight
        
        combined_features = torch.cat((shufflenet_features, modified_alexnet_features), dim=1)
        
        x = self.fc1(combined_features)
        x = torch.tanh(x)  # Using tanh as per best parameters
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.tanh(x)  # Using tanh as per best parameters
        x = self.dropout(x)
        final_output = torch.sigmoid(self.final_classifier(x))
        
        return final_output, first_classifier_output

# Define class labels and device
class_labels = ['Benign', 'Malignant']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(train_dir, variation_name):
    hp = Hyperparameters()
    set_seed(hp.seed)
    criterion = nn.BCELoss()
    
    dataset = create_datasets(train_dir)
    
    filenames = np.array([item[0] for item in dataset.samples])
    labels = np.array(dataset.targets)
    patient_ids = np.array([os.path.basename(fname)[:7] for fname in filenames])

    kf = GroupKFold(n_splits=5)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(filenames, labels, groups=patient_ids)):
        print(f"Training fold {fold + 1}/5 for variation {variation_name}")
        
        # Save patient IDs for this fold
        fold_output_dir = os.path.join(output_dir, variation_name, f'fold_{fold+1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        train_patients = patient_ids[train_indices]
        val_patients = patient_ids[val_indices]
        
        fold_df = pd.DataFrame({
            'patient_id': np.concatenate([train_patients, val_patients]),
            'set': ['train'] * len(train_patients) + ['validation'] * len(val_patients)
        })
        fold_df.to_csv(os.path.join(fold_output_dir, 'patient_ids.csv'), index=False)
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=hp.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=hp.batch_size, shuffle=False, num_workers=4)

        model = CombinedModel(hp).to(device)
        optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.l2_strength)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        best_val_accuracy = 0
        best_val_results = None
        for epoch in range(10):  # You can adjust the number of epochs
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
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
            val_preds_binary = (val_preds > hp.classification_threshold).astype(int)
            val_targets = np.array(val_targets)
            
            val_accuracy = accuracy_score(val_targets, val_preds_binary)
            val_precision = precision_score(val_targets, val_preds_binary)
            val_recall = recall_score(val_targets, val_preds_binary)
            val_f1 = f1_score(val_targets, val_preds_binary)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_results = {
                    'accuracy': val_accuracy,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1,
                    'loss': val_loss,
                    'classification_report': classification_report(val_targets, val_preds_binary, target_names=class_labels)
                }
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        fold_results.append(best_val_results)
        
        # Save fold results
        fold_output_dir = os.path.join(output_dir, variation_name, f'fold_{fold+1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        with open(os.path.join(fold_output_dir, 'results.json'), 'w') as f:
            json.dump(best_val_results, f, indent=4)
        with open(os.path.join(fold_output_dir, 'classification_report.txt'), 'w') as f:
            f.write(best_val_results['classification_report'])

    # Calculate and save average results across folds
    avg_results = {
        'accuracy': np.mean([fold['accuracy'] for fold in fold_results]),
        'precision': np.mean([fold['precision'] for fold in fold_results]),
        'recall': np.mean([fold['recall'] for fold in fold_results]),
        'f1': np.mean([fold['f1'] for fold in fold_results]),
        'loss': np.mean([fold['loss'] for fold in fold_results])
    }
    print(f"Average results across folds for {variation_name}:")
    print(json.dumps(avg_results, indent=4))
    
    variation_output_dir = os.path.join(output_dir, variation_name)
    with open(os.path.join(variation_output_dir, 'average_results.json'), 'w') as f:
        json.dump(avg_results, f, indent=4)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(local_model_dir, f'final_model_{variation_name}.pth'))
    
    return avg_results['accuracy']

if __name__ == "__main__":
    variations = get_subdirectories(base_dir)
    results = {}
    
    for variation in variations:
        print(f"Processing variation: {variation}")
        train_dir = os.path.join(base_dir, variation)
        accuracy = train_and_evaluate(train_dir, variation)
        results[variation] = accuracy
    
    # Print and save overall results
    print("\nOverall Results:")
    for variation, accuracy in results.items():
        print(f"{variation}: {accuracy:.4f}")
    
    with open(os.path.join(output_dir, 'variation_results.txt'), 'w') as f:
        for variation, accuracy in results.items():
            f.write(f"{variation}: {accuracy:.4f}\n")