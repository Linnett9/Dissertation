import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from new_data_utils import create_datasets_from_patient_ids  # Import the correct function
from combine_models import CombinedModel
from early_stopping import EarlyStopping
from utils import set_seed
from hyperparameters import Hyperparameters

def final_training_and_evaluation(train_dir, test_dir, hp, output_dir, local_model_dir, patient_ids_df=None):    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_labels = ['Benign', 'Malignant']

        final_model = CombinedModel(hp).to(device)

        # Use the CSV to directly create the train/validation datasets
        train_dataset, val_dataset = create_datasets_from_patient_ids(train_dir, patient_ids_df)

        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        criterion = nn.CrossEntropyLoss()
        optimizer_class = optim.Adam if hp.optimizer == 'adam' else optim.AdamW
        optimizer = optimizer_class(final_model.parameters(), lr=hp.learning_rate, weight_decay=hp.l2_strength)

        scaler = torch.cuda.amp.GradScaler()
        early_stopping = EarlyStopping(patience=100, verbose=True)
        best_model_path = os.path.join(output_dir, 'final_model32.pth')
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        epoch_times = []
        best_val_accuracy = 0.0

        def unfreeze_layers_at_epoch(epoch):
            if epoch == 100:
                for param in final_model.shufflenet.conv5.parameters():
                    param.requires_grad = True
            elif epoch == 200:
                for param in final_model.alexnet.classifier.parameters():
                    param.requires_grad = True
            elif epoch == 200:
                for param in final_model.shufflenet.parameters():
                    param.requires_grad = True
            optimizer = optimizer_class(filter(lambda p: p.requires_grad, final_model.parameters()), lr=hp.learning_rate, weight_decay=hp.l2_strength)

        for epoch in range(200):  # Increase if needed
            start_time = time.time()

            unfreeze_layers_at_epoch(epoch)

            final_model.train()
            train_loss = 0
            train_correct = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = final_model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == targets).sum().item()

            train_loss /= len(train_loader.dataset)
            train_accuracy = train_correct / len(train_loader.dataset)

            final_model.eval()
            val_loss = 0
            val_correct = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = final_model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == targets).sum().item()
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            val_loss /= len(val_loader.dataset)
            val_accuracy = val_correct / len(val_loader.dataset)

            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}", flush=True)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save({
                    'model_state_dict': final_model.state_dict(),
                    'hyperparameters': hp.__dict__,
                }, best_model_path)
                print(f"Validation accuracy increased ({best_val_accuracy:.4f}). Saving model...")

            early_stopping(val_accuracy, final_model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return history, val_targets, val_preds, epoch_times, val_loss, val_accuracy, final_model, class_labels

    except Exception as e:
        print(f"An error occurred during final training and evaluation: {e}")
        return {}, [], [], [], None, None, None, []
