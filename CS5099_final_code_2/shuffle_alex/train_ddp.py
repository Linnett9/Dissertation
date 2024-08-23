import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import optuna
from early_stopping import EarlyStopping
from utils import set_seed
from combine_models import CombinedModel

def train_single_fold(hp, train_dataset, val_dataset, output_dir, trial=None):
    set_seed(hp.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    model = CombinedModel(hp).to(device)
    
    # Freeze all layers initially
    for param in model.shufflenet.parameters():
        param.requires_grad = False
    for param in model.alexnet.parameters():
        param.requires_grad = False

    optimizer_class = optim.Adam if hp.optimizer == 'adam' else optim.AdamW
    optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.learning_rate, weight_decay=hp.l2_strength)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=3, verbose=True, delta=0.01)
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    scaler = amp.GradScaler()  # Instantiate the GradScaler
    writer = SummaryWriter(log_dir=output_dir)  # Initialize TensorBoard writer

    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    best_val_accuracy = 0.0

    for epoch in range(1):
        # Gradual unfreezing
        if epoch == 2:
            for param in model.shufflenet.conv5.parameters():
                param.requires_grad = True
        elif epoch == 4:
            for param in model.alexnet.classifier.parameters():
                param.requires_grad = True
        elif epoch == 6:
            for param in model.shufflenet.parameters():
                param.requires_grad = True

        # Re-initialize the optimizer after unfreezing layers
        optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.learning_rate, weight_decay=hp.l2_strength)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with amp.autocast():  # Enable autocasting for mixed precision
                final_output = model(inputs)
                loss = criterion(final_output, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(final_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                final_output = model(inputs)
                loss = criterion(final_output, labels)
                val_loss += loss.item()
                _, predicted = torch.max(final_output.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        val_loss /= len(val_loader)

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Log to Optuna
        if trial:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                print("Trial pruned due to insufficient performance")
                raise optuna.exceptions.TrialPruned()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': hp.__dict__,
            }, best_model_path)

        early_stopping(val_accuracy, model, best_model_path)
        if early_stopping.early_stop:
            print(f'Early stopping triggered')
            break

        scheduler.step(val_accuracy)

    writer.close()  # Close the TensorBoard writer

    return best_val_accuracy
