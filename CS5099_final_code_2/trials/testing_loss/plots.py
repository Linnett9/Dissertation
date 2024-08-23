import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import os
import numpy as np


# Set a consistent style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 18})

output_dir = os.path.expanduser('~/Downloads/CNNTestingLossFunction/CNNDataValAccMax')

class_labels = ['Benign', 'Malignant']  # Assuming binary classification

def plot_learning_curves(history, loss_function, seed, fold, learning_rate, output_dir):
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='black')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='black', linestyle='--')
    plt.title(f'Learning Curves - Accuracy (Loss Function: {loss_function}, Seed {seed}, Fold {fold + 1}, LR {learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'learning_curves_accuracy_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.png'), dpi=300)
    plt.close()

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='black')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='black', linestyle='--')
    plt.title(f'Learning Curves - Loss (Loss Function: {loss_function}, Seed {seed}, Fold {fold + 1}, LR {learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'learning_curves_loss_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(best_true_classes, predicted_classes, loss_function, seed, fold, learning_rate, output_dir):
    cm = confusion_matrix(best_true_classes, predicted_classes)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, 
                cbar_kws={'label': 'Percentage'}, annot_kws={"size": 18})

    plt.title(f'Confusion Matrix (Loss Function: {loss_function}, Seed {seed}, Fold {fold + 1}, LR {learning_rate})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.png'), dpi=300)
    plt.close()

def plot_roc_curve(true_classes, predictions, loss_function, seed, fold, learning_rate, output_dir):
    if len(class_labels) == 2:  # Binary classification case
        fpr, tpr, _ = roc_curve(true_classes, predictions[:, 0])  # Use predictions[:, 0] for binary classification
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (Loss Function: {loss_function}, Seed {seed}, Fold {fold + 1}, LR {learning_rate})')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.png'), dpi=300)
        plt.close()

def plot_precision_recall_curve(true_classes, predictions, loss_function, seed, fold, learning_rate, output_dir):
    if len(class_labels) == 2:  # Binary classification case
        precision, recall, _ = precision_recall_curve(true_classes, predictions[:, 0])  # Use predictions[:, 0] for binary classification
        average_precision = average_precision_score(true_classes, predictions[:, 0])
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='black', lw=2, label=f'Precision-Recall curve (area = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve (Loss Function: {loss_function}, Seed {seed}, Fold {fold + 1}, LR {learning_rate})')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'precision_recall_curve_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.png'), dpi=300)
        plt.close()

def plot_class_wise_accuracy(report, loss_function, seed, fold, learning_rate, output_dir):
    class_accuracies = [report[class_name]['precision'] if class_name in report else 0 for class_name in class_labels]
    plt.figure(figsize=(10, 6))
    
    # Define colors and hatches
    colors = ['black', 'white']
    hatches = ['', '//']
    
    # Plot bars with custom colors and hatches
    bars = plt.bar(class_labels, class_accuracies, color=colors, edgecolor='black')
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    plt.ylim(0, 1.0)  # 1.0 corresponds to 100%

    plt.xlabel('Classes')
    plt.ylabel('Precision')
    plt.title(f'Class-wise Precision (Loss Function: {loss_function}, Seed {seed}, Fold {fold + 1}, LR {learning_rate})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'class_wise_accuracy_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.png'), dpi=300)
    plt.close()

def plot_train_vs_val_accuracy(history, loss_function, seed, fold, learning_rate, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='black')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='black', linestyle='--')
    plt.title(f'Training vs Validation Accuracy (Loss Function: {loss_function}, Seed {seed}, Fold {fold + 1}, LR {learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'train_val_accuracy_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.png'), dpi=300)
    plt.close()

def plot_train_vs_val_loss(history, loss_function, seed, fold, learning_rate, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='black')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='black', linestyle='--')
    plt.title(f'Training vs Validation Loss (Loss Function: {loss_function}, Seed {seed}, Fold {fold + 1}, LR {learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'train_val_loss_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.png'), dpi=300)
    plt.close()

def save_summary_and_report(loss_function, seed, fold, learning_rate, training_time, val_loss, val_accuracy, best_val_accuracy, true_classes, predicted_classes, report, model, output_dir):
    # Save the summary
    with open(os.path.join(output_dir, f'summary_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}.txt'), 'w') as f:
        f.write(f'Loss Function: {loss_function}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Fold: {fold + 1}\n')
        f.write(f'Learning Rate: {learning_rate}\n')
        f.write(f'Training Time: {training_time:.2f} seconds\n')
        f.write(f'Validation Loss: {val_loss}\n')
        f.write(f'Validation Accuracy: {val_accuracy * 100:.2f}%\n')
        f.write(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%\n')
        f.write('\nClassification Report:\n')
        f.write(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))
        f.write('\nConfiguration:\n')
        f.write(f'Model Layers:\n')
        for layer in model.layers:
            f.write(f'{layer}\n')

def save_best_configuration(best_config, best_val_accuracy, output_dir):
    # Save the best configuration
    with open(os.path.join(output_dir, 'best_configuration.txt'), 'w') as f:
        f.write(f'Best Config: {best_config[0]}\n')
        f.write(f'Seed: {best_config[1]}\n')
        f.write(f'Fold: {best_config[2] + 1}\n')
        f.write(f'Learning Rate: {best_config[3]}\n')
        f.write(f'Validation Accuracy: {best_val_accuracy * 100:.2f}%\n')