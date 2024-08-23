import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import os
import numpy as np


# Plotting function definitions
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 18})

class_labels = ['Benign', 'Malignant']  # Assuming binary classification

def plot_learning_curves(history, batch_size, seed, fold, l1_strength, l2_strength, output_dir):
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Train Accuracy', color='black')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='black', linestyle='--')
    plt.title(f'Learning Curves - Accuracy (Batch Size {batch_size}, Seed {seed}, Fold {fold + 1}, L1 {l1_strength}, L2 {l2_strength})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    local_path = os.path.join(output_dir, f'learning_curves_accuracy_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.png')
    plt.savefig(local_path, dpi=300)
    plt.close()

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', color='black')
    plt.plot(history['val_loss'], label='Validation Loss', color='black', linestyle='--')
    plt.title(f'Learning Curves - Loss (Batch Size {batch_size}, Seed {seed}, Fold {fold + 1}, L1 {l1_strength}, L2 {l2_strength})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    local_path = os.path.join(output_dir, f'learning_curves_loss_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.png')
    plt.savefig(local_path, dpi=300)
    plt.close()

def plot_confusion_matrix(true_classes, predicted_classes, batch_size, seed, fold, l1_strength, l2_strength, output_dir):
    cm = confusion_matrix(true_classes, predicted_classes)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, 
                cbar_kws={'label': 'Percentage'}, annot_kws={"size": 18})

    plt.title(f'Confusion Matrix (Batch Size {batch_size}, Seed {seed}, Fold {fold + 1}, L1 {l1_strength}, L2 {l2_strength})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    local_path = os.path.join(output_dir, f'confusion_matrix_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.png')
    plt.savefig(local_path, dpi=300)
    plt.close()

def plot_roc_curve(true_classes, predictions, batch_size, seed, fold, l1_strength, l2_strength, output_dir):
    if len(class_labels) == 2:  # Binary classification case
        fpr, tpr, _ = roc_curve(true_classes, predictions)  # Use predictions directly for binary classification
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (Batch Size {batch_size}, Seed {seed}, Fold {fold + 1}, L1 {l1_strength}, L2 {l2_strength})')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        local_path = os.path.join(output_dir, f'roc_curve_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.png')
        plt.savefig(local_path, dpi=300)
        plt.close()

def plot_precision_recall_curve(true_classes, predictions, batch_size, seed, fold, l1_strength, l2_strength, output_dir):
    if len(class_labels) == 2:  # Binary classification case
        precision, recall, _ = precision_recall_curve(true_classes, predictions)  # Use predictions directly for binary classification
        average_precision = average_precision_score(true_classes, predictions)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='black', lw=2, label=f'Precision-Recall curve (area = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve (Batch Size {batch_size}, Seed {seed}, Fold {fold + 1}, L1 {l1_strength}, L2 {l2_strength})')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.tight_layout()
        local_path = os.path.join(output_dir, f'precision_recall_curve_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.png')
        plt.savefig(local_path, dpi=300)
        plt.close()

def plot_class_wise_accuracy(report, batch_size, seed, fold, l1_strength, l2_strength, output_dir):
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
    plt.title(f'Class-wise Precision (Batch Size {batch_size}, Seed {seed}, Fold {fold + 1}, L1 {l1_strength}, L2 {l2_strength})')
    plt.tight_layout()
    local_path = os.path.join(output_dir, f'class_wise_accuracy_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.png')
    plt.savefig(local_path, dpi=300)
    plt.close()

def plot_train_vs_val_accuracy(history, batch_size, seed, fold, l1_strength, l2_strength, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Train Accuracy', color='black')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='black', linestyle='--')
    plt.title(f'Training vs Validation Accuracy (Batch Size {batch_size}, Seed {seed}, Fold {fold + 1}, L1 {l1_strength}, L2 {l2_strength})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    local_path = os.path.join(output_dir, f'train_val_accuracy_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.png')
    plt.savefig(local_path, dpi=300)
    plt.close()

def plot_train_vs_val_loss(history, batch_size, seed, fold, l1_strength, l2_strength, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', color='black')
    plt.plot(history['val_loss'], label='Validation Loss', color='black', linestyle='--')
    plt.title(f'Training vs Validation Loss (Batch Size {batch_size}, Seed {seed}, Fold {fold + 1}, L1 {l1_strength}, L2 {l2_strength})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    local_path = os.path.join(output_dir, f'train_val_loss_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.png')
    plt.savefig(local_path, dpi=300)
    plt.close()

def save_summary_and_report(batch_size, seed, fold, training_time, val_loss, val_accuracy, best_val_accuracy, true_classes, predicted_classes, report, model, l1_strength, l2_strength, hp, output_dir):
    # Save the summary
    summary_file_path = os.path.join(output_dir, f'summary_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}_l1_{l1_strength}_l2_{l2_strength}.txt')
    with open(summary_file_path, 'w') as f:
        f.write(f'Batch Size: {batch_size}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Fold: {fold + 1}\n')
        f.write(f'L1 Strength: {l1_strength}\n')
        f.write(f'L2 Strength: {l2_strength}\n')
        f.write(f'Training Time: {training_time:.2f} seconds\n')
        f.write(f'Validation Loss: {val_loss}\n')
        f.write(f'Validation Accuracy: {val_accuracy * 100:.2f}%\n')
        f.write(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%\n')
        f.write('\nClassification Report:\n')
        f.write(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))
        f.write('\nConfiguration:\n')
        f.write(f'Model Layers:\n')
        for layer in model.get_all_layers():
            f.write(f'{layer}\n')
        f.write('\nHyperparameters:\n')
        f.write(f'Dropout Rate: {hp.dropout_rate}\n')
        f.write(f'Learning Rate: {hp.learning_rate}\n')
        f.write(f'Number of Dense Units: {hp.num_dense_units}\n')
        f.write(f'Activation: {hp.activation}\n')
        f.write(f'Optimizer: {hp.optimizer}\n')
        f.write(f'Filters: {hp.filters}\n')
        f.write(f'Classification Threshold: {hp.classification_threshold}\n')
        f.write(f'Conditional Activation Threshold: {hp.cond_activation_threshold}\n')
        f.write(f'Conditional Activation Weight: {hp.cond_activation_weight}\n')
    print(f"Summary and report saved to {summary_file_path}")

def save_best_configuration(best_config, best_val_accuracy, output_dir):
    # Save the best configuration
    with open(os.path.join(output_dir, 'best_configuration.txt'), 'w') as f:
        f.write(f'Best Batch Size: {best_config[0]}\n')
        f.write(f'Seed: {best_config[1]}\n')
        f.write(f'L1 Strength: {best_config[2]}\n')
        f.write(f'L2 Strength: {best_config[3]}\n')
        f.write(f'Fold: {best_config[4] + 1}\n')
        f.write(f'Validation Accuracy: {best_val_accuracy * 100:.2f}%\n')