import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import os
import numpy as np

sns.set(style='whitegrid')
plt.rcParams.update({'font.size': 18})

# Plot computation time
def plot_computation_time(epoch_times, fold_output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_times) + 1), epoch_times)
    plt.title('Computation Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.savefig(os.path.join(fold_output_dir, 'computation_time.png'))
    plt.close()

# Corrected plot_learning_curves function
def plot_learning_curves(history, variation, batch_size, seed, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Access dictionary keys directly
    plt.plot(history['accuracy'], label='Train Accuracy', color='black')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='black', linestyle='--')
    
    plt.title(f'Learning Curves ({variation}, Batch Size {batch_size}, Seed {seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{output_dir}/learning_curves_{variation}_batch_size_{batch_size}_seed_{seed}.png')
    plt.close()

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', color='black')
    plt.plot(history['val_loss'], label='Validation Loss', color='black', linestyle='--')
    
    plt.title(f'Loss Curves ({variation}, Batch Size {batch_size}, Seed {seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_curves_{variation}_batch_size_{batch_size}_seed_{seed}.png')
    plt.close()

def plot_confusion_matrix(true_classes, predicted_classes, class_labels, variation, batch_size, seed, output_dir):
    cm = confusion_matrix(true_classes, predicted_classes)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100 

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, 
                cbar_kws={'label': 'Percentage'}, annot_kws={"size": 18}) 
    plt.title(f'Confusion Matrix ({variation}, Batch Size {batch_size}, Seed {seed})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{output_dir}/confusion_matrix_{variation}_batch_size_{batch_size}_seed_{seed}.png')
    plt.close()

def plot_roc_curve(true_classes, predictions, class_labels, variation, batch_size, seed, output_dir):
    if len(class_labels) == 2:  # Binary classification case
        fpr, tpr, _ = roc_curve(true_classes, predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic ({variation}, Batch Size {batch_size}, Seed {seed})')
        plt.legend(loc='lower right')
        plt.savefig(f'{output_dir}/roc_curve_{variation}_batch_size_{batch_size}_seed_{seed}.png')
        plt.close()

def plot_precision_recall_curve(true_classes, predictions, class_labels, variation, batch_size, seed, output_dir):
    if len(class_labels) == 2:  # Binary classification case
        precision, recall, _ = precision_recall_curve(true_classes, predictions)
        average_precision = average_precision_score(true_classes, predictions)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='black', lw=2, label=f'Precision-Recall curve (area = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0]) 
        plt.title(f'Precision-Recall Curve ({variation}, Batch Size {batch_size}, Seed {seed})')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/precision_recall_curve_{variation}_batch_size_{batch_size}_seed_{seed}.png')
        plt.close()

def plot_class_wise_accuracy(report, class_labels, variation, batch_size, seed, output_dir):
    class_accuracies = [report[class_name]['precision'] for class_name in class_labels]
    plt.figure(figsize=(10, 6))
    
    # Define colors and hatches
    colors = ['black', 'white']
    hatches = ['', '//']
    
    # Plot bars with custom colors and hatches
    bars = plt.bar(class_labels, class_accuracies, color=colors, edgecolor='black')
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # Set y-axis to always go up to 100%
    plt.ylim(0, 1.0)  # 1.0 corresponds to 100%

    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title(f'Class-wise Accuracy ({variation}, Batch Size {batch_size}, Seed {seed})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_wise_accuracy_{variation}_batch_size_{batch_size}_seed_{seed}.png')
    plt.close()

def plot_train_vs_val_accuracy(history, variation, batch_size, seed, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Train Accuracy', color='black')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='black', linestyle='--')
    plt.title(f'Training vs Validation Accuracy ({variation}, Batch Size {batch_size}, Seed {seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_val_accuracy_{variation}_batch_size_{batch_size}_seed_{seed}.png')
    plt.close()

def plot_train_vs_val_loss(history, variation, batch_size, seed, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', color='black')
    plt.plot(history['val_loss'], label='Validation Loss', color='black', linestyle='--')
    plt.title(f'Training vs Validation Loss  ({variation}, Batch Size {batch_size}, Seed {seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_val_loss_{variation}_batch_size_{batch_size}_seed_{seed}.png')
    plt.close()

def save_best_configuration(best_config, best_val_accuracy, output_dir):
    # Save the best configuration
    with open(f'{output_dir}/best_configuration.txt', 'w') as f:
        f.write(f'Best Variation: {best_config[0]}\n')
        f.write(f'Best Batch Size: {best_config[1]}\n')
        f.write(f'Seed: {best_config[2]}\n')
        f.write(f'Validation Accuracy: {best_val_accuracy * 100:.2f}%\n')

def save_summary_and_report(variation, batch_size, seed, val_loss, val_accuracy, best_val_accuracy, true_classes, predicted_classes, class_labels, model, output_dir):
    # Save the summary
    with open(f'{output_dir}/summary_{variation}_batch_size_{batch_size}_seed_{seed}.txt', 'w') as f:
        f.write(f'Variation: {variation}\n')
        f.write(f'Batch Size: {batch_size}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Validation Loss: {val_loss}\n')
        f.write(f'Validation Accuracy: {val_accuracy * 100:.2f}%\n')
        f.write(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%\n')
        f.write('\nClassification Report:\n')
        f.write(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))
        f.write('\nConfiguration:\n')
  
