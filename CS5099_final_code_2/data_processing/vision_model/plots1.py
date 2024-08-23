import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import numpy as np

# Set a consistent style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 18})

def plot_and_save_metrics(history, train_generator, val_generator, model_name, batch_size, seed, fold, output_dir):
    plot_learning_curves(history, model_name, batch_size, seed, fold, output_dir)
    
    val_generator.reset()
    true_classes = val_generator.classes
    predictions = history.model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    class_labels = list(train_generator.class_indices.keys())
    plot_confusion_matrix(true_classes, predicted_classes, class_labels, model_name, batch_size, seed, fold, output_dir)
    
    if len(class_labels) == 2:
        plot_roc_curve(true_classes, predictions[:, 1], model_name, class_labels, batch_size, seed, fold, output_dir)
        plot_precision_recall_curve(true_classes, predictions[:, 1], model_name, class_labels, batch_size, seed, fold, output_dir)


def plot_learning_curves(history, model_name, batch_size, seed, fold, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='black')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='black', linestyle='--')
    plt.title(f'Learning Curves - Accuracy (Model: {model_name}, Batch Size {batch_size}, Seed {seed}, Fold {fold + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves_accuracy_{model_name}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='black')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='black', linestyle='--')
    plt.title(f'Learning Curves - Loss (Model: {model_name}, Batch Size {batch_size}, Seed {seed}, Fold {fold + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves_loss_{model_name}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.png', dpi=300)
    plt.close()

def plot_confusion_matrix(true_classes, predicted_classes, class_labels, model_name, batch_size, seed, fold, output_dir):
    cm = confusion_matrix(true_classes, predicted_classes)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, 
                cbar_kws={'label': 'Percentage'}, annot_kws={"size": 18})

    plt.title(f'Confusion Matrix (Model: {model_name}, Batch Size {batch_size}, Seed {seed}, Fold {fold + 1})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_{model_name}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.png', dpi=300)
    plt.close()

def plot_roc_curve(true_classes, predictions, model_name, batch_size, seed, fold, output_dir):
    fpr, tpr, _ = roc_curve(true_classes, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (Model: {model_name}, Batch Size {batch_size}, Seed {seed}, Fold {fold + 1})')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve_{model_name}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.png', dpi=300)
    plt.close()

def plot_precision_recall_curve(true_classes, predictions, model_name, batch_size, seed, fold, output_dir):
    precision, recall, _ = precision_recall_curve(true_classes, predictions)
    average_precision = average_precision_score(true_classes, predictions)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='black', lw=2, label=f'Precision-Recall curve (area = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve (Model: {model_name}, Batch Size {batch_size}, Seed {seed}, Fold {fold + 1})')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_curve_{model_name}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.png', dpi=300)
    plt.close()

def plot_class_wise_accuracy(report, class_labels, model_name, batch_size, seed, fold, output_dir):
    class_accuracies = [report[class_name]['precision'] for class_name in class_labels]
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
    plt.ylabel('Accuracy')
    plt.title(f'Class-wise Accuracy (Model: {model_name}, Batch Size {batch_size}, Seed {seed}, Fold {fold + 1})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_wise_accuracy_{model_name}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.png', dpi=300)
    plt.close()

def plot_train_vs_val_accuracy(history, model_name, batch_size, seed, fold, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='black')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='black', linestyle='--')
    plt.title(f'Training vs Validation Accuracy (Model: {model_name}, Batch Size {batch_size}, Seed {seed}, Fold {fold + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_val_accuracy_{model_name}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.png', dpi=300)
    plt.close()

def plot_train_vs_val_loss(history, model_name, batch_size, seed, fold, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='black')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='black', linestyle='--')
    plt.title(f'Training vs Validation Loss (Model: {model_name}, Batch Size {batch_size}, Seed {seed}, Fold {fold + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_val_loss_{model_name}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.png', dpi=300)
    plt.close()

def save_summary_and_report(batch_size, seed, fold, training_time, val_loss, val_accuracy, best_val_accuracy, best_true_classes, predicted_classes, class_labels, vit_model, output_dir):
    # Save the summary
    with open(f'{output_dir}/summary_{vit_model}_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.txt', 'w') as f:
        f.write(f'Model: {vit_model}\n')
        f.write(f'Batch Size: {batch_size}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Fold: {fold + 1}\n')
        f.write(f'Training Time: {training_time:.2f} seconds\n')
        f.write(f'Validation Loss: {val_loss}\n')
        f.write(f'Validation Accuracy: {val_accuracy * 100:.2f}%\n')
        f.write(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%\n')
        f.write('\nClassification Report:\n')
        f.write(classification_report(best_true_classes, predicted_classes, target_names=class_labels, zero_division=0))
        f.write('\nConfiguration:\n')
        f.write(f'Model Layers:\n')
        for layer in vit_model.layers:
            f.write(f'{layer}\n')

def save_best_configuration(best_config, best_val_accuracy, output_dir):
    # Save the best configuration
    with open(f'{output_dir}/best_configuration.txt', 'w') as f:
        f.write(f'Best Model: {best_config[0]}\n')
        f.write(f'Batch Size: {best_config[1]}\n')
        f.write(f'Seed: {best_config[2]}\n')
        f.write(f'Fold: {best_config[3] + 1}\n')
        f.write(f'Validation Accuracy: {best_val_accuracy * 100:.2f}%\n')