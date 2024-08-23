import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report, average_precision_score
import os
import numpy as np

sns.set(style='whitegrid')
plt.rcParams.update({'font.size': 18})

def plot_test_confusion_matrix(true_classes, predicted_classes, class_labels, output_dir):
    cm = confusion_matrix(true_classes, predicted_classes)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100 

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, 
                cbar_kws={'label': 'Percentage'}, annot_kws={"size": 18}) 
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_test.png'))
    plt.close()

def plot_test_roc_curve(true_classes, predictions, class_labels, output_dir):
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
        plt.title('Receiver Operating Characteristic (Test Set)')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'roc_curve_test.png'))
        plt.close()

def plot_test_precision_recall_curve(true_classes, predictions, class_labels, output_dir):
    if len(class_labels) == 2:  # Binary classification case
        precision, recall, _ = precision_recall_curve(true_classes, predictions)
        average_precision = average_precision_score(true_classes, predictions)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='black', lw=2, label=f'Precision-Recall curve (area = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0]) 
        plt.title('Precision-Recall Curve (Test Set)')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve_test.png'))
        plt.close()

def plot_test_class_wise_accuracy(report, class_labels, output_dir):
    class_accuracies = [report[class_name]['precision'] for class_name in class_labels]
    plt.figure(figsize=(10, 6))
    
    colors = ['black', 'white']
    hatches = ['', '//']
    
    bars = plt.bar(class_labels, class_accuracies, color=colors, edgecolor='black')
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    plt.ylim(0, 1.0)  # 1.0 corresponds to 100%

    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy (Test Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_wise_accuracy_test.png'))
    plt.close()

def save_test_summary_and_report(test_loss, test_accuracy, true_classes, predicted_classes, class_labels, output_dir):
    # Save the summary
    with open(os.path.join(output_dir, 'test_summary.txt'), 'w') as f:
        f.write(f'Test Loss: {test_loss}\n')
        f.write(f'Test Accuracy: {test_accuracy * 100:.2f}%\n')
        f.write('\nClassification Report:\n')
        f.write(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))

        
def generate_test_plots(true_classes, predicted_classes, class_labels, output_dir, test_loss=None, test_accuracy=None):
    """
    Generates and saves various plots specifically for the test set.
    """
    # Convert predicted classes from probabilities or logits to labels
    predicted_classes = np.array(predicted_classes)  # Ensure it's a NumPy array

    if len(predicted_classes.shape) > 1:
        predicted_classes_labels = np.argmax(predicted_classes, axis=1)
    else:
        # For binary classification or single output predictions
        predicted_classes_labels = (predicted_classes > 0.5).astype(int)  # Assuming binary with sigmoid output
    
    if test_loss is not None and test_accuracy is not None:
        save_test_summary_and_report(test_loss, test_accuracy, true_classes, predicted_classes_labels, class_labels, output_dir)

    if true_classes is not None and predicted_classes is not None:
        report = classification_report(true_classes, predicted_classes_labels, target_names=class_labels, output_dict=True)
        plot_test_confusion_matrix(true_classes, predicted_classes_labels, class_labels, output_dir)
        plot_test_class_wise_accuracy(report, class_labels, output_dir)

        if len(class_labels) == 2:  # Binary classification case
            # Ensure correct handling of predictions for binary classification
            if len(predicted_classes.shape) == 1:
                plot_test_roc_curve(true_classes, predicted_classes, class_labels, output_dir)
                plot_test_precision_recall_curve(true_classes, predicted_classes, class_labels, output_dir)
            else:
                plot_test_roc_curve(true_classes, predicted_classes[:, 1], class_labels, output_dir)
                plot_test_precision_recall_curve(true_classes, predicted_classes[:, 1], class_labels, output_dir)
