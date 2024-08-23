# plot_call.py
import numpy as np
from sklearn.metrics import classification_report
from plots import (plot_computation_time, plot_learning_curves, plot_confusion_matrix, plot_roc_curve, 
                   plot_precision_recall_curve, plot_class_wise_accuracy, save_summary_and_report)

def generate_plots(history, val_targets, val_preds, class_labels, hp, local_model_dir, epoch_times, val_loss, val_accuracy, final_model):
    # Generate and save various plots and reports
    plot_computation_time(epoch_times, local_model_dir)
    plot_learning_curves(history, 'Final Model', hp.batch_size, hp.seed, local_model_dir)
    plot_confusion_matrix(val_targets, np.argmax(val_preds, axis=1), class_labels, 'Final Model', hp.batch_size, hp.seed, local_model_dir)
    plot_roc_curve(val_targets, np.array(val_preds)[:, 1], class_labels, 'Final Model', hp.batch_size, hp.seed, local_model_dir)
    plot_precision_recall_curve(val_targets, np.array(val_preds)[:, 1], class_labels, 'Final Model', hp.batch_size, hp.seed, local_model_dir)
    
    report = classification_report(val_targets, np.argmax(val_preds, axis=1), target_names=class_labels, output_dict=True, zero_division=0)
    plot_class_wise_accuracy(report, class_labels, 'Final Model', hp.batch_size, hp.seed, local_model_dir)
    
    save_summary_and_report('Final Model', hp.batch_size, hp.seed, val_loss, val_accuracy, val_accuracy, 
                            val_targets, np.argmax(val_preds, axis=1), 
                            class_labels, final_model, local_model_dir)
    
    print(f"Final Model Test Accuracy: {val_accuracy * 100:.2f}%")
