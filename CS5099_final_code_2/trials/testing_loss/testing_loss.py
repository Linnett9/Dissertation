import os
import time
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import math
from plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_wise_accuracy,
    plot_train_vs_val_accuracy,
    plot_train_vs_val_loss,
    save_summary_and_report,
    save_best_configuration,
    plot_learning_curves
)

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define paths
train_dir='/data/bl70/CNNTesting/NewTrainingImages/NewTrainingImages'
output_dir = os.path.expanduser('~/Documents/NewLossTestingDupeBenign/CNNData')
local_model_dir = os.path.expanduser('~/Downloads')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check if the output directory is writable
if os.access(output_dir, os.W_OK):
    print(f"Output directory {output_dir} is writable")
else:
    print(f"Output directory {output_dir} is not writable")

# Create the local model directory if it doesn't exist
os.makedirs(local_model_dir, exist_ok=True)

# Check if the local model directory is writable
if os.access(local_model_dir, os.W_OK):
    print(f"Local model directory {local_model_dir} is writable")
else:
    print(f"Local model directory {local_model_dir} is not writable")

# Different loss functions to test
loss_functions = [
 'binary_crossentropy',
    'kld',
    'poisson',
    'categorical_crossentropy',
    'sparse_categorical_crossentropy',
]

# Learning rates to test
learning_rates = [0.001]

# Batch sizes to test
batch_sizes = [16, 32, 64, 128, 256]


seeds = [0,2000, 4000, 6000, 8000, 10000]

best_val_accuracy = 0
best_config = None
best_config_history = None

# Using GroupKFold for cross-validation
kf = GroupKFold(n_splits=5)

def save_fold_patient_ids(train_indices, val_indices, fold, output_dir, patient_ids):
    # Get the patient IDs for training and validation
    train_patient_ids = patient_ids[train_indices]
    val_patient_ids = patient_ids[val_indices]

    # Create a DataFrame for training and validation IDs
    df_train = pd.DataFrame(train_patient_ids, columns=['PatientID'])
    df_val = pd.DataFrame(val_patient_ids, columns=['PatientID'])

    # Save to CSV files
    train_csv_path = os.path.join(output_dir, f'fold_{fold+1}_train_patient_ids.csv')
    val_csv_path = os.path.join(output_dir, f'fold_{fold+1}_val_patient_ids.csv')

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)

    print(f"Saved training patient IDs to {train_csv_path}")
    print(f"Saved validation patient IDs to {val_csv_path}")

def compile_and_train_model(loss_function, seed, fold, learning_rate, batch_size, train_indices, val_indices, data_generator):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Determine class_mode based on loss function
    if loss_function == 'sparse_categorical_crossentropy' or loss_function == 'binary_crossentropy':
        class_mode = 'binary' if loss_function == 'binary_crossentropy' else 'sparse'
        output_signature = (
            tf.TensorSpec(shape=(None, 244, 244, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    else:
        class_mode = 'categorical'
        output_signature = (
            tf.TensorSpec(shape=(None, 244, 244, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(data_generator.class_indices)), dtype=tf.float32)
        )
    
    # Data generators
    datagen = ImageDataGenerator()

    # Get the filenames and labels
    filenames = np.array(data_generator.filepaths)
    labels = data_generator.classes

    train_filenames = filenames[train_indices]
    val_filenames = filenames[val_indices]
    train_labels = labels[train_indices].astype(str)  # Convert labels to string
    val_labels = labels[val_indices].astype(str)  # Convert labels to string

    def create_dataset(filenames, labels, batch_size, class_mode, output_signature):
        df = pd.DataFrame({'filename': filenames, 'class': labels})
        generator = datagen.flow_from_dataframe(
            df,
            x_col='filename',
            y_col='class',
            target_size=(244, 244),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=output_signature
        )
        return dataset.repeat(), generator

    train_dataset, train_generator = create_dataset(train_filenames, train_labels, batch_size, class_mode, output_signature)
    val_dataset, val_generator = create_dataset(val_filenames, val_labels, batch_size, class_mode, output_signature)

    # Calculate steps per epoch dynamically
    train_steps = len(train_indices) // batch_size
    val_steps = len(val_indices) // batch_size

    print(f"Steps per epoch for training: {train_steps}")
    print(f"Steps per epoch for validation: {val_steps}")

    # Check if there is enough data for the specified number of epochs
    if train_steps * 15 > len(train_indices):
        print(f"Not enough training data for seed {seed}, fold {fold + 1}. Skipping this configuration.")
        return None

    if val_steps * 15 > len(val_indices):
        print(f"Not enough validation data for seed {seed}, fold {fold + 1}. Skipping this configuration.")
        return None

    # Building the CNN model
    model = Sequential([
        Input(shape=(244, 244, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1 if class_mode == 'binary' else len(train_generator.class_indices), activation='sigmoid' if class_mode == 'binary' else 'softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])

    # Early stopping to avoid overfitting
    checkpoint_path = os.path.join(local_model_dir, f'best_model_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}_batch_{batch_size}.keras')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max')

    # Train the model
    start_time = time.time()
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        epochs=20,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=[model_checkpoint, early_stopping]
    )
    training_time = time.time() - start_time

    # Load the best model weights
    model.load_weights(checkpoint_path)

    # Evaluate the model on the validation data
    val_loss, val_accuracy = model.evaluate(val_dataset, steps=val_steps)
    print(f'Validation Accuracy with loss function {loss_function}, Seed {seed}, Fold {fold + 1}, Learning Rate {learning_rate}, Batch Size {batch_size}: {val_accuracy * 100:.2f}%')

    # Predictions and classification report
    val_steps_per_epoch = math.ceil(len(val_indices) / batch_size)
    predictions = model.predict(val_dataset, steps=val_steps_per_epoch)
    if class_mode == 'categorical':
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predicted_classes = np.round(predictions).astype(int).flatten()
    true_classes = val_labels.astype(int)
    class_labels = list(val_generator.class_indices.keys())

    # Ensure consistency in the number of samples
    min_length = min(len(true_classes), len(predicted_classes))
    true_classes = true_classes[:min_length]
    predicted_classes = predicted_classes[:min_length]

    report = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0, output_dict=True)
    print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))

    return history, model, val_dataset, training_time, val_loss, val_accuracy, true_classes, predicted_classes, class_labels, report, predictions

# Iterate over loss functions, learning rates, batch sizes, and seeds to train models
for loss_function in loss_functions:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for seed in seeds:
                tf.random.set_seed(seed)
                np.random.seed(seed)
                print(f"\nTesting with loss function: {loss_function}, learning rate: {learning_rate}, batch size: {batch_size} and seed: {seed}\n")

                # Data generators
                datagen = ImageDataGenerator()

                # Load the data once to split indices
                data_generator = datagen.flow_from_directory(
                    train_dir,
                    target_size=(244, 244),
                    batch_size=1,  # Batch size as 1 to get all images
                    class_mode='binary' if loss_function == 'binary_crossentropy' else 'categorical',
                    shuffle=False  # No shuffling to keep the order
                )

                # Get the filenames and labels
                filenames = np.array(data_generator.filepaths)
                labels = data_generator.classes

                # Convert labels to strings for compatibility with the generator
                labels = labels.astype(str)

                # Extract patient IDs from filenames (first 7 characters of the filename)
                patient_ids = np.array([os.path.basename(fname)[:7] for fname in filenames])

                best_config_val_accuracy = 0  # Track best validation accuracy for current config
                best_history = None
                best_true_classes = None
                best_predicted_classes = None
                best_predictions = None
                best_report = None
                class_labels = list(data_generator.class_indices.keys())

                for fold, (train_indices, val_indices) in enumerate(kf.split(filenames, labels, groups=patient_ids)):
                    print(f"\nFold {fold + 1}\n")

                    # Save patient IDs for the current fold
                    save_fold_patient_ids(train_indices, val_indices, fold, output_dir, patient_ids)

                    train_filenames = filenames[train_indices]
                    val_filenames = filenames[val_indices]
                    train_labels = labels[train_indices]  # Labels are already strings
                    val_labels = labels[val_indices]  # Labels are already strings

                    result = compile_and_train_model(loss_function, seed, fold, learning_rate, batch_size, train_indices, val_indices, data_generator)
                    
                    if result is None:
                        continue

                    history, model, val_dataset, training_time, val_loss, val_accuracy, true_classes, predicted_classes, class_labels, report, predictions = result

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_config = (loss_function, seed, fold, learning_rate, batch_size)
                        print(f'New best configuration found: {best_config} with accuracy {best_val_accuracy * 100:.2f}%')

                    # Check if this is the best model for the current configuration
                    if val_accuracy > best_config_val_accuracy:
                        best_config_val_accuracy = val_accuracy
                        best_history = history
                        best_true_classes = true_classes
                        val_steps_per_epoch = math.ceil(len(val_indices) / batch_size)
                        best_predictions = predictions
                        best_predicted_classes = predicted_classes
                        best_report = report
                        print(classification_report(best_true_classes, best_predicted_classes, target_names=class_labels))

                        # Save the best model locally
                        local_model_path = os.path.join(local_model_dir, f'model_loss_{loss_function}_seed_{seed}_fold_{fold + 1}_lr_{learning_rate}_batch_{batch_size}.keras')
                        model.save(local_model_path)

                # Plot the learning curves for the best model configuration
                if best_history:
                    print(f"Plotting learning curves for loss function {loss_function}, seed {seed}, fold {fold}, learning rate {learning_rate}, batch size {batch_size}")
                    plot_learning_curves(best_history, loss_function, seed, fold, learning_rate, output_dir)

                # Call plotting functions for the best model only after all folds are complete
                if best_history and best_report:
                    print(f"Plotting confusion matrix for loss function {loss_function}, seed {seed}, fold {fold}, learning rate {learning_rate}, batch size {batch_size}")
                    plot_confusion_matrix(best_true_classes, best_predicted_classes, loss_function, seed, fold, learning_rate, output_dir)
                    print(f"Plotting ROC curve for loss function {loss_function}, seed {seed}, fold {fold}, learning rate {learning_rate}, batch size {batch_size}")
                    plot_roc_curve(best_true_classes, best_predictions, loss_function, seed, fold, learning_rate, output_dir)
                    print(f"Plotting precision-recall curve for loss function {loss_function}, seed {seed}, fold {fold}, learning rate {learning_rate}, batch size {batch_size}")
                    plot_precision_recall_curve(best_true_classes, best_predictions, loss_function, seed, fold, learning_rate, output_dir)
                    print(f"Plotting class-wise accuracy for loss function {loss_function}, seed {seed}, fold {fold}, learning rate {learning_rate}, batch size {batch_size}")
                    plot_class_wise_accuracy(best_report, loss_function, seed, fold, learning_rate, output_dir)
                    print(f"Plotting training vs validation accuracy for loss function {loss_function}, seed {seed}, fold {fold}, learning rate {learning_rate}, batch size {batch_size}")
                    plot_train_vs_val_accuracy(best_history, loss_function, seed, fold, learning_rate, output_dir)
                    print(f"Plotting training vs validation loss for loss function {loss_function}, seed {seed}, fold {fold}, learning rate {learning_rate}, batch size {batch_size}")
                    plot_train_vs_val_loss(best_history, loss_function, seed, fold, learning_rate, output_dir)

                # Save summary and report for the best configuration
                if best_history and best_report:
                    save_summary_and_report(
                        loss_function, seed, fold, learning_rate, training_time, val_loss, val_accuracy, best_val_accuracy,
                        best_true_classes, best_predicted_classes, best_report, model, output_dir
                    )

        # Save the best configuration
        save_best_configuration(best_config, best_val_accuracy, output_dir)
        print(f"Best configuration for {loss_function} with learning rate {learning_rate} saved: {best_config} with accuracy {best_val_accuracy * 100:.2f}%")