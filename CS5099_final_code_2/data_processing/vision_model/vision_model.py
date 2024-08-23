import os
import time
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import math
from vit_keras import vit

from transfer_model1 import transfer_model_to_gpu, REMOTE_MODEL_SAVE_DIR
from plots1 import (
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
output_dir = os.path.expanduser('~/Documents/CNNTestingVision/CNNData')
local_model_dir = os.path.expanduser('~/Downloads/Vision/Data')

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(local_model_dir, exist_ok=True)

# Check directory writability
for dir_path, dir_name in [(output_dir, "Output"), (local_model_dir, "Local model")]:
    if os.access(dir_path, os.W_OK):
        print(f"{dir_name} directory {dir_path} is writable")
    else:
        print(f"{dir_name} directory {dir_path} is not writable")

# Batch sizes and seeds
batch_sizes = [16]
seeds = [2000, 4000, 6000, 8000, 10000]

# ViT models to test
vit_models = [
    {"name": "vit_b16", "model_func": vit.vit_b16, "image_size": 224},
    {"name": "vit_b32", "model_func": vit.vit_b32, "image_size": 224},
    {"name": "vit_l16", "model_func": vit.vit_l16, "image_size": 224},
    {"name": "vit_l32", "model_func": vit.vit_l32, "image_size": 224}
]

best_val_accuracy = 0
best_config = None
best_config_history = None

kf = GroupKFold(n_splits=5)

def create_dataset(filenames, labels, batch_size, image_size):
    df = pd.DataFrame({'filename': filenames, 'class': labels})
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='class',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',  # Set to 'binary' for binary classification
        shuffle=True
    )
    
    # Define a generator function
    def generator_func():
        while True:
            x, y = generator.next()
            yield x, y.reshape(-1, 1)  # Reshape y to (batch_size, 1)
    
    dataset = tf.data.Dataset.from_generator(
        generator_func,
        output_signature=(
            tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # Output shape for binary classification
        )
    )
    return dataset.repeat(), generator


history_dict = {}

# Function to verify that both classes are present
def verify_classes(directory):
    classes = os.listdir(directory)
    if len(classes) != 2:
        raise ValueError(f"Expected 2 classes but found {len(classes)} classes in the directory: {directory}")
    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir) or len(os.listdir(class_dir)) == 0:
            raise ValueError(f"Class directory {class_dir} is empty or not a directory.")
    return classes

# Verify the classes
class_labels = verify_classes(train_dir)

for model_info in vit_models:
    for batch_size in batch_sizes:
        for seed in seeds:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            print(f"\nTesting with batch size: {batch_size} and Seed: {seed}\n")

            datagen = ImageDataGenerator()
            data_generator = datagen.flow_from_directory(
                train_dir,
                target_size=(model_info['image_size'], model_info['image_size']),
                batch_size=1,
                class_mode='binary',  # Set to 'binary' for binary classification
                shuffle=False
            )

            filenames = np.array(data_generator.filepaths)
            labels = data_generator.labels.astype(str)
            patient_ids = np.array([os.path.basename(fname)[:7] for fname in filenames])

            best_config_val_accuracy = 0
            best_history = None
            best_true_classes = None
            best_predicted_classes = None
            best_predictions = None
            best_report = None
            old_remote_model_path = None

            for fold, (train_indices, val_indices) in enumerate(kf.split(filenames, labels, groups=patient_ids)):
                print(f"\nFold {fold + 1}\n")

                train_filenames = filenames[train_indices]
                val_filenames = filenames[val_indices]
                train_labels = labels[train_indices]
                val_labels = labels[val_indices]

                train_dataset, train_generator = create_dataset(train_filenames, train_labels, batch_size, model_info['image_size'])
                val_dataset, val_generator = create_dataset(val_filenames, val_labels, batch_size, model_info['image_size'])

                train_steps = len(train_indices) // batch_size
                val_steps = len(val_indices) // batch_size

                print(f"Steps per epoch for training: {train_steps}")
                print(f"Steps per epoch for validation: {val_steps}")

                vit_model = model_info['model_func'](
                    image_size=model_info['image_size'],
                    activation='sigmoid',  # Use 'sigmoid' activation for binary classification
                    pretrained=True,
                    include_top=True,
                    pretrained_top=False,
                    classes=1  # Single output for binary classification
                )

                vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                                  loss='binary_crossentropy', metrics=['accuracy'])

                early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max')
                checkpoint_path = os.path.join(local_model_dir, 'best_model.keras')
                model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')

                start_time = time.time()
                history = vit_model.fit(
                    train_dataset,
                    steps_per_epoch=train_steps,
                    epochs=50,
                    validation_data=val_dataset,
                    validation_steps=val_steps,
                    callbacks=[model_checkpoint, early_stopping]
                )
                training_time = time.time() - start_time

                history_dict[(model_info['name'], batch_size, seed, fold)] = history.history

                vit_model.load_weights(checkpoint_path)

                val_loss, val_accuracy = vit_model.evaluate(val_dataset, steps=val_steps)
                print(f'Validation Accuracy with batch size {batch_size}, Seed {seed}, Fold {fold + 1}: {val_accuracy * 100:.2f}%')

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_config = (model_info['name'], batch_size, seed, fold)
                    best_config_history = history
                    print(f'New best configuration found: {best_config} with accuracy {best_val_accuracy * 100:.2f}%')

                if val_accuracy > best_config_val_accuracy:
                    best_config_val_accuracy = val_accuracy
                    best_history = history
                    best_true_classes = val_labels.astype(int)
                    val_steps_per_epoch = math.ceil(len(val_indices) / batch_size)
                    best_predictions = vit_model.predict(val_dataset, steps=val_steps_per_epoch)
                    best_predicted_classes = np.round(best_predictions).astype(int)  # Use rounding for binary classification
                    best_report = classification_report(best_true_classes, best_predicted_classes, target_names=class_labels, output_dict=True)
                    print(classification_report(best_true_classes, best_predicted_classes, target_names=class_labels))

                    local_model_path = os.path.join(local_model_dir, f'model_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.keras')
                    vit_model.save(local_model_path)
                    remote_model_path = f'{REMOTE_MODEL_SAVE_DIR}/model_batch_size_{batch_size}_seed_{seed}_fold_{fold + 1}.keras'
                    transfer_model_to_gpu(local_model_path, remote_model_path, old_remote_model_path)
                    old_remote_model_path = remote_model_path

            if best_history:
                plot_learning_curves(best_history, model_info['name'], batch_size, seed, fold, output_dir)

            if best_history and best_report:
                plot_confusion_matrix(best_true_classes, best_predicted_classes, class_labels, model_info['name'], batch_size, seed, fold, output_dir)
                plot_roc_curve(best_true_classes, best_predictions, model_info['name'], batch_size, seed, fold, output_dir)
                plot_precision_recall_curve(best_true_classes, best_predictions, model_info['name'], batch_size, seed, fold, output_dir)
                plot_class_wise_accuracy(best_report, class_labels, model_info['name'], batch_size, seed, fold, output_dir)
                plot_train_vs_val_accuracy(best_history, model_info['name'], batch_size, seed, fold, output_dir)
                plot_train_vs_val_loss(best_history, model_info['name'], batch_size, seed, fold, output_dir)

            if best_history and best_report:
                save_summary_and_report(
                    batch_size, seed, fold, training_time, val_loss, val_accuracy, best_val_accuracy,
                    best_true_classes, best_predicted_classes, class_labels, vit_model, output_dir
                )

# Save the best configuration
save_best_configuration(best_config, best_val_accuracy, output_dir)
print(f"Best configuration saved: {best_config} with accuracy {best_val_accuracy * 100:.2f}%")