import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, concatenate, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2, l1_l2
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import keras_tuner as kt

# Custom functions and modules
from plots import plot_learning_curves, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_class_wise_accuracy, plot_train_vs_val_accuracy, plot_train_vs_val_loss, save_summary_and_report, save_best_configuration
from transfer_model import transfer_model_to_gpu, transfer_tuner_results_to_gpu, REMOTE_TUNER_RESULTS_DIR, REMOTE_MODEL_SAVE_DIR

# Enable eager execution mode
tf.config.run_functions_eagerly(True)

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define paths
train_dir='/data/bl70/CNNTesting/NewTrainingImages/NewTrainingImages'
#train_dir = os.path.expanduser('~/Documents/CNNTesting/NewTrainingImages')
output_dir = os.path.expanduser('~/Documents/TestingResInceptionMobile/Data')
local_model_dir = os.path.expanduser('~/Downloads/TestingResInceptionMobileCond/Model')
tuner_results_dir = os.path.expanduser('~/Downloads/TestingResInceptionMobileCond')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(local_model_dir, exist_ok=True)
os.makedirs(tuner_results_dir, exist_ok=True)

# Check if the directories are writable
if not os.access(output_dir, os.W_OK) or not os.access(local_model_dir, os.W_OK) or not os.access(tuner_results_dir, os.W_OK):
    raise Exception("Output or local model directory is not writable")

# Using GroupKFold for cross-validation
kf = GroupKFold(n_splits=5)

def build_combined_model(hp):
    # Set the random seed for reproducibility
    seed = hp.Int('seed', min_value=0, max_value=10000, step=1)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    input_shape = (244, 244, 3)
    dropout_rate = hp.Float('dropout_rate', 0.3, 0.7, step=0.1)
    l2_strength = hp.Float('l2_strength', 1e-5, 1e-2, sampling='log')
    l1_strength = hp.Float('l1_strength', 1e-5, 1e-2, sampling='log')
    learning_rate = hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')
    num_dense_units = hp.Int('num_dense_units', 128, 1024, step=128)
    activation = hp.Choice('activation', ['relu', 'tanh', 'sigmoid'])
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    batch_size = hp.Int('batch_size', 16, 64, step=16)
    kernel_size = hp.Choice('kernel_size', [3, 5, 7])
    filters = hp.Int('filters', 32, 128, step=32)
    classification_threshold = hp.Float('classification_threshold', 0.1, 0.9, step=0.1)
    conditional_weight = hp.Float('conditional_weight', 0.1, 0.9, step=0.1)

    # Input layer
    input_tensor = Input(shape=input_shape)

    # Load ResNet50 as the primary base model (not trainable during fine-tuning)
    resnet_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in resnet_base.layers:
        layer.trainable = False

    # Load MobileNetV2 as a secondary base model (not trainable during fine-tuning)
    mobile_net_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in mobile_net_base.layers:
        layer.trainable = False

    # Load InceptionV3 as a tertiary base model (not trainable during fine-tuning)
    inception_base = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in inception_base.layers:
        layer.trainable = False

    # Add GlobalAveragePooling2D to all models
    resnet_output = GlobalAveragePooling2D()(resnet_base.output)
    mobile_net_output = GlobalAveragePooling2D()(mobile_net_base.output)
    inception_output = GlobalAveragePooling2D()(inception_base.output)

    # Combine outputs of all models
    combined = concatenate([resnet_output, mobile_net_output, inception_output])

    # Encoder part using ResNet blocks on top of combined base models
    x = Dense(filters, activation=activation, kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength))(combined)
    x = Dropout(dropout_rate)(x)
    x = resnet_block(x, filters, l1_strength=l1_strength, l2_strength=l2_strength, activation=activation)
    x = Dropout(dropout_rate)(x)

    # First Classifier
    primary_classifier = Dense(1, activation='sigmoid', name='primary_output')(x)

    # Conditional activation of the second classifier
    def conditional_activation(inputs, threshold, weight):
        primary_output, secondary_input = inputs
        secondary_output = tf.where(primary_output > threshold, secondary_input, primary_output * weight)
        return secondary_output

    # Second Classifier (activated conditionally)
    secondary_classifier = Dense(1, activation='sigmoid', name='secondary_output')(x)
    conditional_output = tf.keras.layers.Lambda(conditional_activation, arguments={'threshold': classification_threshold, 'weight': conditional_weight}, name='conditional_output')([primary_classifier, secondary_classifier])

    model = Model(inputs=input_tensor, outputs=[primary_classifier, conditional_output])

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate)

    model.compile(optimizer=opt, 
                  loss={'primary_output': 'binary_crossentropy', 'conditional_output': 'binary_crossentropy'}, 
                  metrics={'primary_output': 'accuracy', 'conditional_output': 'accuracy'})

    # Store the classification threshold and weight as attributes of the model
    model.classification_threshold = classification_threshold
    model.conditional_weight = conditional_weight

    return model

def resnet_block(input_tensor, filters, kernel_size=3, l1_strength=0.0, l2_strength=0.01, activation='relu'):
    x = Dense(filters, kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dense(filters, kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength))(x)
    x = BatchNormalization()(x)

    input_tensor = Dense(filters, kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength))(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)

    x = Add()([x, input_tensor])
    x = Activation(activation)(x)
    return x

def create_dataset(filenames, labels, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Ensure labels are converted to numpy array before creating dataframe
    labels = np.array(labels)

    df = pd.DataFrame({'filename': filenames, 'class': labels.astype(str)})

    generator = datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='class',
        target_size=(244, 244),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: ((x, y[:, np.newaxis]) for x, y in generator),
        output_signature=(
            tf.TensorSpec(shape=(None, 244, 244, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    )
    return dataset.repeat(), generator

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = build_combined_model(hp)
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Int('batch_size', 16, 64, step=16)
        x_train, y_train = kwargs.pop('x')
        x_val, y_val = kwargs.pop('validation_data')

        # Ensure y_train and y_val are converted to numpy arrays
        y_train = np.array(y_train)
        y_val = np.array(y_val)

        train_dataset, train_generator = create_dataset(x_train, y_train, batch_size)
        val_dataset, val_generator = create_dataset(x_val, y_val, batch_size)
        train_steps = len(x_train) // batch_size
        val_steps = len(x_val) // batch_size

        callbacks = kwargs.pop('callbacks', None)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        history = model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=callbacks or [EarlyStopping(monitor='val_primary_output_accuracy', patience=10, restore_best_weights=True, mode='max'), lr_reducer],
            **kwargs
        )

        return history

class TrialCallback(tf.keras.callbacks.Callback):
    def __init__(self, tuner_dir, model_dir):
        super().__init__()
        self.trial_count = 0
        self.tuner_dir = tuner_dir
        self.model_dir = model_dir

    def on_train_begin(self, logs=None):
        self.trial_count += 1
        print(f"Starting trial {self.trial_count}")

    def on_train_end(self, logs=None):
        print(f"Attempting to transfer tuner results from {self.tuner_dir}")
        best_model_path = os.path.join(self.model_dir, 'best_model.keras')
        if os.path.exists(best_model_path):
            transfer_tuner_results_to_gpu(self.tuner_dir, best_model_path)
        else:
            print(f"Model file not found at {best_model_path}")

def tune_model(train_filenames, train_labels, val_filenames, val_labels):
    trial_callback = TrialCallback(tuner_results_dir, local_model_dir)
    tuner = kt.Hyperband(
        MyHyperModel(),
        objective=kt.Objective('val_primary_output_accuracy', direction='max'),
        max_epochs=20,
        factor=3,
        directory=tuner_results_dir,
        project_name='hyperparameter_tuning'
    )

    tuner.search(
        x=(train_filenames, train_labels),
        validation_data=(val_filenames, val_labels),
        epochs=20,
        callbacks=[EarlyStopping(monitor='val_primary_output_accuracy', patience=10, restore_best_weights=True, mode='max'), trial_callback]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    return model, best_hps

def save_best_results(best_val_accuracy, best_config, output_dir):
    best_config_file = os.path.join(output_dir, 'best_config.txt')
    with open(best_config_file, 'w') as f:
        f.write(f"Best validation accuracy: {best_val_accuracy}\n")
        f.write(f"Best hyperparameters:\n")
        for key, value in best_config.values.items():
            f.write(f"{key}: {value}\n")

# Load the data once to split indices
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

data_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(244, 244),
    batch_size=3,
    class_mode='binary',
    shuffle=False
)

filenames = np.array(data_generator.filepaths)
labels = data_generator.classes
patient_ids = np.array([os.path.basename(fname)[:7] for fname in filenames])

# Hyperparameter tuning
best_val_accuracy = 0
best_config = None
best_model = None

for train_indices, val_indices in kf.split(filenames, labels, groups=patient_ids):
    train_filenames = filenames[train_indices]
    val_filenames = filenames[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    model, best_hps = tune_model(train_filenames, train_labels, val_filenames, val_labels)

    val_dataset, val_generator = create_dataset(val_filenames, val_labels, best_hps['batch_size'])
    val_steps = len(val_indices) // best_hps['batch_size']
    val_loss, val_accuracy = model.evaluate(val_dataset, steps=val_steps)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_config = best_hps
        best_model = model

print(f"Best validation accuracy: {best_val_accuracy}")
print(f"Best hyperparameters: {best_config}")

# Save the best model, configuration, and results
best_model_path = os.path.join(local_model_dir, 'best_model.keras')
best_model.save(best_model_path)
save_best_configuration(best_config, best_val_accuracy, output_dir)
save_best_results(best_val_accuracy, best_config, output_dir)

# Final transfer to ensure everything is copied
transfer_model_to_gpu(best_model_path, os.path.join(REMOTE_MODEL_SAVE_DIR, 'best_model.keras'))
#transfer_tuner_results_to_gpu(tuner_results_dir)