#Description: This file contains the hyperparameters class that is used to define the hyperparameters for the model. 
#The hyperparameters are defined using the Optuna library. The hyperparameters are used to define the model architecture and the training parameters.
import optuna

class Hyperparameters:
    def __init__(self, trial=None):
        if trial:
            self.seed = trial.suggest_int('seed', 0, 100000)
            self.dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.5)
            self.l2_strength = trial.suggest_float('l2_strength', 1e-6, 1e-2, log=True)
            self.l1_strength = trial.suggest_float('l1_strength', 1e-6, 1e-2, log=True)
            
            # Setting the batch size and scaling the learning rate
            self.batch_size = trial.suggest_int('batch_size', 128, 512)
            self.learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-4, log=True)
            
            self.num_dense_units = trial.suggest_int('num_dense_units', 128, 1024)
            self.activation = trial.suggest_categorical('activation', ['relu'])
            self.optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
            self.filters = trial.suggest_int('filters', 32, 128)
            self.classification_threshold = trial.suggest_float('classification_threshold', 0.1, 0.9)
            self.cond_activation_threshold = trial.suggest_float('cond_activation_threshold', 0.6, 0.8)
            self.cond_activation_weight = trial.suggest_float('cond_activation_weight', 0.5, 0.9)
        else:
            self.seed = 42
            self.dropout_rate = 0.5
            self.l2_strength = 1e-4
            self.l1_strength = 1e-4
            self.learning_rate = 1e-3
            self.num_dense_units = 512
            self.activation = 'relu'
            self.optimizer = 'adam'
            self.batch_size = 32
            self.filters = 64
            self.classification_threshold = 0.5
            self.cond_activation_threshold = 0.7
            self.cond_activation_weight = 0.5