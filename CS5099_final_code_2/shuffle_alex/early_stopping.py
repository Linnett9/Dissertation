
#Desction: Early stopping class for pytorch models
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.delta = delta

    def __call__(self, val_accuracy, model, path):
        score = val_accuracy
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_accuracy, model, path):
        '''Saves model when validation accuracy increases.'''
        if self.verbose:
          print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_accuracy:.6f}).  Saving model ...')
        torch.save({
          'model_state_dict': model.state_dict(),
          'hyperparameters': model.hp.__dict__,  # or hp.__dict__ if hp is passed in
        }, path)
        self.val_acc_max = val_accuracy