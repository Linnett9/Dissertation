import os
from hyperparameters import Hyperparameters
import torch
from torch.utils.data import DataLoader, Dataset
from data_utils import load_test_dataset  # Assume you have a function to load the test dataset
from combine_models import CombinedModel
from sklearn.metrics import classification_report
from test_plots import generate_test_plots  # Assuming this generates the plots
from data_utils import load_test_dataset  # Import the new function

def test_model_on_test_set(test_dir, hp, output_dir, local_model_dir, class_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model and hyperparameters
    checkpoint = torch.load(os.path.join(output_dir, 'final_model.pth'))
    state_dict = checkpoint['model_state_dict']
    saved_hyperparameters = checkpoint['hyperparameters']

    # Recreate the Hyperparameters object with saved values
    hp = Hyperparameters()
    hp.__dict__.update(saved_hyperparameters)

    # Instantiate the model with the correct hyperparameters
    final_model = CombinedModel(hp).to(device)

    # Load the model's state dictionary
    final_model.load_state_dict(state_dict)

    # Load the test dataset
    test_dataset = load_test_dataset(test_dir)  # Directly load the test dataset
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    final_model.eval()
    test_preds = []
    test_targets = []
    test_loss = 0.0
    test_correct = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == targets).sum().item()
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Generate plots for the test set
    generate_test_plots(test_targets, test_preds, class_labels, local_model_dir, test_loss, test_accuracy)
    
    return test_loss, test_accuracy, test_preds, test_targets