from faneModel import SimpleCNN
from faneDataLoader import get_loaders
from faneTrain import train_model
import torch

import sys
import os

# Append the directory path of the previous directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate import report_metrics

data_path = "../../data/fane_experiment_data"

# Load tensors
train_loader, valid_loader, test_loader = get_loaders(data_path)

# Initialize the model
model = SimpleCNN(num_classes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
train_model(model, train_loader, valid_loader, device)

class_labels = ["angry", "confused", "happy", "neutral"]

# Load the model weights
model.load_state_dict(torch.load('model_weights.pth'))

# Report the classification metrics for the training, validation, and testing sets
report_metrics(train_loader, model, device, "Training Set", class_labels)
report_metrics(valid_loader, model, device, "Validation Set", class_labels)
report_metrics(test_loader, model, device, "Testing Set", class_labels)
