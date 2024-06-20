# from variant3_model_7by7 import SimpleCNN
from variant3_model import SimpleCNN
from dataLoader import get_loaders
from train import train_model
from evaluate import report_metrics
import torch


data_path = "../data/classes"

# Load tensors
train_loader, valid_loader, test_loader = get_loaders(data_path)

# Initialize the model
model = SimpleCNN(num_classes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
# Commented out to skip training
train_model(model, train_loader, valid_loader, device)

class_labels = ["angry", "happy", "neutral", "focused"]

# Load the model weights
model.load_state_dict(torch.load('weights_variant3.pth'))

# Report the classification metrics for the training, validation, and testing sets
report_metrics(train_loader, model, device, "Training Set", class_labels)
report_metrics(valid_loader, model, device, "Validation Set", class_labels)
report_metrics(test_loader, model, device, "Testing Set", class_labels)
