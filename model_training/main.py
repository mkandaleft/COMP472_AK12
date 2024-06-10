from mainModel import SimpleCNN
# from oldModel import SimpleCNN # Use this line if you want to use the old model
from dataLoader import get_loaders
from train import train_model
from evaluate import report_metrics
import torch

data_path = "./data/classes"

# Load tensors
train_loader, valid_loader, test_loader = get_loaders(data_path)

# Initialize the model
model = SimpleCNN(num_classes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
train_model(model, train_loader, valid_loader, device)

class_labels = ["angry", "focused", "happy", "neutral"]

# Load the model weights
model.load_state_dict(torch.load('model_weights.pth'))

# Report the classification metrics for the training, validation, and testing sets
report_metrics(train_loader, model, device, "Training Set", class_labels)
report_metrics(valid_loader, model, device, "Validation Set", class_labels)
report_metrics(test_loader, model, device, "Testing Set", class_labels)
