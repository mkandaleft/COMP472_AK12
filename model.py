from matplotlib import pyplot as plt
import sklearn
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 12 * 12, num_classes)
        # self.dropout1 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(32 * 12 * 12, 128) ## Commented out for simplicity of model
        # self.relu3 = nn.ReLU()
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(128, 64)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        # x = self.dropout1(x)
        # x = self.fc1(x)
        # x = self.relu3(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # x = self.relu4(x)
        # x = self.fc3(x)
        return x

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = ImageFolder("./comp472", transform=transform)

# Adjust proportions for training, validation, and testing
train_size = int(0.7 * len(dataset))
valid_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - valid_size

# Split the dataset
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False) # batch_size=64 is also good
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Initialize the CNN model
model = SimpleCNN(num_classes=4).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #lr=0.00019516502701463062 was found optimal

# Training loop
for epoch in range(20): # Could be increased to 50 or more
    # Train the model
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    valid_losses = []
    valid_predictions = []
    valid_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            valid_losses.append(loss.item())
            valid_predictions.extend(predicted.cpu().numpy())
            valid_labels.extend(labels.cpu().numpy())
    
    # Calculate the validation accuracy and F1 score
    valid_accuracy = accuracy_score(valid_labels, valid_predictions)
    valid_f1 = f1_score(valid_labels, valid_predictions, average='weighted')

    print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}, Valid Accuracy = {valid_accuracy:.4f}, Valid F1 Score = {valid_f1:.4f}")

# Test the model and report the classification metrics
def report_metrics(loader, model, device, title, class_labels):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_true.extend(labels.tolist())

            outputs = torch.softmax(model(inputs), dim=1)
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy().tolist())

    class_report = sklearn.metrics.classification_report(y_true, y_pred)
    print(f"Classification report for the {title}:")
    print(class_report)

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix for the {title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Report the classification metrics for the training, validation, and testing sets
class_labels = ["angry", "focused", "happy", "neutral"]

report_metrics(train_loader, model, device, "Training Set", class_labels)
report_metrics(valid_loader, model, device, "Validation Set", class_labels)
report_metrics(test_loader, model, device, "Testing Set", class_labels)

