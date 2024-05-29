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

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Initialize the CNN model
model = SimpleCNN(num_classes=4).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"Epoch {epoch+1}: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

model.eval()

# Test and report on the training data.
y_train_true = []
y_train_pred = []

for data in train_loader:
  train_inputs, train_labels = data
  train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
  y_train_true.extend(train_labels.tolist())

  output = torch.softmax(model(train_inputs), dim=1)
  y_train_pred.extend(output.argmax(dim=1).cpu().numpy().tolist())

class_report_train = sklearn.metrics.classification_report(y_train_true, y_train_pred)
print("Classification report for the training set:")
print(class_report_train)

# Display the confusion matrix
class_labels = ["angry", "focused", "happy", "neutral"]
cm_train = sklearn.metrics.confusion_matrix(y_train_true, y_train_pred)

sns.heatmap(cm_train, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix for the Training Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# Test and report on the testing data.
y_test_true = []
y_test_pred = []

for data in test_loader:
  test_inputs, test_labels = data
  test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
  y_test_true.extend(test_labels.tolist())

  output = torch.softmax(model(test_inputs), dim=1)
  y_test_pred.extend(output.argmax(dim=1).cpu().numpy().tolist())

class_report_test = sklearn.metrics.classification_report(y_test_true, y_test_pred)
print("\nClassification report for the testing set:")
print(class_report_test)

# Display the confusion matrix
cm_test = sklearn.metrics.confusion_matrix(y_test_true, y_test_pred)

sns.heatmap(cm_test, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix for the Testing Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()