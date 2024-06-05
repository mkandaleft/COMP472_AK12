
import torch.nn as nn

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