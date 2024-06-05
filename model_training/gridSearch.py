
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader

from mainModel import SimpleCNN
from dataLoader import get_loaders
import torch

data_path = "./../data/classes"

# Load tensors
train_loader, valid_loader, test_loader = get_loaders(data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=SimpleCNN(num_classes=4).to(device)




# Wrap the model in Skorch
net = NeuralNetClassifier(
    model,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam(model.parameters()),
    lr=0.001,
    batch_size=64,
    max_epochs=10,
    iterator_train__shuffle=True
)

param_grid = {
    'lr': [0.001, 0.0001],
    'batch_size': [32, 64, 100],
    'optimizer': [optim.Adam],
    'module__num_classes': [4],  # Assuming you are working with a dataset with 10 classes
    'max_epochs': [17, 20]
}


# Convert datasets into a format suitable for scikit-learn
# This function will iterate over the dataset and return the images and labels in numpy arrays
def get_dataset(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, labels = next(iter(loader))
    return data.numpy(), labels.numpy()

X_train, y_train = get_dataset(train_loader.dataset)
X_test, y_test = get_dataset(valid_loader.dataset)


grid_search = GridSearchCV(net, param_grid, refit=True, cv=2, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))