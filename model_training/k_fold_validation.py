import torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataLoader import get_loaders
from mainModel import SimpleCNN
from train import train_model
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

def train_and_validate_model(train_idx, dataset, device):
    # Split train_idx into train and validation
    train_indices, val_indices = train_test_split(train_idx, test_size=0.15, random_state=10)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=64, shuffle=False)

    # Initialize and train the model
    model = SimpleCNN(num_classes=4).to(device)
    train_model(model, train_loader, val_loader, device)
    
    return model, train_loader, val_loader

def compute_metrics(loader, model, device):
    model.eval()
    y_true = []
    y_pred = []
    # Compute predictions and store true and predicted labels
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.softmax(model(inputs), dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate and return metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

def k_fold_cross_validation(data_path, k=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolder(data_path, transform=transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4926176471], std=[0.2526960784])
    ]))
    
    kf = KFold(n_splits=k, shuffle=True, random_state=10)
    
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    # Print the header of the table
    print(f"{'Fold':<5} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{k}')
        
        model, train_loader, val_loader = train_and_validate_model(train_idx, dataset, device)
        
        # Evaluate the model on the validation set
        accuracy, precision, recall, f1 = compute_metrics(val_loader, model, device)
        
        # Store the results for each fold
        fold_results['accuracy'].append(accuracy)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1_score'].append(f1)
        
        # Print the results for each fold
        print(f"{fold + 1:<5} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

    # Calculate and print the average metrics
    avg_accuracy = np.mean(fold_results['accuracy'])
    avg_precision = np.mean(fold_results['precision'])
    avg_recall = np.mean(fold_results['recall'])
    avg_f1 = np.mean(fold_results['f1_score'])
    
    # Print the average metrics
    print(f"{'Avg':<5} {avg_accuracy:<10.4f} {avg_precision:<10.4f} {avg_recall:<10.4f} {avg_f1:<10.4f}")

    return fold_results

if __name__ == "__main__":
    data_path = "../data/classes"
    results = k_fold_cross_validation(data_path, k=10)
