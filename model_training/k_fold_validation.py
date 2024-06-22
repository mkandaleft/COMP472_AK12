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

    model = SimpleCNN(num_classes=4).to(device)
    train_model(model, train_loader, val_loader, device)
    
    return model, train_loader, val_loader

def compute_metrics(loader, model, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

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
        'macro_precision': [],
        'macro_recall': [],
        'macro_f1': [],
        'micro_precision': [],
        'micro_recall': [],
        'micro_f1': []
    }
    
    # Print the header of the table
    print(f"{'Fold':<5} {'Accuracy':<10} {'Macro P':<10} {'Macro R':<10} {'Macro F':<10} {'Micro P':<10} {'Micro R':<10} {'Micro F':<10}")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{k}')
        
        model, train_loader, val_loader = train_and_validate_model(train_idx, dataset, device)
        
        # Load the best model state for evaluation
        # model.load_state_dict(torch.load('best_model.pth'))
        
        # Evaluate the model on the validation set
        accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = compute_metrics(val_loader, model, device)
        
        fold_results['accuracy'].append(accuracy)
        fold_results['macro_precision'].append(macro_precision)
        fold_results['macro_recall'].append(macro_recall)
        fold_results['macro_f1'].append(macro_f1)
        fold_results['micro_precision'].append(micro_precision)
        fold_results['micro_recall'].append(micro_recall)
        fold_results['micro_f1'].append(micro_f1)
        
        # Print the results for each fold
        print(f"{fold + 1:<5} {accuracy:<10.4f} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f} {micro_precision:<10.4f} {micro_recall:<10.4f} {micro_f1:<10.4f}")

    avg_accuracy = np.mean(fold_results['accuracy'])
    avg_macro_precision = np.mean(fold_results['macro_precision'])
    avg_macro_recall = np.mean(fold_results['macro_recall'])
    avg_macro_f1 = np.mean(fold_results['macro_f1'])
    avg_micro_precision = np.mean(fold_results['micro_precision'])
    avg_micro_recall = np.mean(fold_results['micro_recall'])
    avg_micro_f1 = np.mean(fold_results['micro_f1'])
    
    # Print the average metrics
    print(f"{'Avg':<5} {avg_accuracy:<10.4f} {avg_macro_precision:<10.4f} {avg_macro_recall:<10.4f} {avg_macro_f1:<10.4f} {avg_micro_precision:<10.4f} {avg_micro_recall:<10.4f} {avg_micro_f1:<10.4f}")

    return fold_results

if __name__ == "__main__":
    data_path = "../data/classes"
    results = k_fold_cross_validation(data_path, k=10)
