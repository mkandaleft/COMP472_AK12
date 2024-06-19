

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def get_loaders(data_path, batch_size=64):
    # Define the transforms for data preprocessing
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4926176471], std=[0.2526960784])
    ])

    # Load the dataset
    dataset = ImageFolder(data_path, transform=transform)

    # Adjust proportions for training, validation, and testing
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    # Create a generator with a fixed seed
    generator = torch.Generator().manual_seed(10)

    # Split the dataset
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size], generator=generator)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader