import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def train_model(model, train_loader, valid_loader, device):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    patience = 5
    best_valid_loss = float('inf')
    counter = 0

    # Training loop
    for epoch in range(50):
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

        print(
            f"Epoch {epoch + 1}: Train Loss = {loss.item():.4f}, Valid Accuracy = {valid_accuracy:.4f}, Valid F1 Score = {valid_f1:.4f}")

        # Check for early stopping and save the best model
        valid_loss = np.mean(valid_losses)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Updated best model at epoch {epoch + 1}, Validation Loss = {valid_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    print("Training finished.")
