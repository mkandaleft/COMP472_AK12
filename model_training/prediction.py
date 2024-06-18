from PIL import Image
from mainModel import SimpleCNN
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4926176471], std=[0.2526960784])
])


# Function to predict on img
def predict_image(model, image_tensor, device, class_labels):
    image_tensor = image_tensor.to(device)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]


# Function to load the image using transformation from dataLoader
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure correct image mode
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


if __name__ == "__main__":
    model_path = 'best_model.pth'
    image_path = '../data/testing/image0023281.jpg'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = SimpleCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load transformed img
    image_tensor = load_image(image_path)

    class_labels = ["angry", "focused", "happy", "neutral"]

    # Make prediction on img
    prediction = predict_image(model, image_tensor, device, class_labels)
    print(f'Predicted class: {prediction}')
