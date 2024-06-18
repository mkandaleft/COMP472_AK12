from PIL import Image
from sklearn.metrics import accuracy_score

from mainModel import SimpleCNN
import torch
import torchvision.transforms as transforms
import torch.nn.functional as Funct

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4926176471], std=[0.2526960784])
])


# Function to predict on img
def predict_image(model, image_tensor, device, class_labels):
    image_tensor = image_tensor.to(device)
    model.to(device)
    model.eval()  # Set eval mode
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = Funct.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_labels[predicted.item()], confidence.item()


# Function to load the image using transformation from dataLoader
def load_image(image_path):

    image = Image.open(image_path).convert('RGB') # Match model 
    image = transform(image).unsqueeze(0)

    return image


# Function to calculate accuracy
# def calculate_accuracy(model, data_loader, device):
#     model.to(device)
#     model.eval()
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for images, labels in data_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     accuracy = accuracy_score(all_labels, all_preds)
#     return accuracy

if __name__ == "__main__":
    model_path = 'best_model.pth'
    image_path = '../data/testing/ffhq_2044.png'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = SimpleCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load transformed img
    image_tensor = load_image(image_path)

    class_labels = ["angry", "focused", "happy", "neutral"]

    # Make prediction on img
    predicted_class, confidence_score = predict_image(model, image_tensor, device, class_labels)
    print(f'Predicted class: {predicted_class}, Confidence Score = {confidence_score:.4f}')
