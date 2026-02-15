import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

# Charger un modèle ResNet pré-entraîné
image_model = resnet18(pretrained=True)
image_model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def analyze_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = image_model(image_tensor)
    return output.argmax(1).item()  # Retourner la classe prédite
