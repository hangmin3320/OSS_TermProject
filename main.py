import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms

device = torch.device("cpu") # 자원 소모를 줄이기 위해 cpu 사용

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, momentum=0.05),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, momentum=0.05),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x

IMAGE_PATH = 'test.jpeg'
MODEL_PATH = "result/PreTrained_ResNet50_SavemodelTest_dataset.pt"
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def load_and_prepare_image(image_path, device):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)


def load_model(model_path, device='cpu'):
    model_data = torch.load(model_path, map_location=torch.device(device))
    model = model_data["model"]
    model.eval()
    return model


def predict_image_class(image, model):
    with torch.no_grad():
        output = model(image)
    return torch.argmax(output).item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image = load_and_prepare_image(IMAGE_PATH, device)
model = load_model(MODEL_PATH)
predicted_class_index = predict_image_class(image, model)
predicted_class_name = CLASS_NAMES[predicted_class_index]
print(f"예측 결과: {predicted_class_name}")