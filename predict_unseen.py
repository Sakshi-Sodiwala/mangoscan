import os
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Class names (same order as your training folder)
classes = ['Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Unseen image folder
unseen_folder = "unseen"

# Predict and show each image
for img_name in os.listdir(unseen_folder):
    img_path = os.path.join(unseen_folder, img_name)
    if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = classes[pred.item()]

        # Show image with predicted label
        plt.imshow(image)
        plt.title(f"{img_name} ➜ {predicted_class}")
        plt.axis('off')
        plt.show()
