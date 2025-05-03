import torch
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model = resnet18(weights=None)  # Do NOT use pretrained=True
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model = model.to(device)

# Load trained weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Transforms (must match validation/test transforms)
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load test dataset
test_dataset = ImageFolder("dataset/test", transform=val_test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32)

# Evaluate accuracy
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
