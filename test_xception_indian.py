import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Device
device = torch.device("cpu")  # or "mps" if you want

# Test transforms (SAME as training)
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Indian dataset path
test_dir = "/Users/sachita/Desktop/image"

# Load dataset
test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

print("Classes:", test_dataset.classes)
print("Total Images:", len(test_dataset))
# Create model
model = timm.create_model(
    'xception',
    pretrained=False,
    num_classes=2
)

# Load weights
model.load_state_dict(
    torch.load(
        "best_xception.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")
all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total

print(f"\nIndian Dataset Accuracy: {accuracy:.2f}%")
print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.classes
    )
)
cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\nConfusion Matrix:")
print(cm)