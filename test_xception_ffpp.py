import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dir = "/Users/sachita/Desktop/MEITY INTERNSHIP/DeepFake Detection model/dataset/test"

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Classes:", test_dataset.classes)
print("Test Images:", len(test_dataset))
model = timm.create_model(
    'xception',
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "best_xception.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Best model loaded successfully!")
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

test_acc = 100 * correct / total

print(f"\nTest Accuracy: {test_acc:.2f}%")