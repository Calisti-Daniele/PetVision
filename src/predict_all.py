# predict_all.py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = Path("data/test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📡 Device:", device)


# -------------------------
# Modello base CNN
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# Loader immagini test
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_ds = datasets.ImageFolder(DATA_DIR.parent / "test", transform=transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
classes = test_ds.classes
num_classes = len(classes)

print("Classi:", classes)


# -------------------------
# Funzione valutazione
# -------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0


# -------------------------
# Carica e valuta entrambi i modelli
# -------------------------
# 1) SimpleCNN
cnn_model = SimpleCNN(num_classes).to(device)
cnn_model.load_state_dict(torch.load("models/cnn_base_torch.pth", map_location=device))
acc_cnn = evaluate(cnn_model, test_loader)
print(f"🧠 SimpleCNN accuracy: {acc_cnn*100:.2f}%")

# 2) MobileNetV2 Transfer Learning
mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
mobilenet.load_state_dict(torch.load("models/mobilenetv2.pth", map_location=device))
mobilenet = mobilenet.to(device)

acc_transfer = evaluate(mobilenet, test_loader)
print(f"🚀 MobileNetV2 Transfer accuracy: {acc_transfer*100:.2f}%")
