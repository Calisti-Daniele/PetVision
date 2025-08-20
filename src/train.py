import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path

print("🔥 CUDA disponibile:", torch.cuda.is_available())
print("🧠 Device in uso:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# === PARAMETRI ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = Path("data")  # stessa struttura: train/, val/, test/

# === TRASFORMAZIONI ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet
                         std=[0.229, 0.224, 0.225])
])

# === DATASET ===
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
val_dataset   = datasets.ImageFolder(DATA_DIR / "val",   transform=transform)
test_dataset  = datasets.ImageFolder(DATA_DIR / "test",  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === INFO CLASSI ===
num_classes = len(train_dataset.classes)
print("Classi:", train_dataset.classes)

# === MODELLO CNN BASE ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE//4) * (IMG_SIZE//4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📡 Device in uso:", device)

model = SimpleCNN(num_classes).to(device)

# === LOSS & OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === TRAINING LOOP ===
for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # valutazione su val
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_corrects += (outputs.argmax(1) == labels).sum().item()
    val_acc = val_corrects / len(val_dataset)
    print(f"   🔎 Val Accuracy: {val_acc:.4f}")

# === TEST FINALE ===
model.eval()
test_corrects = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_corrects += (outputs.argmax(1) == labels).sum().item()

test_acc = test_corrects / len(test_dataset)
print(f"🎯 Test Accuracy: {test_acc:.4f}")

# === SALVATAGGIO MODELLO ===
torch.save(model.state_dict(), "models/cnn_base_torch.pth")
print("✅ Modello salvato in models/cnn_base_torch.pth")
