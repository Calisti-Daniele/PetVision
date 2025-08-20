from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = Path("data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📡 Device: {device}")

# -------------------------
# Funzioni di utilità
# -------------------------
def get_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_ds   = datasets.ImageFolder(data_dir / "val", transform=transform)
    test_ds  = datasets.ImageFolder(data_dir / "test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print("Classi:", train_ds.classes)
    return train_loader, val_loader, test_loader, train_ds.classes


def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    epoch_loss, epoch_correct = 0.0, 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        epoch_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return epoch_loss / total, epoch_correct / total


# -------------------------
# Main script
# -------------------------
def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)

    # Carica MobileNetV2 pretrained
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False  # congela feature extractor

    # Sostituisci classificatore
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion)

        print(f"📅 Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Salva modello
    torch.save(model.state_dict(), "models/mobilenetv2.pth")

    # Valutazione finale
    test_loss, test_acc = run_epoch(model, test_loader, criterion)
    print(f"🎯 Test accuracy: {test_acc:.2f}")


if __name__ == "__main__":
    main()
