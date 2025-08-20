# predict_torch.py
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from PIL import Image

IMG_SIZE = 224

# -----------------------------
# Modelli supportati
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
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

def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "simplecnn":
        return SimpleCNN(num_classes)
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        # sostituisci la testa per il numero di classi
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Architettura non supportata: {arch}. Usa 'simplecnn' o 'mobilenet_v2'.")

# -----------------------------
# Preprocess immagine
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet
                         std=[0.229, 0.224, 0.225]),
])

def load_image_tensor(img_path: Path) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # shape: [1, 3, H, W]
    return tensor

# -----------------------------
# Predizione
# -----------------------------
def predict(model_path: Path, img_path: Path, data_train_dir: Path, arch: str = "simplecnn"):
    # class labels in base alle cartelle (ordine alfabetico come ImageFolder)
    train_ds = datasets.ImageFolder(data_train_dir)
    classes = train_ds.classes  # es: ['cat', 'dog'] o ['cat', 'dog', 'other']
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📡 Device:", device)

    # ricostruisci il modello e carica i pesi
    model = build_model(arch, num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # prepara immagine
    x = load_image_tensor(img_path).to(device)

    # inferenza
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = classes[pred_idx.item()]
    print(f"📸 Immagine: {img_path}")
    print(f"🔮 Predizione: {label} (confidence {conf.item():.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Utilizzo:")
        print("  python predict_torch.py <modello.pth> <immagine.jpg> [data/train] [arch]")
        print("Esempi:")
        print("  python predict_torch.py models/cnn_base_torch.pth test_images/dog1.jpg")
        print("  python predict_torch.py models/mobilenetv2_torch.pth test_images/cat1.jpg data/train mobilenet_v2")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    img_path = Path(sys.argv[2])
    data_train_dir = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path("data/train")
    arch = sys.argv[4] if len(sys.argv) == 5 else "simplecnn"

    predict(model_path, img_path, data_train_dir, arch)
