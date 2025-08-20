import os
import cv2
import shutil
import random
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === PARAMETRI ===
RAW_DIR = Path("../data/raw")    # cartella con immagini originali (cat/, dog/, other/)
OUT_DIR = Path("../data")        # cartella finale con train/ val/ test/
IMG_SIZE = (224, 224)
SPLIT_RATIO = (0.7, 0.2, 0.1)  # train, val, test

def preprocess_and_split():
    """Legge immagini raw, le preprocessa e le divide in train/val/test."""
    for label in os.listdir(RAW_DIR):
        input_dir = RAW_DIR / label
        if not input_dir.is_dir():
            continue

        # raccogli tutte le immagini
        files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))]
        random.shuffle(files)
        total = len(files)

        # split
        train_split = int(total * SPLIT_RATIO[0])
        val_split = int(total * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))

        splits = {
            "train": files[:train_split],
            "val": files[train_split:val_split],
            "test": files[val_split:]
        }

        # salva immagini preprocessate
        for split, subset in splits.items():
            out_dir = OUT_DIR / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            for fname in subset:
                img_path = input_dir / fname
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                save_path = out_dir / fname
                cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"✅ {label}: {total} immagini preprocessate e divise in train/val/test")

def get_datagens(out_directory, batch_size=32):
    """Restituisce i generatori Keras per train/val/test."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        out_directory / "train",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_gen = test_val_datagen.flow_from_directory(
        out_directory / "val",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary"
    )

    test_gen = test_val_datagen.flow_from_directory(
        out_directory / "test",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    preprocess_and_split()
    print("🎉 Preprocessing completato. Ora puoi usare get_datagens() per il training.")
