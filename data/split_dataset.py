import os
import shutil
import random
from pathlib import Path

# === PARAMETRI ===
input_dir = Path("../original_dataset_from_kaggle")  # cartella con cat*.jpg e dog*.jpg
output_dir = Path(".")    # dove verranno salvate le cartelle finali
split_ratio = (0.7, 0.2, 0.1)  # train, val, test

# === FUNZIONE PER SPLIT ===
def split_and_copy(img_list, label):
    random.shuffle(img_list)
    total = len(img_list)
    train_split = int(total * split_ratio[0])
    val_split = int(total * (split_ratio[0] + split_ratio[1]))

    splits = {
        "train": img_list[:train_split],
        "val": img_list[train_split:val_split],
        "test": img_list[val_split:]
    }

    for split, files in splits.items():
        split_dir = output_dir / split / label
        split_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy(f, split_dir / Path(f).name)

    print(f"✅ {label}: {total} immagini divise in train/val/test")

# === MAIN ===
if __name__ == "__main__":
    cats = [input_dir / f for f in os.listdir(input_dir) if f.startswith("cat")]
    dogs = [input_dir / f for f in os.listdir(input_dir) if f.startswith("dog")]

    split_and_copy(cats, "cat")
    split_and_copy(dogs, "dog")

    print("🎉 Dataset pronto in:", output_dir.resolve())
