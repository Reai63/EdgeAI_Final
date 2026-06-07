import os
import shutil
import random

PROCESSED_DIR = "data/processed"
SPLIT_DIR = "data/split"
CLASSES = ["plastic", "paper", "metal", "glass", "trash"]

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

random.seed(42)

for cls in CLASSES:
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

print("切割資料集中...\n")

for cls in CLASSES:
    src_dir = os.path.join(PROCESSED_DIR, cls)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:],
    }

    for split, split_files in splits.items():
        for f in split_files:
            shutil.copy2(
                os.path.join(src_dir, f),
                os.path.join(SPLIT_DIR, split, cls, f)
            )

    print(f"  {cls}: train={len(splits['train'])} / val={len(splits['val'])} / test={len(splits['test'])}")

print("\n切割完成！")
