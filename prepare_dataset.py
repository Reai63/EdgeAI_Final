import os
import shutil

TRASHNET_DIR  = "data/raw/trashnet"
KAGGLE_DIR    = "data/raw/garbage_classification"
OUTPUT_DIR    = "data/processed"

MERGE_MAP = {
    "plastic": [
        (TRASHNET_DIR,  "plastic"),
        (KAGGLE_DIR,    "plastic"),
    ],
    "paper": [
        (TRASHNET_DIR,  "paper"),
        (KAGGLE_DIR,    "paper"),
    ],
    "metal": [
        (TRASHNET_DIR,  "metal"),
        (KAGGLE_DIR,    "metal"),

    ],
    "glass": [
        (TRASHNET_DIR,  "glass"),
        (KAGGLE_DIR,    "green-glass"),
        (KAGGLE_DIR,    "brown-glass"),
        (KAGGLE_DIR,    "white-glass"),
    ],
    "trash": [
        (TRASHNET_DIR,  "trash"),
        (KAGGLE_DIR,    "trash"),
    ],
}

for cls in MERGE_MAP:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

print("整合資料集中...\n")

for target_cls, sources in MERGE_MAP.items():
    dst_dir = os.path.join(OUTPUT_DIR, target_cls)
    count = 0

    for src_root, src_cls in sources:
        src_dir = os.path.join(src_root, src_cls)
        if not os.path.exists(src_dir):
            print(f"  警告：找不到 {src_dir}，跳過")
            continue

        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in files:
            new_name = f"{src_cls}_{count:05d}{os.path.splitext(f)[1]}"
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, new_name))
            count += 1

    print(f"  {target_cls}: {count} 張")

print("\n整合完成！")
