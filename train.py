import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

SPLIT_DIR   = "data/split"
MODELS_DIR  = "models"
NUM_CLASSES = 5

BATCH_SIZE = 32
EPOCHS = 100
LR         = 1e-3
IMG_SIZE   = 224
PATIENCE   = 10

def get_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(SPLIT_DIR, "val"),   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}\n")

    train_loader, val_loader, train_dataset, val_dataset = get_loaders()

    print("類別對應：")
    for cls, idx in train_dataset.class_to_idx.items():
        print(f"  {idx}: {cls}")
    print()

    class_counts  = [len(os.listdir(os.path.join(SPLIT_DIR, "train", c))) for c in train_dataset.classes]
    total         = sum(class_counts)
    class_weights = torch.tensor([total / (NUM_CLASSES * c) for c in class_counts], dtype=torch.float).to(device)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()

        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_dataset) * 100
        val_acc   = val_correct   / len(val_dataset)   * 100
        print(f"  Train Loss: {train_loss/len(train_dataset):.4f}  Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss/len(val_dataset):.4f}  Acc: {val_acc:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
            print(f"  --> 儲存最佳模型（Val Acc: {val_acc:.2f}%）\n")
        else:
            no_improve += 1
            print(f"  Early Stopping 計數：{no_improve}/{PATIENCE}\n")
            if no_improve >= PATIENCE:
                print(f"Val Acc 連續 {PATIENCE} 個 epoch 未進步，提早停止訓練")
                break

    print(f"訓練完成！最佳 Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()
