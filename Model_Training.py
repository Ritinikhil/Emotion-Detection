import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fer_root   = r"C:\Users\sys\PycharmProjects\Major_Project\dataset"           # FER2013 root (train/test/<class>)
raf_root   = r"C:\Users\sys\PycharmProjects\Major_Project\dataset-2\DATASET" # RAF-DB root (train/test/1..7)
output_dir = r"C:\Users\sys\PycharmProjects\Major_Project\experiments_joint"

os.makedirs(output_dir, exist_ok=True)

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# MODEL
# =========================

class SimpleFERCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# FER2013 DATASETS
# =========================

fer_train_dir = os.path.join(fer_root, "train")
fer_test_dir  = os.path.join(fer_root, "test")

fer_train_ds = datasets.ImageFolder(fer_train_dir, transform=transform_train)
fer_test_ds  = datasets.ImageFolder(fer_test_dir,  transform=transform_test)

fer_train_loader = DataLoader(fer_train_ds, batch_size=64, shuffle=True,  num_workers=0)
fer_test_loader  = DataLoader(fer_test_ds,  batch_size=64, shuffle=False, num_workers=0)

print("FER classes:", fer_train_ds.classes)

# =========================
# RAF-DB DATASETS (remapped)
# =========================

RAF_NUM_TO_EMO = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}
emo_to_idx = {e: i for i, e in enumerate(EMOTION_CLASSES)}

class RAFFolderRemapped(Dataset):
    def __init__(self, root, transform=None):
        self.inner = datasets.ImageFolder(root, transform=transform)
        self.transform = transform
        self.folder_classes = self.inner.classes
        self.inner_to_unified = {}
        for inner_idx, folder_name in enumerate(self.folder_classes):
            emo_name = RAF_NUM_TO_EMO[folder_name]
            self.inner_to_unified[inner_idx] = emo_to_idx[emo_name]

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, inner_label = self.inner[idx]
        unified_label = self.inner_to_unified[inner_label]
        return img, unified_label

raf_train_dir = os.path.join(raf_root, "train")
raf_test_dir  = os.path.join(raf_root, "test")

raf_train_ds = RAFFolderRemapped(raf_train_dir, transform=transform_train)
raf_test_ds  = RAFFolderRemapped(raf_test_dir,  transform=transform_test)

raf_train_loader = DataLoader(raf_train_ds, batch_size=64, shuffle=True,  num_workers=0)
raf_test_loader  = DataLoader(raf_test_ds,  batch_size=64, shuffle=False, num_workers=0)

print("RAF folders:", raf_train_ds.folder_classes)

# =========================
# GENERIC TRAIN / EVAL
# =========================

def train_and_eval(name, train_loader, test_loader, num_epochs=30):
    model = SimpleFERCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    best_model_path = os.path.join(output_dir, f"{name}_best_model.pth")

    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total if total > 0 else 0.0
        train_acc  = correct / total if total > 0 else 0.0

        # Eval
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_loss = running_loss / total if total > 0 else 0.0
        test_acc  = correct / total if total > 0 else 0.0

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"[{name}] Epoch {epoch+1:02d}/{num_epochs} "
              f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
              f"- Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> [{name}] Saved new best model to {best_model_path}")

    # Save history & plot
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(output_dir, f"{name}_history.csv")
    hist_df.to_csv(hist_csv, index=False)

    fig_path = os.path.join(output_dir, f"{name}_loss_acc.png")
    epochs = history["epoch"]
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["test_loss"],  label="Test loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{name} Loss")
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["test_acc"],  label="Test acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{name} Accuracy")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[{name}] Best test acc: {best_acc:.4f}")
    print(f"[{name}] History saved to: {hist_csv}")
    print(f"[{name}] Plot saved to:    {fig_path}")

    return best_acc

# =========================
# RUN BOTH TRAININGS
# =========================

fer_best = train_and_eval("FER_only", fer_train_loader, fer_test_loader, num_epochs=30)
raf_best = train_and_eval("RAF_only", raf_train_loader, raf_test_loader, num_epochs=30)

# =========================
# SUMMARY VISUALIZATION
# =========================

summary_df = pd.DataFrame({
    "model": ["FER_only", "RAF_only"],
    "best_test_acc": [fer_best, raf_best],
})
summary_csv = os.path.join(output_dir, "summary_fer_raf.csv")
summary_df.to_csv(summary_csv, index=False)

fig_sum = os.path.join(output_dir, "summary_fer_raf.png")
plt.figure(figsize=(5,4))
plt.bar(summary_df["model"], summary_df["best_test_acc"], color=["steelblue", "orange"])
plt.ylim(0,1.0)
for i, v in enumerate(summary_df["best_test_acc"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
plt.ylabel("Best test accuracy")
plt.title("FER-only vs RAF-only best accuracy")
plt.tight_layout()
plt.savefig(fig_sum, dpi=300)
plt.close()

print("Summary CSV saved to:", summary_csv)
print("Summary plot saved to:", fig_sum)
