import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# CONFIG
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raf_root   = r"C:\Users\sys\PycharmProjects\Major_Project\dataset-2\DATASET"
output_dir = r"C:\Users\sys\PycharmProjects\Major_Project\raf_only"

raf_train_dir = os.path.join(raf_root, "train")   # folders 1..7
raf_test_dir  = os.path.join(raf_root, "test")    # folders 1..7

os.makedirs(output_dir, exist_ok=True)

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Mapping RAF folder number -> emotion name
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
# DATASET WRAPPER FOR RAF
# =========================

class RAFFolderRemapped(Dataset):
    """
    Wraps ImageFolder on RAF train/test (folders '1'..'7') and remaps
    labels to standard EMOTION_CLASSES indices.
    """
    def __init__(self, root, transform=None):
        self.inner = datasets.ImageFolder(root, transform=transform)
        self.transform = transform
        self.folder_classes = self.inner.classes   # e.g. ['1','2',...,'7']

        self.inner_to_unified = {}
        for inner_idx, folder_name in enumerate(self.folder_classes):
            if folder_name not in RAF_NUM_TO_EMO:
                raise ValueError(f"Unknown RAF class folder: {folder_name}")
            emo_name = RAF_NUM_TO_EMO[folder_name]
            self.inner_to_unified[inner_idx] = emo_to_idx[emo_name]

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, inner_label = self.inner[idx]
        unified_label = self.inner_to_unified[inner_label]
        return img, unified_label

# Instantiate
raf_train_ds = RAFFolderRemapped(raf_train_dir, transform=transform_train)
raf_test_ds  = RAFFolderRemapped(raf_test_dir,  transform=transform_test)

raf_train_loader = DataLoader(raf_train_ds, batch_size=64, shuffle=True,  num_workers=0)
raf_test_loader  = DataLoader(raf_test_ds,  batch_size=64, shuffle=False, num_workers=0)

print("RAF train size:", len(raf_train_ds), "RAF test size:", len(raf_test_ds))
print("RAF folders:", raf_train_ds.folder_classes)

# =========================
# MODEL
# =========================

class SimpleFERCNN(nn.Module):
    """
    Same architecture as before, trained only on RAF.
    """
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

model = SimpleFERCNN(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =========================
# TRAIN / EVAL
# =========================

history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

def train_one_epoch(model, loader):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
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
    return running_loss / total, correct / total

def evaluate(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

num_epochs = 30
best_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, raf_train_loader)
    test_loss, test_acc   = evaluate(model, raf_test_loader)

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    print(f"Epoch {epoch+1:02d}/{num_epochs} "
          f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
          f"- Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        best_path = os.path.join(output_dir, "raf_only_best_model.pth")
        torch.save(model.state_dict(), best_path)
        print("  -> Saved new best model to", best_path)

print("Best RAF test accuracy:", best_acc)

# =========================
# SAVE HISTORY & PLOTS
# =========================

history_df = pd.DataFrame(history)
hist_csv = os.path.join(output_dir, "raf_only_history.csv")
history_df.to_csv(hist_csv, index=False)
print("Saved training history to:", hist_csv)

epochs = history["epoch"]

fig1_path = os.path.join(output_dir, "raf_only_loss_acc.png")
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, history["train_loss"], label="Train loss")
plt.plot(epochs, history["test_loss"],  label="Test loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("RAF-only Loss")
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs, history["train_acc"], label="Train acc")
plt.plot(epochs, history["test_acc"],  label="Test acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("RAF-only Accuracy")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(fig1_path, dpi=300)
plt.close()
print("Saved RAF-only loss/accuracy plot to:", fig1_path)
