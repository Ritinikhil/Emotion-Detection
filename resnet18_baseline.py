# resnet18_baseline.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ⚠️ UPDATE THESE PATHS IF NEEDED
raf_root = r"C:\Users\sys\PycharmProjects\Major_Project\dataset-2\DATASET"
output_dir = r"C:\Users\sys\PycharmProjects\Major_Project\experiments_resnet"

os.makedirs(output_dir, exist_ok=True)

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========================
# RAF-DB DATASET (Same as your code)
# ========================

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
raf_test_dir = os.path.join(raf_root, "test")

raf_train_ds = RAFFolderRemapped(raf_train_dir, transform=transform_train)
raf_test_ds = RAFFolderRemapped(raf_test_dir, transform=transform_test)

raf_train_loader = DataLoader(raf_train_ds, batch_size=64, shuffle=True, num_workers=0)
raf_test_loader = DataLoader(raf_test_ds, batch_size=64, shuffle=False, num_workers=0)

print("RAF-DB loaded successfully!")
print(f"Train samples: {len(raf_train_ds)}")
print(f"Test samples: {len(raf_test_ds)}")


# ========================
# ResNet-18 Model (adapted for grayscale)
# ========================

class ResNet18_Grayscale(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18_Grayscale, self).__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=False)

        # Adapt first conv layer for grayscale (1 channel instead of 3)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final FC layer for 7 emotions
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)

        self.model = resnet

    def forward(self, x):
        return self.model(x)


# ========================
# TRAINING FUNCTION (Same as your code)
# ========================

def train_and_eval(name, train_loader, test_loader, num_epochs=30):
    """Train and evaluate model"""

    model = ResNet18_Grayscale(num_classes=7).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    best_model_path = os.path.join(output_dir, f"{name}_best_model.pth")

    for epoch in range(num_epochs):
        # ============ TRAINING PHASE ============
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
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

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss={loss.item():.4f}")

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        # ============ EVALUATION PHASE ============
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
        test_acc = correct / total if total > 0 else 0.0

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"[{name}] Epoch {epoch + 1:02d}/{num_epochs} "
              f"| Train: loss={train_loss:.4f}, acc={train_acc:.4f} "
              f"| Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

        # Adjust learning rate
        scheduler.step(test_loss)

        # ============ SAVE BEST MODEL ============
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved! Accuracy: {test_acc:.4f}")

    # ============ SAVE RESULTS ============
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(output_dir, f"{name}_history.csv")
    hist_df.to_csv(hist_csv, index=False)

    # Plot training curves
    fig_path = os.path.join(output_dir, f"{name}_loss_acc.png")
    epochs = history["epoch"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss", marker='o')
    plt.plot(epochs, history["test_loss"], label="Test loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} - Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train acc", marker='o')
    plt.plot(epochs, history["test_acc"], label="Test acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} - Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"\n{'=' * 60}")
    print(f"[{name}] BEST TEST ACCURACY: {best_acc:.4f} ({best_acc * 100:.2f}%)")
    print(f"[{name}] History saved to: {hist_csv}")
    print(f"[{name}] Plot saved to:    {fig_path}")
    print(f"[{name}] Model saved to:   {best_model_path}")
    print(f"{'=' * 60}\n")

    return best_acc


print("Starting RAF-DB training with ResNet-18...")
print("=" * 60)

resnet_best = train_and_eval("ResNet18", raf_train_loader, raf_test_loader, num_epochs=30)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"ResNet-18 Best accuracy: {resnet_best:.4f} ({resnet_best * 100:.2f}%)")
print(f"Results saved to: {output_dir}")
print("=" * 60)
