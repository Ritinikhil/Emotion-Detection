import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

combined_csv = r"C:\Users\sys\PycharmProjects\Major_Project\combined.csv"
fer_root     = r"C:\Users\sys\PycharmProjects\Major_Project\dataset"
raf_root     = r"C:\Users\sys\PycharmProjects\Major_Project\dataset-2\DATASET"
output_dir   = r"C:\Users\sys\PycharmProjects\Major_Project"

model_path   = os.path.join(output_dir, "curriculum_model.pth")

fer_test_dir = os.path.join(fer_root, "test")
raf_train_dir = os.path.join(raf_root, "train")   # use this as RAF eval set (or change to 'test')

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

transform_eval = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

os.makedirs(output_dir, exist_ok=True)

# =========================
# MODEL (same as training)
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

model = SimpleFERCNN(num_classes=7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Loaded model from:", model_path)

# =========================
# DATASETS
# =========================

# FER test loader
fer_test_ds = datasets.ImageFolder(fer_test_dir, transform=transform_eval)
fer_test_loader = DataLoader(fer_test_ds, batch_size=64, shuffle=False, num_workers=0)

# RAF eval loader (using folder labels, but labels already mapped during training)
raf_eval_ds = datasets.ImageFolder(raf_train_dir, transform=transform_eval)
raf_eval_loader = DataLoader(raf_eval_ds, batch_size=64, shuffle=False, num_workers=0)

# Combined dataset with difficulty (for bin analysis)
df_combined = pd.read_csv(combined_csv)

class CombinedEvalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("L")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        diff  = float(row["difficulty"])
        return img, label, diff

combined_eval_ds = CombinedEvalDataset(df_combined, transform=transform_eval)
combined_eval_loader = DataLoader(combined_eval_ds, batch_size=64, shuffle=False, num_workers=0)

# =========================
# EVAL HELPERS
# =========================

def eval_simple(loader, model):
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = ce(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = total_correct / total if total > 0 else 0.0
    loss = total_loss / total if total > 0 else 0.0
    return loss, acc, np.array(all_preds), np.array(all_labels)

def eval_by_difficulty_bins(ds, model, bins=(0.0, 0.3, 0.6, 1.0)):
    model.eval()
    # Pre-compute logits for whole dataset
    all_logits, all_labels, all_diffs = [], [], []
    with torch.no_grad():
        for images, labels, diffs in DataLoader(ds, batch_size=64, shuffle=False, num_workers=0):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_diffs.append(diffs)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_diffs  = torch.cat(all_diffs,  dim=0).numpy()
    preds = all_logits.argmax(dim=1).numpy()

    results = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (all_diffs >= lo) & (all_diffs < hi)
        if mask.sum() == 0:
            acc = None
        else:
            acc = (preds[mask] == all_labels[mask]).mean()
        results.append({"bin": f"[{lo:.1f}, {hi:.1f})", "count": int(mask.sum()), "acc": acc})
    return results

# =========================
# RUN EVALUATIONS
# =========================

fer_loss, fer_acc, _, _ = eval_simple(fer_test_loader, model)
print(f"FER2013 test -> loss: {fer_loss:.4f}, acc: {fer_acc:.4f}")

raf_loss, raf_acc, _, _ = eval_simple(raf_eval_loader, model)
print(f"RAF-DB eval  -> loss: {raf_loss:.4f}, acc: {raf_acc:.4f}")

bin_results = eval_by_difficulty_bins(combined_eval_ds, model)
print("Difficulty-bin results:")
for r in bin_results:
    print(r)

# Save numeric results
results_df = pd.DataFrame({
    "dataset": ["FER2013_test", "RAF_eval"],
    "loss": [fer_loss, raf_loss],
    "acc":  [fer_acc,  raf_acc],
})
bins_df = pd.DataFrame(bin_results)

results_csv = os.path.join(output_dir, "eval_summary.csv")
bins_csv    = os.path.join(output_dir, "eval_by_difficulty.csv")
results_df.to_csv(results_csv, index=False)
bins_df.to_csv(bins_csv, index=False)
print("Saved eval_summary to:", results_csv)
print("Saved eval_by_difficulty to:", bins_csv)

# =========================
# VISUALIZATIONS
# =========================

# 1) Bar plot: dataset accuracies
fig1_path = os.path.join(output_dir, "eval_datasets_accuracy.png")
plt.figure(figsize=(5,4))
plt.bar(["FER2013_test", "RAF_eval"], [fer_acc, raf_acc], color=["steelblue", "orange"])
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Accuracy on FER2013 test vs RAF-DB")
for i, v in enumerate([fer_acc, raf_acc]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig(fig1_path, dpi=300)
plt.close()
print("Saved dataset accuracy plot to:", fig1_path)

# 2) Bar plot: difficulty-bin accuracy
fig2_path = os.path.join(output_dir, "eval_difficulty_bins.png")
labels_bins = [r["bin"] for r in bin_results]
acc_bins    = [0.0 if r["acc"] is None else r["acc"] for r in bin_results]
counts_bins = [r["count"] for r in bin_results]

plt.figure(figsize=(6,4))
bars = plt.bar(labels_bins, acc_bins, color="seagreen")
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.xlabel("Difficulty bin")
plt.title("Accuracy by difficulty bins")
for b, a, c in zip(bars, acc_bins, counts_bins):
    plt.text(b.get_x() + b.get_width()/2, a + 0.01, f"{a:.2f}\n(n={c})", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(fig2_path, dpi=300)
plt.close()
print("Saved difficulty-bin accuracy plot to:", fig2_path)
