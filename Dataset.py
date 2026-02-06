import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 0. CONFIG
# =========================

# Unified emotion order we will use everywhere:
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
label_to_idx = {name: i for i, name in enumerate(EMOTION_CLASSES)}

# FER2013 root (already in folders train/test/<class_name>)
fer_root  = r"C:\Users\sys\PycharmProjects\Major_Project\dataset"
fer_train_dir = os.path.join(fer_root, "train")
fer_test_dir  = os.path.join(fer_root, "test")

# RAF-DB root (your screenshot)
raf_root  = r"C:\Users\sys\PycharmProjects\Major_Project\dataset-2\DATASET"
raf_train_dir = os.path.join(raf_root, "train")   # contains 1..7
raf_test_dir  = os.path.join(raf_root, "test")    # contains 1..7

combined_index_csv = r"C:\Users\sys\PycharmProjects\Major_Project\combined.csv"

common_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# 1. GENERIC REMAPPED FOLDER DATASET
# =========================

class RemappedFolderDataset(Dataset):
    """
    Wraps an ImageFolder and remaps its class indices to EMOTION_CLASSES.
    For FER: folder names are 'angry', ...
    For RAF: folder names are '1','2',...,'7' (we map them manually).
    """
    def __init__(self, root, dataset_type, transform=None):
        """
        dataset_type: 'fer' or 'raf'
        """
        self.inner = datasets.ImageFolder(root, transform=transform)
        self.transform = transform
        self.dataset_type = dataset_type
        self.folder_classes = self.inner.classes  # list of folder names

        self.inner_to_unified = {}

        if dataset_type == "fer":
            # folder names are emotion strings
            for inner_idx, class_name in enumerate(self.folder_classes):
                name_l = class_name.lower()
                if name_l not in label_to_idx:
                    raise ValueError(f"Unknown FER class folder: {class_name}")
                self.inner_to_unified[inner_idx] = label_to_idx[name_l]

        elif dataset_type == "raf":
            # folder names are '1'..'7' (Kaggle-style basic expressions)
            # Map to standard 0..6 (Angry..Neutral)
            # Kaggle RAF basic mapping typically:
            # 1=Surprise, 2=Fear, 3=Disgust, 4=Happy, 5=Sad, 6=Anger, 7=Neutral
            # We map that to our EMOTION_CLASSES indices.
            num_to_emotion = {
                "1": "surprise",
                "2": "fear",
                "3": "disgust",
                "4": "happy",
                "5": "sad",
                "6": "angry",
                "7": "neutral",
            }
            for inner_idx, folder_name in enumerate(self.folder_classes):
                if folder_name not in num_to_emotion:
                    raise ValueError(f"Unknown RAF class folder: {folder_name}")
                emo_name = num_to_emotion[folder_name]
                self.inner_to_unified[inner_idx] = label_to_idx[emo_name]
        else:
            raise ValueError("dataset_type must be 'fer' or 'raf'")

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, inner_label = self.inner[idx]
        unified_label = self.inner_to_unified[inner_label]
        return img, unified_label

# =========================
# 2. INSTANTIATE DATASETS
# =========================

fer_train_ds = RemappedFolderDataset(fer_train_dir, dataset_type="fer", transform=common_transform)
fer_test_ds  = RemappedFolderDataset(fer_test_dir,  dataset_type="fer", transform=common_transform)

raf_train_ds = RemappedFolderDataset(raf_train_dir, dataset_type="raf", transform=common_transform)
raf_test_ds  = RemappedFolderDataset(raf_test_dir,  dataset_type="raf", transform=common_transform)

print("FER folders:", fer_train_ds.folder_classes)
print("RAF folders:", raf_train_ds.folder_classes)
print("FER train size:", len(fer_train_ds), "FER test size:", len(fer_test_ds))
print("RAF train size:", len(raf_train_ds), "RAF test size:", len(raf_test_ds))

# Combine FER train + RAF train as a big training set
combined_train_ds = ConcatDataset([fer_train_ds, raf_train_ds])
print("Combined train size:", len(combined_train_ds))

# =========================
# 3. VISUALIZATION
# =========================

def count_labels(dataset, n_max=None):
    counts = [0] * len(EMOTION_CLASSES)
    length = len(dataset) if n_max is None else min(len(dataset), n_max)
    for i in range(length):
        _, y = dataset[i]
        counts[y] += 1
    return counts

fer_train_counts = count_labels(fer_train_ds)
raf_train_counts = count_labels(raf_train_ds)
combined_counts  = [f + r for f, r in zip(fer_train_counts, raf_train_counts)]

x = np.arange(len(EMOTION_CLASSES))
plt.figure(figsize=(9, 4))
plt.bar(x - 0.25, fer_train_counts, width=0.25, label="FER2013 train")
plt.bar(x,         raf_train_counts, width=0.25, label="RAF-DB train")
plt.bar(x + 0.25,  combined_counts,  width=0.25, label="Combined train")
plt.xticks(x, EMOTION_CLASSES, rotation=45)
plt.ylabel("Number of images")
plt.title("Class distribution: FER2013 vs RAF-DB vs Combined")
plt.legend()
plt.tight_layout()
plt.show()

def show_combined_examples(fer_ds, raf_ds, n=16):
    from math import ceil
    tagged_ds = [("FER2013", fer_ds), ("RAF-DB", raf_ds)]
    all_imgs, all_labels, all_sources = [], [], []

    per_source = n // 2
    for src_name, ds in tagged_ds:
        idxs = torch.randperm(len(ds))[:per_source]
        for idx in idxs:
            img, label = ds[idx]
            all_imgs.append(img)
            all_labels.append(label)
            all_sources.append(src_name)

    n_samples = len(all_imgs)
    n_cols = 4
    n_rows = int(ceil(n_samples / n_cols))

    plt.figure(figsize=(10, 2 * n_rows))
    for i in range(n_samples):
        img = all_imgs[i].numpy().squeeze()
        label = all_labels[i]
        src   = all_sources[i]

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{EMOTION_CLASSES[label]}\n({src})", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

show_combined_examples(fer_train_ds, raf_train_ds, n=16)

# =========================
# 4. SAVE COMBINED INDEX
# =========================

def export_index(fer_ds, raf_ds, out_csv_path):
    rows = []

    # FER part
    for rel_path, inner_label in fer_ds.inner.samples:
        unified_label = fer_ds.inner_to_unified[inner_label]
        abs_path = os.path.abspath(os.path.join(fer_train_dir, rel_path))
        rows.append({"source": "FER2013", "path": abs_path, "label": unified_label})

    # RAF part
    for rel_path, inner_label in raf_ds.inner.samples:
        unified_label = raf_ds.inner_to_unified[inner_label]
        abs_path = os.path.abspath(os.path.join(raf_train_dir, rel_path))
        rows.append({"source": "RAF-DB", "path": abs_path, "label": unified_label})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv_path, index=False)
    print(f"Saved combined index to: {out_csv_path}")
    print(df_out.head())

export_index(fer_train_ds, raf_train_ds, combined_index_csv)
