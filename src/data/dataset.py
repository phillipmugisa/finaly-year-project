"""
dataset.py
----------
PyTorch Dataset for the VCI estimator.

Each sample contains:
    image   : (3, H, W) float tensor, normalised to ImageNet stats
    grades  : (6,) long tensor  — one grade per visible defect (0-indexed: grade-1)
    vvci    : scalar float tensor — normalised visual VCI [0, 100]

The dataset supports:
- Multi-image segments: at training time one image is sampled randomly from the
  segment; at val/test time the first image is used (deterministic).
- Standard ImageNet augmentation pipeline (configurable).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Defect ordering (must match model head order)
# ---------------------------------------------------------------------------

DEFECT_COLS = [
    "all_cracking_grade",
    "wide_cracking_grade",
    "ravelling_grade",
    "bleeding_grade",
    "drainage_road_grade",
    "pothole_grade",
]
N_DEFECTS = len(DEFECT_COLS)   # 6
N_GRADES  = 5                  # 1-5


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PavementDataset(Dataset):
    """
    Parameters
    ----------
    csv_path  : path to the dataset.csv produced by build_dataset.py
    split     : 'train', 'val', or 'test'
    transform : torchvision transform (default: appropriate for split)
    image_size: square crop size fed to the model
    """

    def __init__(
        self,
        csv_path:   str | Path,
        split:      str = "train",
        transform=None,
        image_size: int = 224,
    ):
        self.split = split
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {csv_path}")

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Group by segment key for multi-image access
        self._segment_groups = self.df.groupby(
            ["road_code", "segment_start"]
        ).groups  # {key: [row indices]}

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load image
        img = Image.open(row["image_path"]).convert("RGB")

        # Crop away the Mobicap blue header before passing to model.
        # Header is ~8% of image height (200px on 2420px images, ~80px on 952px).
        w, h = img.size
        header_px = h // 12
        img = img.crop((0, header_px, w, h))

        img_tensor = self.transform(img)

        # Defect grades: convert 1-5 → 0-4 (class indices for cross-entropy)
        grades = torch.tensor(
            [int(row[col]) - 1 for col in DEFECT_COLS],
            dtype=torch.long,
        )

        vvci = torch.tensor(float(row["vvci"]), dtype=torch.float32)

        return {
            "image":      img_tensor,
            "grades":     grades,           # (6,) long, 0-indexed
            "vvci":       vvci,             # scalar float
            "vci":        torch.tensor(float(row["vci"]),  dtype=torch.float32),
            "image_path": str(row["image_path"]),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    csv_path:    str | Path,
    image_size:  int = 224,
    batch_size:  int = 32,
    num_workers: int = 4,
    pin_memory:  bool = True,
) -> dict[str, DataLoader]:
    """
    Returns a dict with keys 'train', 'val', 'test'.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = PavementDataset(csv_path, split=split, image_size=image_size)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )
        print(f"  {split:5s}: {len(ds)} images")
    return loaders


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    csv = sys.argv[1]
    loaders = make_dataloaders(csv, batch_size=4, num_workers=0)
    batch = next(iter(loaders["train"]))
    print("image shape :", batch["image"].shape)
    print("grades shape:", batch["grades"].shape)
    print("vvci        :", batch["vvci"])
