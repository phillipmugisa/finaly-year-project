"""
src/data/rdd2022_dataset.py
---------------------------
PyTorch Dataset for RDD2022 (Road Damage Detection 2022) images.

RDD2022 provides bounding box annotations for 4 road damage categories:
  D00 — Longitudinal crack
  D10 — Transverse crack
  D20 — Alligator (crocodile) crack
  D40 — Pothole

These are mapped to PCI distress categories and converted to pseudo-PCI
labels using linearised ASTM D6433 deduct value curves.

Pseudo-PCI derivation
---------------------
1. For each damage category, compute coverage = sum(bbox areas) / image area.
2. Look up a deduct value from the piecewise-linear table.
3. Run the ASTM CDV procedure → PCI = 100 − CDV.
4. Images with no annotations receive PCI = 100 (undamaged).

Supported country subsets (each stored under <data_root>/<country>/):
  Japan, India, Norway, United_States, Czech, China_MotorBike

Dataset folder structure expected:
  <data_root>/
    <country>/
      train/
        images/         ← .jpg/.png files
        annotations/xml/  ← PASCAL VOC .xml files (same stem as image)
      test/
        images/         ← no annotations, skipped for training
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── ImageNet normalisation (same as the main model) ─────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Pseudo-PCI lookup tables
# Piecewise-linear: coverage_pct → deduct value
# Calibrated to ASTM D6433 curves for each distress type.
# ---------------------------------------------------------------------------

# Each entry: (coverage_pct, deduct_value)
_DEDUCT_CURVES: dict[str, list[tuple[float, float]]] = {
    # D20: Alligator cracking — highest impact per unit area
    "D20": [(0, 0), (1, 22), (5, 40), (10, 52), (25, 65), (50, 76), (100, 84)],
    # D00: Longitudinal cracks
    "D00": [(0, 0), (1,  5), (5, 12), (10, 18), (25, 26), (50, 33), (100, 38)],
    # D10: Transverse cracks
    "D10": [(0, 0), (1,  5), (5, 12), (10, 18), (25, 26), (50, 33), (100, 38)],
    # D40: Potholes — coverage used as proxy for count density
    "D40": [(0, 0), (1, 20), (5, 42), (10, 58), (25, 70), (50, 80), (100, 85)],
}

# Ignore non-structural categories
_IGNORED_CATEGORIES = {"D43", "D44"}


def _interp_deduct(curve: list[tuple[float, float]], coverage: float) -> float:
    """Piecewise-linear interpolation on the deduct curve."""
    if coverage <= 0:
        return 0.0
    for i in range(len(curve) - 1):
        x0, y0 = curve[i]
        x1, y1 = curve[i + 1]
        if x0 <= coverage <= x1:
            t = (coverage - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return float(curve[-1][1])


def _cdv_from_deducts(deducts: list[float]) -> float:
    """ASTM D6433 CDV iterative procedure (same as pci_formula.py)."""
    active  = sorted([v for v in deducts if v > 2.0], reverse=True)
    passive = sum(min(v, 2.0) for v in deducts if v <= 2.0)

    if not active:
        return 0.0

    best_cdv = 0.0
    working  = list(active)

    while True:
        q   = len(working)
        tdv = sum(working) + passive
        cdv = tdv / (1.0 + 0.003 * (q - 1) * tdv) if q > 1 else tdv
        best_cdv = max(best_cdv, cdv)
        if q == 1:
            break
        passive    += 2.0
        working[-1] = None
        working     = [v for v in working[:-1] if v > 2.0]
        if not working:
            break

    return best_cdv


def parse_annotation(xml_path: Path, img_w: int, img_h: int) -> float:
    """
    Parse a PASCAL VOC annotation XML and return pseudo-PCI [0, 100].
    Returns 100.0 if the file does not exist or has no valid objects.
    """
    if not xml_path.exists():
        return 100.0

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return 100.0

    root     = tree.getroot()
    img_area = max(img_w * img_h, 1)

    # Accumulate covered area per damage category
    coverage: dict[str, float] = {k: 0.0 for k in _DEDUCT_CURVES}

    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().upper()
        if name in _IGNORED_CATEGORIES or name not in _DEDUCT_CURVES:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin", 0))
            ymin = float(bb.findtext("ymin", 0))
            xmax = float(bb.findtext("xmax", img_w))
            ymax = float(bb.findtext("ymax", img_h))
        except (ValueError, TypeError):
            continue
        w   = max(0.0, xmax - xmin)
        h   = max(0.0, ymax - ymin)
        coverage[name] += (w * h) / img_area * 100.0  # as percent

    # Compute deduct values and PCI
    deducts = [_interp_deduct(_DEDUCT_CURVES[cat], min(cov, 100.0))
               for cat, cov in coverage.items()]
    cdv = _cdv_from_deducts(deducts)
    return float(np.clip(100.0 - cdv, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RDD2022Dataset(Dataset):
    """
    Parameters
    ----------
    data_root : path to the root RDD2022 directory (contains country sub-dirs)
    countries : list of country names to include, e.g. ['Japan', 'India']
                Default: all available countries
    split     : 'train' only (test set has no annotations)
    augment   : apply random augmentation (recommended for training)
    """

    COUNTRIES = ["Japan", "India", "Norway", "United_States", "Czech", "China_MotorBike"]

    def __init__(
        self,
        data_root:  str | Path,
        countries:  Optional[list[str]] = None,
        split:      str = "train",
        augment:    bool = True,
    ):
        self.data_root = Path(data_root)
        self.transform = _AUGMENT_TRANSFORM if augment else _TRANSFORM
        self.samples: list[tuple[Path, Path | None]] = []  # (image_path, xml_path|None)

        target_countries = countries or self.COUNTRIES

        for country in target_countries:
            img_dir = self.data_root / country / split / "images"
            ann_dir = self.data_root / country / split / "annotations" / "xml"

            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                xml_path = ann_dir / (img_path.stem + ".xml")
                self.samples.append((img_path, xml_path if xml_path.exists() else None))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.data_root} for countries {target_countries}. "
                "Run scripts/download_rdd2022.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, xml_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        pci = parse_annotation(xml_path, w, h) if xml_path else 100.0

        return {
            "image":      self.transform(img),       # (3, 224, 224)
            "pci":        torch.tensor(pci, dtype=torch.float32),
            "image_path": str(img_path),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_rdd2022_loader(
    data_root:  str | Path,
    countries:  Optional[list[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    augment:    bool = True,
) -> torch.utils.data.DataLoader:
    ds = RDD2022Dataset(data_root, countries=countries, augment=augment)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/rdd2022"
    try:
        ds = RDD2022Dataset(root, countries=["Japan"])
        print(f"Dataset: {len(ds)} images")
        sample = ds[0]
        print(f"  image shape : {sample['image'].shape}")
        print(f"  pci label   : {sample['pci'].item():.1f}")
    except FileNotFoundError as e:
        print(e)
