"""
scripts/extract_features.py
----------------------------
Run the frozen EfficientNet-B3 backbone over every image in dataset.csv
and save the 1536-d feature vectors to disk.

The resulting features.npz (~500 MB for 80k images) can be uploaded to
Google Drive and used to train the three heads on Colab without needing
the raw images.

Usage
-----
# Full dataset (run after pipeline produces dataset.csv)
python scripts/extract_features.py \
    --dataset outputs/dataset.csv \
    --output  outputs/features.npz \
    --batch-size 32

# Partial dataset (2021-22 only, for a quick first Colab run)
python scripts/extract_features.py \
    --dataset outputs/dataset_2122.csv \
    --output  outputs/features_2122.npz \
    --batch-size 32

Output
------
outputs/features.npz containing:
  features    : float16  (N, 1536)  backbone feature vectors
  image_paths : str      (N,)       absolute paths (index into dataset.csv)

The caller merges features with dataset.csv on the image_path column.
float16 halves the file size (1536 × 2 bytes × 80k ≈ 245 MB) with
negligible accuracy loss since these are just intermediate activations.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Preprocessing — must match dataset.py and image_utils.py ─────────────────
_MOBICAP_HEADER_PX = 80   # blue text overlay at top of Mobicap images
_IMAGENET_MEAN     = [0.485, 0.456, 0.406]
_IMAGENET_STD      = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _is_mobicap(path: str) -> bool:
    """Heuristic: filenames without EXIF GPS are Mobicap (blue header present)."""
    p = Path(path).name.upper()
    # Mobicap filenames follow ROADCODE_LINK or TOWN-TOWN patterns
    return (p.startswith("A0") or p.startswith("B") or p.startswith("C0")
            or "-PAVE-" in p or "LINK" in p or "NJERU" in p
            or "BUSUNJU" in p or "HOIMA" in p or "JINJA" in p
            or "LUGAZI" in p or "LWAMATA" in p)


def load_image(path: str) -> torch.Tensor | None:
    """Load and preprocess a single image. Returns None on failure."""
    try:
        img = Image.open(path).convert("RGB")
        # Crop Mobicap blue header (top 80px)
        w, h = img.size
        if _is_mobicap(path) and h > _MOBICAP_HEADER_PX + 32:
            img = img.crop((0, _MOBICAP_HEADER_PX, w, h))
        return _TRANSFORM(img)
    except Exception:
        return None


# ── Simple image dataset ──────────────────────────────────────────────────────

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        tensor = load_image(self.paths[idx])
        if tensor is None:
            # Return zeros so we can flag this index later
            tensor = torch.zeros(3, 224, 224)
            valid  = torch.tensor(False)
        else:
            valid  = torch.tensor(True)
        return tensor, valid, idx


def collate(batch):
    imgs   = torch.stack([b[0] for b in batch])
    valids = torch.stack([b[1] for b in batch])
    idxs   = [b[2] for b in batch]
    return imgs, valids, idxs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract EfficientNet-B3 backbone features for all dataset images"
    )
    parser.add_argument("--dataset",    default="outputs/dataset.csv",
                        help="Path to dataset CSV (image_path column required)")
    parser.add_argument("--output",     default="outputs/features.npz",
                        help="Output .npz file path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers",type=int, default=4)
    parser.add_argument("--device",     default=None,
                        help="cpu | cuda (default: auto)")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip images already present in an existing output file")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    df      = pd.read_csv(args.dataset)
    paths   = df["image_path"].tolist()
    n_total = len(paths)
    print(f"  {n_total:,} images to process")

    # ── Resume: skip already-done images ─────────────────────────────────────
    done_set: set[str] = set()
    if args.resume and output_path.exists():
        existing = np.load(output_path, allow_pickle=True)
        done_set = set(existing["image_paths"].tolist())
        print(f"  Resuming: {len(done_set):,} already extracted, "
              f"{n_total - len(done_set):,} remaining")

    todo_paths   = [p for p in paths if p not in done_set]
    todo_indices = [i for i, p in enumerate(paths) if p not in done_set]

    if not todo_paths:
        print("All images already extracted. Nothing to do.")
        return

    # ── Load model (backbone only) ────────────────────────────────────────────
    import timm
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nDevice: {device}")
    print("Loading EfficientNet-B3 backbone (pretrained=True) …")

    backbone = timm.create_model(
        "efficientnet_b3",
        pretrained=True,
        num_classes=0,     # remove classifier head
        global_pool="avg", # GAP → 1536-d
    )
    backbone.eval()
    backbone.to(device)

    for p in backbone.parameters():
        p.requires_grad_(False)

    print(f"  Feature dim: {backbone.num_features}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    ds     = ImagePathDataset(todo_paths)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        collate_fn  = collate,
        pin_memory  = device.type == "cuda",
    )

    # ── Extract features ──────────────────────────────────────────────────────
    feat_dim  = backbone.num_features
    new_feats = np.zeros((len(todo_paths), feat_dim), dtype=np.float16)
    failed    = []

    print(f"\nExtracting features …")
    with torch.no_grad():
        for imgs, valids, idxs in tqdm(loader, unit="batch"):
            imgs = imgs.to(device)
            feats = backbone(imgs).cpu().numpy().astype(np.float16)

            for local_i, (global_i, valid) in enumerate(zip(idxs, valids)):
                if valid.item():
                    new_feats[local_i] = feats[local_i]
                else:
                    failed.append(todo_paths[local_i])

    if failed:
        print(f"\nWarning: {len(failed)} images failed to load — zeros stored:")
        for p in failed[:5]:
            print(f"  {p}")
        if len(failed) > 5:
            print(f"  … and {len(failed)-5} more")

    # ── Merge with any existing features and save ────────────────────────────
    if args.resume and output_path.exists():
        existing  = np.load(output_path, allow_pickle=True)
        all_paths = np.concatenate([existing["image_paths"], np.array(todo_paths)])
        all_feats = np.concatenate([existing["features"],    new_feats])
    else:
        all_paths = np.array(todo_paths)
        all_feats = new_feats

    np.savez_compressed(
        output_path,
        features    = all_feats,      # (N, 1536) float16
        image_paths = all_paths,      # (N,) str
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved → {output_path}")
    print(f"  Features shape : {all_feats.shape}")
    print(f"  File size      : {size_mb:.0f} MB")
    print(f"  Failed images  : {len(failed)}")
    print(f"\nNext step: upload {output_path} + {args.dataset} to Google Drive")


if __name__ == "__main__":
    main()
