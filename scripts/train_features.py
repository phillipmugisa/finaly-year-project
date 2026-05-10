"""
scripts/train_features.py
--------------------------
Train the three task heads (Defect + vVCI + PCI) on pre-extracted
backbone feature vectors instead of raw images.

This is the Colab-friendly training path:
  - No raw images needed — just features.npz (~500 MB) + dataset.csv
  - All heavy backbone computation already done locally
  - Trains only the lightweight heads → fast on any GPU or even CPU

Usage (on Colab or any machine with features.npz)
-------------------------------------------------
python scripts/train_features.py \
    --features  outputs/features.npz \
    --dataset   outputs/dataset.csv \
    --device    cuda \
    --epochs    40 \
    --batch-size 256 \
    --pci-pretrain outputs/pci_pretrain/pci_head.pt

Notes
-----
- batch-size can be large (256+) because there is no backbone forward pass
- No Stage 2 (backbone fine-tuning) — features are fixed
- All other hyperparameters mirror train_model.py defaults
- Checkpoint format is identical to train_model.py → evaluate.py works as-is
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.model import HeadsModel
from src.training.losses import PavementLoss, compute_metrics
from src.models.pci_formula import grade_to_pci

_DEFECT_COLS = [
    "all_cracking_grade", "wide_cracking_grade", "ravelling_grade",
    "bleeding_grade", "drainage_road_grade", "pothole_grade",
]
_DEFECT_NAMES = ["all_cracking","wide_cracking","ravelling","bleeding","drainage_road","potholes"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    """
    Loads pre-extracted 1536-d backbone features + labels from dataset.csv.
    Much faster than loading images — fits entirely in RAM (~500 MB for 80k rows).
    """

    def __init__(self, features: np.ndarray, df: pd.DataFrame):
        """
        features : (N, 1536) float16 array aligned with df rows
        df       : subset of dataset.csv rows (already filtered + split)
        """
        self.features = torch.from_numpy(features.astype(np.float32))  # (N, 1536)
        self.vvci     = torch.tensor(df["vvci"].values,       dtype=torch.float32)
        self.grades   = torch.tensor(
            df[_DEFECT_COLS].values - 1, dtype=torch.long   # 1-indexed → 0-indexed
        )

    def __len__(self):
        return len(self.vvci)

    def __getitem__(self, idx):
        return {
            "feature": self.features[idx],   # (1536,)
            "vvci":    self.vvci[idx],        # scalar
            "grades":  self.grades[idx],      # (6,)
        }


def make_feature_loaders(
    features_path: str,
    dataset_csv:   str,
    batch_size:    int = 256,
    num_workers:   int = 2,
    seed:          int = 42,
) -> dict[str, DataLoader]:

    print(f"Loading features: {features_path} …")
    npz   = np.load(features_path, allow_pickle=True)
    feats = npz["features"]         # (N, 1536) float16
    paths = npz["image_paths"]      # (N,) str
    print(f"  {len(paths):,} feature vectors  shape={feats.shape}")

    path_to_idx = {p: i for i, p in enumerate(paths)}

    print(f"Loading dataset:  {dataset_csv} …")
    df = pd.read_csv(dataset_csv)
    df = df.dropna(subset=_DEFECT_COLS + ["vvci"])

    loaders = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)

        # Keep only rows whose image_path appears in the features file
        mask  = sub["image_path"].isin(path_to_idx)
        sub   = sub[mask].reset_index(drop=True)
        idxs  = [path_to_idx[p] for p in sub["image_path"]]
        fvecs = feats[idxs]

        ds = FeatureDataset(fvecs, sub)
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = (split == "train"),
        )
        print(f"  {split:5s}: {len(ds):,} samples")

    return loaders


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train(is_train)
    total_loss = 0.0
    all_pred_vvci, all_true_vvci = [], []
    all_pred_logits = [[] for _ in range(6)]
    all_true_grades = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            feat        = batch["feature"].to(device)
            true_vvci   = batch["vvci"].to(device)
            true_grades = batch["grades"].to(device)

            grades_np = true_grades.cpu().numpy() + 1
            true_pci  = torch.tensor(
                [grade_to_pci(dict(zip(_DEFECT_NAMES, g.tolist()))) for g in grades_np],
                dtype=torch.float32, device=device,
            )

            out         = model(feat)
            pred_vvci   = out["vvci"]
            pred_logits = out["defect_logits"]
            pred_pci    = out["pci"]

            loss_dict = criterion(pred_vvci, true_vvci, pred_logits, true_grades,
                                  pred_pci=pred_pci, true_pci=true_pci)
            loss = loss_dict["total"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            bs = feat.size(0)
            total_loss += loss.item() * bs
            all_pred_vvci.append(pred_vvci.detach().cpu())
            all_true_vvci.append(true_vvci.detach().cpu())
            for i, lg in enumerate(pred_logits):
                all_pred_logits[i].append(lg.detach().cpu())
            all_true_grades.append(true_grades.detach().cpu())

    n = sum(len(b) for b in all_true_vvci)
    all_pred_vvci   = torch.cat(all_pred_vvci)
    all_true_vvci   = torch.cat(all_true_vvci)
    all_pred_logits = [torch.cat(lg) for lg in all_pred_logits]
    all_true_grades = torch.cat(all_true_grades)

    metrics = compute_metrics(all_pred_vvci, all_true_vvci, all_pred_logits, all_true_grades)
    metrics["loss"] = total_loss / n
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train heads on pre-extracted features")
    parser.add_argument("--features",      required=True,
                        help="Path to features.npz from extract_features.py")
    parser.add_argument("--dataset",       required=True,
                        help="Path to dataset.csv")
    parser.add_argument("--output-dir",    default="outputs")
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--num-workers",   type=int,   default=2)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--lambda-vvci",   type=float, default=1.0)
    parser.add_argument("--lambda-defect", type=float, default=0.5)
    parser.add_argument("--lambda-pci",    type=float, default=0.5)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--pci-pretrain",  default=None,
                        help="Path to pre-trained PCI head weights (pci_head.pt)")
    parser.add_argument("--device",        default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = make_feature_loaders(
        args.features, args.dataset, args.batch_size, args.num_workers, args.seed
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HeadsModel(dropout=args.dropout).to(device)

    if args.pci_pretrain:
        ckpt  = torch.load(args.pci_pretrain, map_location="cpu")
        state = ckpt.get("model", ckpt)
        pci_state = {k.replace("pci_head.", ""): v
                     for k, v in state.items() if "pci_head" in k}
        model.pci_head.load_state_dict(pci_state, strict=False)
        print(f"Loaded pre-trained PCI head from {args.pci_pretrain}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Compute class weights from training split
    train_df = pd.read_csv(args.dataset)
    train_df = train_df[train_df["split"] == "train"]
    from src.training.losses import compute_class_weights
    cw = compute_class_weights(args.dataset, _DEFECT_COLS, n_grades=5, split="train")
    criterion = PavementLoss(
        lambda_vvci   = args.lambda_vvci,
        lambda_defect = args.lambda_defect,
        lambda_pci    = args.lambda_pci,
        class_weights = cw,
        huber_delta   = 5.0,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mae = float("inf")
    best_epoch   = 0
    history      = []
    best_ckpt    = ckpt_dir / "best.pt"

    print(f"\nTraining {args.epochs} epochs on {device} …")
    print(f"  Batch size: {args.batch_size}  (no backbone forward pass — fast!)")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_m = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_m   = run_epoch(model, loaders["val"],   criterion, None,      device, False)
        scheduler.step()

        elapsed  = time.time() - t0
        val_mae  = val_m["mae_vvci"]
        row      = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}":   v for k, v in val_m.items()})
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.0f}s) | "
            f"train loss {train_m['loss']:.4f} | "
            f"val MAE {val_mae:.2f}  RMSE {val_m['rmse_vvci']:.2f}  "
            f"defect acc {val_m['acc_defect_mean']:.3f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_mae":   val_mae,
            "feature_based": True,
            "config": {"feat_dim": 1536, "n_defects": 6},
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            torch.save(ckpt, best_ckpt)
            print(f"  ✓ Best checkpoint (val MAE = {best_val_mae:.2f})")

    # ── Save history ──────────────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(output_dir / "metrics_features.csv", index=False)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating on test set …")
    ckpt_data = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model"])
    test_m = run_epoch(model, loaders["test"], criterion, None, device, False)
    print(f"Test MAE  : {test_m['mae_vvci']:.2f}")
    print(f"Test RMSE : {test_m['rmse_vvci']:.2f}")
    print(f"Defect acc: {test_m['acc_defect_mean']:.3f}")
    print(f"\nBest checkpoint → {best_ckpt}")


if __name__ == "__main__":
    main()
