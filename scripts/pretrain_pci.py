"""
scripts/pretrain_pci.py
-----------------------
Pre-train the PCI head on RDD2022 road damage images.

Strategy
--------
Stage 1 (backbone + defect/vVCI heads FROZEN):
    Only the PCI head is trained. The backbone has already learned good
    pavement features from ImageNet — we just need to attach a PCI
    regression head on top.

Stage 2 (top backbone blocks + PCI head):
    Fine-tune the top 2 backbone blocks together with the PCI head, with
    a reduced backbone LR. Defect and vVCI heads remain frozen to avoid
    overwriting any pre-existing Uganda training.

The resulting checkpoint can be passed to train_model.py via
--pci-pretrain to initialise the PCI head before joint training.

Usage
-----
# 1. Download RDD2022 first
python scripts/download_rdd2022.py --output data/rdd2022

# 2. Pre-train PCI head (start from ImageNet backbone)
python scripts/pretrain_pci.py --data data/rdd2022

# 3. (Optional) Start from an existing Uganda checkpoint
python scripts/pretrain_pci.py --data data/rdd2022 \
    --base-checkpoint outputs/checkpoints/best.pt

# 4. Train the full model with the pre-trained PCI head
python scripts/train_model.py \
    --dataset outputs/dataset.csv \
    --pci-pretrain outputs/pci_pretrain/pci_head.pt \
    --device cuda --epochs 40 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.model import PavementVCIModel
from src.data.rdd2022_dataset import make_rdd2022_loader


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, device, is_train: bool) -> tuple[float, float]:
    model.train(is_train)
    total_loss = 0.0
    total_mae  = 0.0
    n          = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            images   = batch["image"].to(device)
            true_pci = batch["pci"].to(device)

            out      = model(images)
            pred_pci = out["pci"].squeeze(1)

            loss = F.huber_loss(pred_pci, true_pci, delta=5.0)
            mae  = (pred_pci - true_pci).abs().mean().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            bs          = images.size(0)
            total_loss += loss.item() * bs
            total_mae  += mae * bs
            n          += bs

    return total_loss / n, total_mae / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-train PCI head on RDD2022")
    parser.add_argument("--data",             required=True,
                        help="Path to RDD2022 root directory")
    parser.add_argument("--countries",        nargs="+", default=["Japan"],
                        help="Country subsets to use (default: Japan)")
    parser.add_argument("--base-checkpoint",  default=None,
                        help="Optional existing checkpoint to load backbone from")
    parser.add_argument("--output-dir",       default="outputs/pci_pretrain")
    parser.add_argument("--backbone",         default="efficientnet_b3")
    parser.add_argument("--epochs-stage1",    type=int, default=10,
                        help="Epochs with backbone frozen (heads only)")
    parser.add_argument("--epochs-stage2",    type=int, default=10,
                        help="Epochs with top backbone blocks unfrozen")
    parser.add_argument("--unfreeze-blocks",  type=int, default=2)
    parser.add_argument("--batch-size",       type=int, default=32)
    parser.add_argument("--num-workers",      type=int, default=4)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.05)
    parser.add_argument("--device",           default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")
    print(f"Data   : {args.data}  countries={args.countries}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = PavementVCIModel(backbone=args.backbone, pretrained=True)

    if args.base_checkpoint:
        ckpt = torch.load(args.base_checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded base checkpoint ({len(missing)} missing keys)")

    model.to(device)

    # ── Data ────────────────────────────────────────────────────────────────
    print("Building RDD2022 dataloader …")
    train_loader = make_rdd2022_loader(
        args.data,
        countries   = args.countries,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        augment     = True,
    )
    print(f"  {len(train_loader.dataset):,} images")

    # ── Stage 1: freeze everything except PCI head ──────────────────────────
    print(f"\n--- Stage 1: PCI head only ({args.epochs_stage1} epochs) ---")
    for p in model.backbone.parameters():   p.requires_grad_(False)
    for p in model.defect_head.parameters(): p.requires_grad_(False)
    for p in model.vvci_head.parameters():  p.requires_grad_(False)
    for p in model.pci_head.parameters():   p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_stage1, eta_min=1e-6)

    best_mae   = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs_stage1):
        train_loss, train_mae = _run_epoch(model, train_loader, optimizer, device, True)
        scheduler.step()
        print(f"  Epoch {epoch+1:2d}/{args.epochs_stage1}  "
              f"loss={train_loss:.4f}  MAE={train_mae:.2f}")
        if train_mae < best_mae:
            best_mae   = train_mae
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "mae": train_mae, "stage": 1},
                       out_dir / "pci_head_stage1.pt")

    # ── Stage 2: unfreeze top blocks ─────────────────────────────────────────
    print(f"\n--- Stage 2: top {args.unfreeze_blocks} backbone blocks "
          f"+ PCI head ({args.epochs_stage2} epochs) ---")
    model.unfreeze_top_blocks(args.unfreeze_blocks)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    pci_params      = list(model.pci_head.parameters())
    trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_factor},
        {"params": pci_params,      "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_stage2, eta_min=1e-7)

    for epoch in range(args.epochs_stage2):
        train_loss, train_mae = _run_epoch(model, train_loader, optimizer, device, True)
        scheduler.step()
        print(f"  Epoch {epoch+1:2d}/{args.epochs_stage2}  "
              f"loss={train_loss:.4f}  MAE={train_mae:.2f}")
        if train_mae < best_mae:
            best_mae   = train_mae
            best_epoch = args.epochs_stage1 + epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch,
                        "mae": train_mae, "stage": 2},
                       out_dir / "pci_head_stage2.pt")

    # ── Save best ────────────────────────────────────────────────────────────
    # Copy whichever stage gave the best MAE to the canonical output
    best_src = (out_dir / "pci_head_stage2.pt"
                if (out_dir / "pci_head_stage2.pt").exists()
                else out_dir / "pci_head_stage1.pt")
    import shutil
    shutil.copy(best_src, out_dir / "pci_head.pt")

    print(f"\nPre-training complete.")
    print(f"  Best MAE : {best_mae:.2f}  (epoch {best_epoch})")
    print(f"  Saved    : {out_dir}/pci_head.pt")
    print(f"\nNext step:")
    print(f"  python scripts/train_model.py \\")
    print(f"    --dataset outputs/dataset.csv \\")
    print(f"    --pci-pretrain {out_dir}/pci_head.pt \\")
    print(f"    --device cuda --epochs 40 --batch-size 32")


if __name__ == "__main__":
    main()
