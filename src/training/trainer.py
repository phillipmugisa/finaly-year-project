"""
trainer.py
----------
Two-stage training loop for the PavementVCIModel.

Stage 1 (epochs 0 … freeze_epochs - 1)
    Backbone frozen; only heads are trained with a standard LR.

Stage 2 (epoch freeze_epochs … total_epochs - 1)
    Top `unfreeze_top_blocks` backbone blocks are unfrozen.
    Backbone gets lr × backbone_lr_factor; heads keep the base lr.
    The scheduler restarts with the new parameter groups.

Checkpointing
-------------
Best checkpoint is saved by val_mae_vvci (lower = better).
A 'last.pt' checkpoint is always written at the end of each epoch.

Logging
-------
Per-epoch metrics are printed to stdout and saved to outputs/metrics.csv.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.model import PavementVCIModel
from .losses import PavementLoss, compute_metrics, compute_class_weights
from ..data.dataset import make_dataloaders, DEFECT_COLS
from ..models.pci_formula import grade_to_pci


# ---------------------------------------------------------------------------
# Helper: one epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model:      PavementVCIModel,
    loader,
    criterion:  PavementLoss,
    optimizer:  optim.Optimizer | None,
    device:     torch.device,
    is_train:   bool,
) -> dict[str, float]:
    model.train(is_train)
    total_loss = 0.0
    all_pred_vvci, all_true_vvci = [], []
    all_pred_logits = [[] for _ in range(model.defect_head.classifiers.__len__())]
    all_true_grades = []

    _DEFECT_NAMES = ["all_cracking","wide_cracking","ravelling",
                     "bleeding","drainage_road","potholes"]

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            images      = batch["image"].to(device)
            true_vvci   = batch["vvci"].to(device)
            true_grades = batch["grades"].to(device)   # (B, 6)  0-indexed

            # Compute formula-based PCI from ground-truth grades as supervision
            grades_np = true_grades.cpu().numpy() + 1   # 0-indexed → 1-5
            true_pci  = torch.tensor(
                [grade_to_pci(dict(zip(_DEFECT_NAMES, g.tolist()))) for g in grades_np],
                dtype=torch.float32, device=device,
            )

            out = model(images)
            pred_vvci   = out["vvci"]
            pred_logits = out["defect_logits"]
            pred_pci    = out["pci"]

            loss_dict = criterion(pred_vvci, true_vvci, pred_logits, true_grades,
                                  pred_pci=pred_pci, true_pci=true_pci)
            loss      = loss_dict["total"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_pred_vvci.append(pred_vvci.detach().cpu())
            all_true_vvci.append(true_vvci.detach().cpu())
            for i, lg in enumerate(pred_logits):
                all_pred_logits[i].append(lg.detach().cpu())
            all_true_grades.append(true_grades.detach().cpu())

    n = sum(len(b) for b in all_true_vvci)
    all_pred_vvci   = torch.cat(all_pred_vvci)
    all_true_vvci   = torch.cat(all_true_vvci)
    all_pred_logits = [torch.cat(lg_list) for lg_list in all_pred_logits]
    all_true_grades = torch.cat(all_true_grades)

    metrics = compute_metrics(all_pred_vvci, all_true_vvci, all_pred_logits, all_true_grades)
    metrics["loss"] = total_loss / n
    return metrics


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train(
    csv_path:              str | Path,
    output_dir:            str | Path  = "outputs",
    backbone:              str         = "efficientnet_b3",
    pretrained:            bool        = True,
    image_size:            int         = 224,
    batch_size:            int         = 32,
    num_workers:           int         = 4,
    epochs:                int         = 40,
    freeze_epochs:         int         = 10,
    unfreeze_top_blocks:   int         = 3,
    base_lr:               float       = 1e-3,
    weight_decay:          float       = 1e-4,
    backbone_lr_factor:    float       = 0.1,
    lambda_vvci:           float       = 1.0,
    lambda_defect:         float       = 0.5,
    lambda_pci:            float       = 0.5,
    huber_delta:           float       = 5.0,
    dropout:               float       = 0.3,
    vvci_hidden:           int         = 256,
    pci_hidden:            int         = 256,
    pci_pretrain:          str | None  = None,
    seed:                  int         = 42,
    device:                str | None  = None,
) -> dict:
    """
    Full two-stage training run.

    Returns
    -------
    dict with keys 'best_val_mae', 'best_epoch', 'checkpoint_path'
    """
    # ── Setup ────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Device: {dev}")

    output_dir  = Path(output_dir)
    ckpt_dir    = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    print("Building dataloaders …")
    loaders = make_dataloaders(csv_path, image_size, batch_size, num_workers)

    # ── Class weights for defect loss ─────────────────────────────────────
    print("Computing class weights …")
    cw = compute_class_weights(csv_path, DEFECT_COLS, n_grades=5, split="train")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"Building model: {backbone} …")
    model = PavementVCIModel(
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        vvci_hidden=vvci_hidden,
        pci_hidden=pci_hidden,
    ).to(dev)

    # Load pre-trained PCI head weights if provided
    if pci_pretrain is not None:
        pci_ckpt = torch.load(pci_pretrain, map_location=dev)
        state    = pci_ckpt.get("model", pci_ckpt)
        # Load only pci_head weights — leave everything else untouched
        pci_state = {k: v for k, v in state.items() if k.startswith("pci_head.")}
        missing, _ = model.load_state_dict(pci_state, strict=False)
        print(f"  Loaded pre-trained PCI head from {pci_pretrain} "
              f"({len(pci_state)} keys, {len(missing)} missing)")

    criterion = PavementLoss(
        lambda_vvci=lambda_vvci,
        lambda_defect=lambda_defect,
        lambda_pci=lambda_pci,
        class_weights=cw,
        huber_delta=huber_delta,
    ).to(dev)

    # ── Stage 1 optimizer (heads only) ───────────────────────────────────
    model.freeze_backbone()
    print(f"Stage 1 — trainable params: {model.n_trainable_params:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_mae  = float("inf")
    best_epoch    = 0
    best_ckpt     = ckpt_dir / "best.pt"
    history       = []

    for epoch in range(epochs):
        t0 = time.time()

        # ── Stage transition ────────────────────────────────────────────
        if epoch == freeze_epochs:
            model.unfreeze_top_blocks(unfreeze_top_blocks)
            print(f"\n--- Stage 2 start (epoch {epoch}) ---")
            print(f"  Trainable params: {model.n_trainable_params:,}")
            param_groups = model.get_parameter_groups(backbone_lr_factor, base_lr)
            optimizer    = optim.AdamW(param_groups, weight_decay=weight_decay)
            scheduler    = CosineAnnealingLR(optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6)

        # ── Train ────────────────────────────────────────────────────────
        train_metrics = _run_epoch(model, loaders["train"], criterion, optimizer, dev, is_train=True)
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────
        val_metrics = _run_epoch(model, loaders["val"], criterion, None, dev, is_train=False)

        elapsed = time.time() - t0
        val_mae = val_metrics["mae_vvci"]

        # ── Logging ──────────────────────────────────────────────────────
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}":   v for k, v in val_metrics.items()})
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{epochs} ({elapsed:.0f}s) | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val MAE {val_mae:.2f}  RMSE {val_metrics['rmse_vvci']:.2f}  "
            f"defect acc {val_metrics['acc_defect_mean']:.3f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────
        ckpt = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "val_mae":    val_mae,
            "config": {
                "backbone": backbone, "image_size": image_size,
                "n_defects": model.defect_head.classifiers.__len__(),
            },
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            torch.save(ckpt, best_ckpt)
            print(f"  ✓ Best checkpoint saved (val MAE = {best_val_mae:.2f})")

    # ── Save history ──────────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.2f} at epoch {best_epoch}")
    print(f"Best checkpoint: {best_ckpt}")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\nEvaluating on test set …")
    ckpt_data = torch.load(best_ckpt, map_location=dev)
    model.load_state_dict(ckpt_data["model"])
    test_metrics = _run_epoch(model, loaders["test"], criterion, None, dev, is_train=False)
    print(f"Test MAE : {test_metrics['mae_vvci']:.2f}")
    print(f"Test RMSE: {test_metrics['rmse_vvci']:.2f}")
    print(f"Test defect acc (mean): {test_metrics['acc_defect_mean']:.3f}")

    return {
        "best_val_mae":    best_val_mae,
        "best_epoch":      best_epoch,
        "checkpoint_path": str(best_ckpt),
        "test_metrics":    test_metrics,
    }
