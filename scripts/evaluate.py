"""
scripts/evaluate.py
-------------------
Full evaluation of the best checkpoint on the test set.
Produces: metrics printout, scatter plot, confusion matrices per defect,
          and a predictions CSV.

Usage
-----
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --dataset    outputs/dataset.csv \
    --output-dir outputs/evaluation
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    import torch
    from src.models.model import PavementVCIModel
    from src.data.dataset import make_dataloaders, DEFECT_COLS
    from src.training.losses import compute_metrics
    from src.models.pci_formula import grade_to_pci

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("config", {})
    model = PavementVCIModel(
        backbone=cfg.get("backbone", "efficientnet_b3"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  (val MAE={ckpt['val_mae']:.2f})")

    # Data
    loaders = make_dataloaders(args.dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = loaders["test"]

    # Collect predictions
    all_pred_vvci, all_true_vvci = [], []
    all_pred_logits = [[] for _ in range(6)]
    all_true_grades = []
    all_paths       = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            out    = model(images)
            all_pred_vvci.append(out["vvci"].cpu())
            all_true_vvci.append(batch["vvci"])
            for i, lg in enumerate(out["defect_logits"]):
                all_pred_logits[i].append(lg.cpu())
            all_true_grades.append(batch["grades"])
            all_paths.extend(batch["image_path"])

    pred_vvci   = torch.cat(all_pred_vvci).squeeze(1)
    true_vvci   = torch.cat(all_true_vvci)
    pred_logits = [torch.cat(lg_list) for lg_list in all_pred_logits]
    true_grades = torch.cat(all_true_grades)

    metrics = compute_metrics(pred_vvci, true_vvci, pred_logits, true_grades)

    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  vVCI MAE    : {metrics['mae_vvci']:.2f}")
    print(f"  vVCI RMSE   : {metrics['rmse_vvci']:.2f}")

    # R²
    ss_res = ((pred_vvci - true_vvci)**2).sum().item()
    ss_tot = ((true_vvci - true_vvci.mean())**2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"  vVCI R²     : {r2:.4f}")
    print(f"  Defect acc  : {metrics['acc_defect_mean']:.3f}")
    for i, name in enumerate(DEFECT_COLS):
        print(f"    {name:30s}: {metrics[f'acc_defect_{i}']:.3f}")

    # ── PCI metrics ───────────────────────────────────────────────────────
    defect_names = ["all_cracking","wide_cracking","ravelling","bleeding","drainage_road","potholes"]
    true_g_np  = true_grades.numpy()    # (N, 6), 0-indexed
    pred_g_np  = np.stack([pred_logits[i].argmax(1).numpy() for i in range(6)], axis=1)

    # Convert 0-indexed grade arrays to PCI; grade 0-idx → 1-5
    true_pcis = np.array([grade_to_pci(dict(zip(defect_names, (true_g_np[j] + 1).tolist())))
                          for j in range(len(true_g_np))])
    pred_pcis = np.array([grade_to_pci(dict(zip(defect_names, (pred_g_np[j] + 1).tolist())))
                          for j in range(len(pred_g_np))])

    pci_mae  = float(np.abs(true_pcis - pred_pcis).mean())
    pci_rmse = float(np.sqrt(((true_pcis - pred_pcis)**2).mean()))
    pci_r2   = (1 - ((true_pcis - pred_pcis)**2).sum() /
                    max(((true_pcis - true_pcis.mean())**2).sum(), 1e-8))

    print(f"\n  --- Estimated PCI (formula-based from grades) ---")
    print(f"  PCI MAE  : {pci_mae:.2f}")
    print(f"  PCI RMSE : {pci_rmse:.2f}")
    print(f"  PCI R²   : {pci_r2:.4f}  (R² of formula-PCI vs predicted-grade-PCI)")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "image_path":  all_paths,
        "true_vvci":   true_vvci.numpy(),
        "pred_vvci":   pred_vvci.numpy(),
        "residual":    (pred_vvci - true_vvci).numpy(),
        "true_pci":    true_pcis,
        "pred_pci":    pred_pcis,
        "pci_residual": (pred_pcis - true_pcis),
    })
    for i, name in enumerate(defect_names):
        pred_df[f"true_{name}_grade"]  = true_grades[:, i].numpy() + 1
        pred_df[f"pred_{name}_grade"]  = pred_logits[i].argmax(1).numpy() + 1
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    DEFECT_LABELS = ["All cracking","Wide cracking","Ravelling","Bleeding","Drainage","Potholes"]
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.4)

    # Scatter: pred vs true vVCI
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.scatter(true_vvci.numpy(), pred_vvci.numpy(), alpha=0.4, s=20, color="#534AB7")
    lims = [min(true_vvci.min(), pred_vvci.min()).item()-2,
            max(true_vvci.max(), pred_vvci.max()).item()+2]
    ax0.plot(lims, lims, "r--", lw=1)
    ax0.set_xlabel("True vVCI"); ax0.set_ylabel("Predicted vVCI")
    ax0.set_title(f"Predicted vs True vVCI  (MAE={metrics['mae_vvci']:.2f}, R²={r2:.3f})")
    ax0.set_xlim(lims); ax0.set_ylim(lims)

    # Residual histogram
    ax1 = fig.add_subplot(gs[0, 2])
    resid = (pred_vvci - true_vvci).numpy()
    ax1.hist(resid, bins=20, color="#1D9E75", edgecolor="none")
    ax1.axvline(0, color="red", lw=1)
    ax1.set_xlabel("Residual"); ax1.set_title("Prediction residuals")

    # Residual by true vVCI bin
    ax2 = fig.add_subplot(gs[0, 3])
    bins = np.linspace(0, 100, 6)
    bin_labels = [f"{int(b)}-{int(bins[i+1])}" for i, b in enumerate(bins[:-1])]
    bin_idx = np.digitize(true_vvci.numpy(), bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins)-2)
    means = [resid[bin_idx==i].mean() if (bin_idx==i).any() else 0 for i in range(len(bins)-1)]
    ax2.bar(range(len(means)), means, color="#BA7517", edgecolor="none")
    ax2.set_xticks(range(len(means))); ax2.set_xticklabels(bin_labels, fontsize=7)
    ax2.axhline(0, color="red", lw=1); ax2.set_title("Residual by vVCI range")

    # PCI scatter: pred vs true (formula-derived from grade preds vs grade truth)
    ax_pci = fig.add_subplot(gs[0, 2])
    ax_pci.scatter(true_pcis, pred_pcis, alpha=0.4, s=20, color="#B04040")
    lims_pci = [max(0, min(true_pcis.min(), pred_pcis.min())-2),
                min(100, max(true_pcis.max(), pred_pcis.max())+2)]
    ax_pci.plot(lims_pci, lims_pci, "r--", lw=1)
    ax_pci.set_xlabel("True PCI (formula)"); ax_pci.set_ylabel("Pred PCI (formula)")
    ax_pci.set_title(f"PCI pred vs true\n(MAE={pci_mae:.2f}, R²={pci_r2:.3f})", fontsize=9)

    # Correlation scatter: vVCI vs PCI (shows alignment)
    ax_corr = fig.add_subplot(gs[0, 3])
    ax_corr.scatter(true_vvci.numpy(), true_pcis, alpha=0.4, s=15, color="#408080")
    ax_corr.set_xlabel("vVCI"); ax_corr.set_ylabel("PCI (formula)")
    ax_corr.set_title("vVCI vs PCI correlation\n(ground truth)", fontsize=9)

    # Confusion matrices for each defect
    for i in range(6):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r+1, c])
        true_g = true_grades[:, i].numpy()
        pred_g = pred_logits[i].argmax(1).numpy()
        cm = np.zeros((5, 5), dtype=int)
        for t, p in zip(true_g, pred_g):
            cm[t, p] += 1
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels([1,2,3,4,5], fontsize=8)
        ax.set_yticklabels([1,2,3,4,5], fontsize=8)
        for ti in range(5):
            for tj in range(5):
                if cm[ti,tj] > 0:
                    ax.text(tj, ti, cm[ti,tj], ha="center", va="center", fontsize=7,
                            color="white" if cm[ti,tj] > cm.max()*0.5 else "black")
        acc = metrics[f"acc_defect_{i}"]
        ax.set_title(f"{DEFECT_LABELS[i]}\nacc={acc:.3f}", fontsize=8)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)

    fig.savefig(out_dir / "evaluation.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_dir}/evaluation.png")
    print(f"Saved: {out_dir}/predictions.csv")

if __name__ == "__main__":
    main()
