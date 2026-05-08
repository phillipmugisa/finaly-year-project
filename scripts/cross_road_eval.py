"""
scripts/cross_road_eval.py
--------------------------
Generalisation test: train on one set of roads, evaluate on a held-out road.
Run this after normal training to check cross-road generalisation.

Usage
-----
python scripts/cross_road_eval.py \
    --checkpoint outputs/checkpoints/best.pt \
    --dataset    outputs/dataset.csv \
    --hold-out-road A001   # road name to hold out as test
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--dataset",        required=True)
    parser.add_argument("--hold-out-road",  required=True, help="Road name to evaluate on")
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--device",         default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from src.models.model import PavementVCIModel
    from src.data.dataset import PavementDataset, get_val_transforms, DEFECT_COLS
    from src.training.losses import compute_metrics

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build a custom dataset filtered to hold-out road
    df = pd.read_csv(args.dataset)
    held_df = df[df["road_name"].str.upper() == args.hold_out_road.upper()].copy()
    if len(held_df) == 0:
        sys.exit(f"Road '{args.hold_out_road}' not found in dataset. "
                 f"Available: {df['road_name'].unique().tolist()}")
    # Temporarily mark all as 'test' split for evaluation
    held_df["split"] = "test"
    tmp_csv = Path("outputs/tmp_holdout.csv")
    held_df.to_csv(tmp_csv, index=False)
    print(f"Hold-out road: {args.hold_out_road}  ({len(held_df)} images)")

    ckpt  = torch.load(args.checkpoint, map_location=device)
    cfg   = ckpt.get("config", {})
    model = PavementVCIModel(backbone=cfg.get("backbone","efficientnet_b3"), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds     = PavementDataset(tmp_csv, split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_pred_vvci, all_true_vvci = [], []
    all_pred_logits = [[] for _ in range(6)]
    all_true_grades = []

    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            all_pred_vvci.append(out["vvci"].cpu())
            all_true_vvci.append(batch["vvci"])
            for i, lg in enumerate(out["defect_logits"]):
                all_pred_logits[i].append(lg.cpu())
            all_true_grades.append(batch["grades"])

    pred_vvci   = torch.cat(all_pred_vvci).squeeze(1)
    true_vvci   = torch.cat(all_true_vvci)
    pred_logits = [torch.cat(lg_list) for lg_list in all_pred_logits]
    true_grades = torch.cat(all_true_grades)

    metrics = compute_metrics(pred_vvci, true_vvci, pred_logits, true_grades)
    print(f"\n{'='*45}")
    print(f"CROSS-ROAD GENERALISATION: {args.hold_out_road}")
    print(f"{'='*45}")
    print(f"  vVCI MAE  : {metrics['mae_vvci']:.2f}")
    print(f"  vVCI RMSE : {metrics['rmse_vvci']:.2f}")
    print(f"  Defect acc: {metrics['acc_defect_mean']:.3f}")
    tmp_csv.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
