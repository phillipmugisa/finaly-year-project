"""
scripts/analyse_contributions.py
---------------------------------
After training: analyse which defects the model learns best,
and how each predicted defect grade correlates with vVCI error.

Usage
-----
python scripts/analyse_contributions.py \
    --predictions outputs/evaluation/predictions.csv
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

DEFECT_NAMES = ["all_cracking","wide_cracking","ravelling","bleeding","drainage","potholes"]
DEFECT_WEIGHTS = {"all_cracking": 7.0, "wide_cracking": 10.0, "ravelling": 5.0,
                  "bleeding": 2.5, "drainage": 2.5, "potholes": 7.5}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir",  default="outputs/evaluation")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resid = df["residual"].values

    print(f"\n{'='*55}")
    print("DEFECT CONTRIBUTION ANALYSIS")
    print(f"{'='*55}")

    # Per-defect: grade accuracy and within-1 accuracy
    print(f"\n{'Defect':25s} {'Weight':>7} {'Exact':>7} {'±1 acc':>7} {'MAE':>7}")
    print("-" * 55)
    rows = []
    for name in DEFECT_NAMES:
        true_col = f"true_{name}_grade"
        pred_col = f"pred_{name}_grade"
        if true_col not in df.columns:
            continue
        true_g = df[true_col].values
        pred_g = df[pred_col].values
        exact  = (true_g == pred_g).mean()
        within1= (np.abs(true_g - pred_g) <= 1).mean()
        mae    = mean_absolute_error(true_g, pred_g)
        w      = DEFECT_WEIGHTS.get(name, 0)
        print(f"{name:25s} {w:>7.1f} {exact:>7.3f} {within1:>7.3f} {mae:>7.3f}")
        rows.append({"defect": name, "weight": w, "exact_acc": exact, "within1_acc": within1, "grade_mae": mae})

    # Correlation of grade error with vVCI residual
    print(f"\n{'Defect':25s} {'Corr(grade_err, vvci_resid)':>30}")
    print("-" * 55)
    for name in DEFECT_NAMES:
        true_col = f"true_{name}_grade"
        pred_col = f"pred_{name}_grade"
        if true_col not in df.columns:
            continue
        grade_err = df[pred_col].values - df[true_col].values
        corr = np.corrcoef(grade_err, resid)[0, 1]
        print(f"{name:25s} {corr:>+.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].barh(stats_df["defect"], stats_df["exact_acc"], color="#534AB7")
    axes[0].set_xlabel("Exact grade accuracy")
    axes[0].set_title("Per-defect accuracy")
    axes[0].set_xlim(0, 1)

    axes[1].barh(stats_df["defect"], stats_df["within1_acc"], color="#1D9E75")
    axes[1].set_xlabel("Within-1 accuracy")
    axes[1].set_title("Within-1 grade accuracy")
    axes[1].set_xlim(0, 1)

    axes[2].barh(stats_df["defect"], stats_df["weight"], color="#BA7517")
    axes[2].set_xlabel("VCI weight (%)")
    axes[2].set_title("Defect VCI weights")

    plt.tight_layout()
    out_path = out_dir / "defect_contributions.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_path}")

if __name__ == "__main__":
    main()
