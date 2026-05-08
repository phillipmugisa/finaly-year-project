"""
scripts/dataset_report.py
Generates a quality report on dataset.csv after prepare_data.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DEFECT_COLS = ["all_cracking_grade","wide_cracking_grade","ravelling_grade","bleeding_grade","drainage_road_grade","pothole_grade"]

DEFECT_NAMES = [
    "All cracking", "Wide cracking", "Ravelling",
    "Bleeding", "Drainage", "Potholes"
]

def report(csv_path: str, output_dir: str = "outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"DATASET QUALITY REPORT")
    print(f"Source : {csv_path}")
    print(f"{'='*60}")

    print(f"\n[1] Size")
    print(f"  Total images  : {len(df):,}")
    for split, grp in df.groupby("split"):
        segs = grp.groupby(["road_code","segment_start"]).ngroups
        print(f"  {split:6s}: {len(grp):5,} images  |  {segs:4d} unique segments")

    print(f"\n[2] Roads covered")
    for road, grp in df.groupby("road_name"):
        print(f"  {road:10s}: {len(grp):5,} images  |  "
              f"vVCI mean={grp['vvci'].mean():.1f}  min={grp['vvci'].min():.0f}  max={grp['vvci'].max():.0f}")

    print(f"\n[3] vVCI distribution (all)")
    print(df["vvci"].describe().round(2).to_string())

    print(f"\n[4] Grade distributions (train split)")
    train = df[df["split"] == "train"]
    for col, name in zip(DEFECT_COLS, DEFECT_NAMES):
        dist = train[col].value_counts().sort_index().to_dict()
        total = len(train)
        pct = {k: f"{v/total*100:.0f}%" for k, v in dist.items()}
        print(f"  {name:20s}: {dist}  {pct}")

    print(f"\n[5] Segments with zero matched images")
    # We can't compute this directly from the CSV alone, skip
    print("  (check match rate printed by prepare_data.py)")

    # ── Plots ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # vVCI histogram per split
    ax0 = fig.add_subplot(gs[0, :2])
    colors = {"train": "#1D9E75", "val": "#BA7517", "test": "#D85A30"}
    for split, grp in df.groupby("split"):
        ax0.hist(grp["vvci"], bins=20, alpha=0.6, label=split,
                 color=colors.get(split, "gray"), edgecolor="none")
    ax0.set_xlabel("vVCI"); ax0.set_ylabel("Count")
    ax0.set_title("vVCI distribution by split"); ax0.legend()

    # vVCI vs VCI scatter
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.scatter(df["vci"], df["vvci"], alpha=0.15, s=4, color="#534AB7")
    ax1.set_xlabel("VCI (full)"); ax1.set_ylabel("vVCI (visual)")
    ax1.set_title("VCI vs vVCI")
    from scipy.stats import pearsonr
    r, _ = pearsonr(df["vci"].dropna(), df["vvci"].dropna())
    ax1.text(0.05, 0.92, f"r = {r:.3f}", transform=ax1.transAxes, fontsize=9)

    # Grade distributions
    for i, (col, name) in enumerate(zip(DEFECT_COLS, DEFECT_NAMES)):
        row, col_pos = divmod(i, 3)
        ax = fig.add_subplot(gs[row + 1, col_pos])
        vals = train[col].value_counts().sort_index()
        ax.bar(vals.index, vals.values, color="#378ADD", edgecolor="none")
        ax.set_title(name, fontsize=9); ax.set_xlabel("Grade"); ax.set_ylabel("n")
        ax.set_xticks([1, 2, 3, 4, 5])

    out_path = output_dir / "dataset_report.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_path}")

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    report(csv, out)
