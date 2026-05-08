"""
scripts/baseline.py
-------------------
Two baselines for comparison with the CNN model:

1. Segment-level formula baseline  (theoretical upper bound)
   Uses the ground-truth defect grades from the Excel to compute vVCI.
   This shows how accurate the vVCI formula itself is vs full VCI.
   MAE = 0 by definition (it IS the formula). Reports VCI-vs-vVCI gap.

2. Image texture baseline  (practical lower bound for image-based methods)
   Extracts simple hand-crafted features from images (colour stats, edge
   density, LBP texture) and trains Ridge Regression to predict vVCI.
   This is what a non-deep-learning image approach achieves.

The CNN must beat baseline 2 to justify the deep learning approach.

Usage
-----
python scripts/baseline.py --dataset outputs/dataset.csv
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image

DEFECT_COLS = [
    "all_cracking_grade", "wide_cracking_grade", "ravelling_grade",
    "bleeding_grade",     "drainage_road_grade", "pothole_grade",
]

# ---------------------------------------------------------------------------
# Hand-crafted image features
# ---------------------------------------------------------------------------

HEADER_PX = 80   # Mobicap header height to skip

def extract_image_features(image_path: str) -> np.ndarray | None:
    """
    Extract 24-dimensional hand-crafted feature vector from one image:
      - 9 colour stats (mean/std/skewness per RGB channel)
      - 4 edge density stats (Sobel-like gradients on grayscale)
      - 8 texture stats (simple LBP-like neighbourhood differences)
      - 3 darkness/brightness percentiles
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        img = img.crop((0, HEADER_PX, w, h))
        img = img.resize((128, 96))
        arr = np.array(img, dtype=float) / 255.0

        feats = []
        # Colour stats per channel
        for ch in range(3):
            c = arr[:, :, ch].ravel()
            feats += [c.mean(), c.std(), float(((c - c.mean()) ** 3).mean())]

        # Edge density (difference between adjacent pixels)
        gray = arr.mean(axis=2)
        dx   = np.abs(np.diff(gray, axis=1)).mean()
        dy   = np.abs(np.diff(gray, axis=0)).mean()
        dx2  = np.abs(np.diff(gray, axis=1, n=2)).mean()
        dy2  = np.abs(np.diff(gray, axis=0, n=2)).mean()
        feats += [dx, dy, dx2, dy2]

        # Texture (variance in local 4x4 patches)
        patches = gray[:96, :128].reshape(24, 4, 32, 4).mean(axis=(1, 3)).ravel()
        feats += [patches.mean(), patches.std(),
                  np.percentile(patches, 25), np.percentile(patches, 75),
                  patches.max() - patches.min(),
                  (patches < patches.mean()).mean(),
                  (patches > patches.mean() + patches.std()).mean(),
                  (patches < patches.mean() - patches.std()).mean()]

        # Darkness percentiles
        for p in [10, 50, 90]:
            feats.append(float(np.percentile(gray, p)))

        return np.array(feats, dtype=float)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    # ------------------------------------------------------------------
    # Baseline 1: VCI vs vVCI formula gap (segment-level, no images)
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("BASELINE 1 — Formula gap (VCI vs vVCI, segment-level)")
    print(f"{'='*55}")
    seg_test = df[df["split"] == "test"]
    if len(seg_test) == 0:
        seg_test = df  # fallback if splits not yet assigned
    vci_mae  = mean_absolute_error(seg_test["vci"], seg_test["vvci"])
    vci_rmse = float(np.sqrt(mean_squared_error(seg_test["vci"], seg_test["vvci"])))
    r, _     = __import__("scipy.stats", fromlist=["pearsonr"]).pearsonr(df["vci"], df["vvci"])
    print(f"  vVCI vs full VCI  MAE  : {vci_mae:.2f}")
    print(f"  vVCI vs full VCI  RMSE : {vci_rmse:.2f}")
    print(f"  Pearson r(VCI, vVCI)   : {r:.4f}")
    print(f"  (This gap = the unmeasured rutting/IRI/FWD contribution)")
    print(f"  CNN target: predict vVCI from images. Upper bound MAE=0.")

    # ------------------------------------------------------------------
    # Baseline 2: Image texture features → vVCI
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("BASELINE 2 — Image texture features → vVCI (Ridge)")
    print(f"{'='*55}")

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    if len(test) == 0:
        print("  No test split found — skipping image baseline.")
        return

    print(f"  Extracting features from {len(train)+len(val)+len(test)} images ...")
    def get_feats(subset):
        feats, targets = [], []
        for _, row in subset.iterrows():
            f = extract_image_features(row["image_path"])
            if f is not None:
                feats.append(f)
                targets.append(float(row["vvci"]))
        return np.array(feats), np.array(targets)

    X_tr, y_tr = get_feats(train)
    X_va, y_va = get_feats(val)
    X_te, y_te = get_feats(test)
    print(f"  Features extracted: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    if len(X_te) < 2:
        print("  Too few test images — need more data to run this baseline.")
        return

    sc    = StandardScaler()
    X_tr  = sc.fit_transform(X_tr)
    X_va  = sc.transform(X_va) if len(X_va) > 0 else X_va
    X_te  = sc.transform(X_te)

    best_mae, best_m = float("inf"), None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        m   = Ridge(alpha=alpha).fit(X_tr, y_tr)
        mae = mean_absolute_error(y_va, m.predict(X_va)) if len(X_va)>0 else float("inf")
        if mae < best_mae:
            best_mae, best_m = mae, m

    p_te  = best_m.predict(X_te)
    mae   = mean_absolute_error(y_te, p_te)
    rmse  = float(np.sqrt(mean_squared_error(y_te, p_te)))
    r2    = r2_score(y_te, p_te)
    print(f"\n  Test MAE  : {mae:.2f}   ← CNN must beat this")
    print(f"  Test RMSE : {rmse:.2f}")
    print(f"  Test R²   : {r2:.4f}")
    print(f"\n  Interpretation:")
    print(f"    vVCI std = {df['vvci'].std():.2f}  (baseline for naive mean predictor)")
    print(f"    Texture MAE = {mae:.2f}  (hand-crafted features)")
    print(f"    CNN target  < {mae:.2f}  (CNN should significantly beat this)")

    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        "baseline": ["formula_gap", "texture_ridge"],
        "mae":      [vci_mae, mae],
        "rmse":     [vci_rmse, rmse],
    })
    results.to_csv(Path(args.output_dir) / "baseline_results.csv", index=False)
    print(f"\n  Saved → {args.output_dir}/baseline_results.csv")

if __name__ == "__main__":
    main()
