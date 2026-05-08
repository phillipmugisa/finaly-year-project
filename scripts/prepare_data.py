"""
scripts/prepare_data.py
-----------------------
Step 1 of the pipeline.  Run this BEFORE training.

Single-year usage
-----------------
python scripts/prepare_data.py \
    --excel  "/path/to/Mobicap Paved Network - Combined  2025-26.xlsx" \
    --images "/path/to/images" \
    --output outputs/dataset.csv

Multi-year usage (recommended — correct per-year label assignment)
------------------------------------------------------------------
python scripts/prepare_data.py \
    --excel  "/path/2021-22/Mobicap Paved Network - Combined  2021-22.xlsx" \
    --excel  "/path/2023-24/Final Data Submitted 2024/Mobicap Paved 2023-2024.xlsx" \
    --excel  "/path/2025-26/Mobicap Paved Network - Combined  2025-26.xlsx" \
    --images "/path/to/Jinja road" \
    --output outputs/dataset.csv

In multi-year mode each Excel file is auto-paired with the matching year
subfolder under --images (e.g. 2021-22 Excel → <images>/2021-22/).
Images are matched only to segments from the same survey year so that
defect labels are temporally correct.

Outputs
-------
outputs/dataset.csv   — one row per matched image with labels and split
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.build_dataset import build_dataset
from src.data.parse_excel   import load_excel


# ---------------------------------------------------------------------------
# Year-tag helpers
# ---------------------------------------------------------------------------

def _detect_year_tag(excel_path: Path) -> str | None:
    """
    Extract a year tag like '2021-22' from an Excel file path.
    Handles both short (2021-22) and long (2023-2024) year formats.
    """
    for part in reversed(excel_path.parts):
        m = re.search(r'(20\d{2})[-_](\d{2,4})', part)
        if m:
            y1, y2 = m.group(1), m.group(2)
            if len(y2) == 4:        # 2023-2024 → 2023-24
                y2 = y2[2:]
            return f"{y1}-{y2}"
    return None


def _find_year_images(images_root: Path, year_tag: str | None) -> Path | None:
    """Return the images subfolder matching the given year tag, or None."""
    if year_tag is None:
        return None
    # Exact match first
    candidate = images_root / year_tag
    if candidate.exists():
        return candidate
    # Fuzzy: any subfolder whose name contains the year digits
    short = year_tag.replace("-", "")
    for d in sorted(images_root.iterdir()):
        if d.is_dir() and (year_tag in d.name or short in d.name.replace("-", "")):
            return d
    return None


# ---------------------------------------------------------------------------
# Image-level temporal split (used in multi-year mode)
# ---------------------------------------------------------------------------

def _assign_image_splits(
    df:         pd.DataFrame,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> pd.Series:
    """
    Assign 'train'/'val'/'test' to each image row based on survey_year.

    Per road_code:
      - Most recent year  → test
      - Second most recent → val
      - Older years       → train
      - Single year       → random 70/15/15 split
    """
    splits = pd.Series("train", index=df.index)
    rng    = np.random.default_rng(seed)

    for road_code, grp in df.groupby("road_code"):
        years = sorted(grp["survey_year"].dropna().unique())
        if len(years) == 0:
            continue

        if len(years) >= 3:
            splits.loc[grp[grp["survey_year"] == years[-1]].index] = "test"
            splits.loc[grp[grp["survey_year"] == years[-2]].index] = "val"

        elif len(years) == 2:
            splits.loc[grp[grp["survey_year"] == years[-1]].index] = "test"
            sub = grp[grp["survey_year"] == years[-2]]
            n_val = max(1, int(len(sub) * val_frac))
            idx   = sub.index.to_numpy().copy()
            rng.shuffle(idx)
            splits.loc[idx[:n_val]] = "val"

        else:
            idx = grp.index.to_numpy().copy()
            rng.shuffle(idx)
            n  = len(idx)
            nt = max(1, int(n * 0.15))
            nv = max(1, int(n * val_frac))
            splits.loc[idx[:nt]]          = "test"
            splits.loc[idx[nt:nt + nv]]   = "val"

    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build dataset.csv from Mobicap images + Excel survey data"
    )

    parser.add_argument(
        "--excel", required=True, action="append", dest="excels",
        metavar="EXCEL_PATH",
        help="Mobicap Excel workbook. Repeat for multi-year mode.",
    )
    parser.add_argument(
        "--images", required=True, action="append", dest="images_roots",
        metavar="IMAGES_ROOT",
        help="Root directory containing pavement images. Repeat to include "
             "multiple locations (e.g. --images .../Jinja --images .../Hoima).",
    )
    parser.add_argument("--output",      default="outputs/dataset.csv",
                        help="Path to write the output CSV")
    parser.add_argument("--max-dist",    type=float, default=0.6,
                        help="Max GPS matching distance in km (default 0.6)")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Use random split instead of temporal split")
    parser.add_argument("--train-frac",  type=float, default=0.70)
    parser.add_argument("--val-frac",    type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no-cache",    action="store_true",
                        help="Disable GPS OCR cache (re-OCR every image)")

    args         = parser.parse_args()
    excel_paths  = [Path(e) for e in args.excels]
    images_roots = [Path(r) for r in args.images_roots]
    output_path  = Path(args.output)

    # ── Validate inputs ───────────────────────────────────────────────────
    for ep in excel_paths:
        if not ep.exists():
            sys.exit(f"Excel file not found: {ep}")
    for ir in images_roots:
        if not ir.exists():
            sys.exit(f"Images root not found: {ir}")

    # ======================================================================
    # SINGLE-YEAR MODE (original behaviour)
    # ======================================================================
    if len(excel_paths) == 1:
        print("=== Single-year mode ===", flush=True)
        ep = excel_paths[0]

        # Merge images from all roots into one temp directory (via a combined list)
        # For single-year mode just use the first images root
        images_root = images_roots[0]

        print("Excel sanity check …", flush=True)
        segments = load_excel(ep)
        print(f"  Segments with GPS + VCI : {len(segments)}", flush=True)
        print(f"  Roads                   : {segments['road_name'].nunique()}", flush=True)
        years = sorted(segments['survey_year'].dropna().unique().astype(int).tolist())
        print(f"  Survey years            : {years}", flush=True)
        print(f"  vVCI mean / std         : {segments['vvci'].mean():.1f} / {segments['vvci'].std():.1f}", flush=True)

        dataset = build_dataset(
            excel_path        = ep,
            images_root       = images_root,
            output_csv        = output_path,
            max_match_dist_km = args.max_dist,
            train_frac        = args.train_frac,
            val_frac          = args.val_frac,
            temporal_split    = not args.no_temporal,
            seed              = args.seed,
            use_gps_cache     = not args.no_cache,
            assign_splits     = True,
        )

    # ======================================================================
    # MULTI-YEAR MODE — per-year matching then combine + split
    # ======================================================================
    else:
        print(f"=== Multi-year mode ({len(excel_paths)} Excel files, "
              f"{len(images_roots)} image root(s)) ===\n", flush=True)

        year_dfs = []

        for ep in excel_paths:
            year_tag = _detect_year_tag(ep)

            # Collect matching year-subfolders from ALL image roots
            year_image_dirs = []
            for ir in images_roots:
                d = _find_year_images(ir, year_tag)
                if d is not None:
                    year_image_dirs.append(d)

            if not year_image_dirs:
                print(f"WARNING: no image subfolder found for year '{year_tag}' "
                      f"in any images root — skipping {ep.name}", flush=True)
                continue

            print(f"--- {year_tag}: {ep.name}", flush=True)
            for d in year_image_dirs:
                print(f"    Images: {d}", flush=True)

            segments = load_excel(ep)
            print(f"    Segments loaded: {len(segments)}", flush=True)

            # Run build_dataset once per image root for this year, then combine
            root_dfs = []
            for yr_img_dir in year_image_dirs:
                df_root = build_dataset(
                    excel_path        = ep,
                    images_root       = yr_img_dir,
                    output_csv        = None,
                    max_match_dist_km = args.max_dist,
                    train_frac        = args.train_frac,
                    val_frac          = args.val_frac,
                    temporal_split    = False,
                    seed              = args.seed,
                    use_gps_cache     = not args.no_cache,
                    assign_splits     = False,
                    verbose           = True,
                )
                root_dfs.append(df_root)

            df_year = pd.concat(root_dfs, ignore_index=True) if root_dfs else pd.DataFrame()
            year_dfs.append(df_year)
            print(f"    Matched images  : {len(df_year)}\n", flush=True)

        if not year_dfs:
            sys.exit("No images were matched. Check paths and that tesseract is installed.")

        dataset = pd.concat(year_dfs, ignore_index=True)
        print(f"Combined dataset: {len(dataset)} images across {len(year_dfs)} years", flush=True)

        # Assign temporal splits at the image level
        dataset["split"] = _assign_image_splits(
            dataset, val_frac=args.val_frac, seed=args.seed
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)

        print(f"\nDataset saved → {output_path}", flush=True)
        print(f"  Total rows : {len(dataset)}", flush=True)
        print(f"  Splits     : {dataset['split'].value_counts().to_dict()}", flush=True)
        print(f"\nvVCI distribution:\n{dataset['vvci'].describe().round(2)}", flush=True)

    # ── Summary (both modes) ──────────────────────────────────────────────
    print("\n=== Per-split stats ===", flush=True)
    print(dataset.groupby("split")[["vvci", "vci"]].describe().round(2).to_string(), flush=True)

    print("\nGrade distributions (train split):", flush=True)
    train = dataset[dataset["split"] == "train"]
    for col in [c for c in dataset.columns if c.endswith("_grade")]:
        dist = train[col].value_counts().sort_index().to_dict()
        print(f"  {col:25s}: {dist}", flush=True)


if __name__ == "__main__":
    main()
