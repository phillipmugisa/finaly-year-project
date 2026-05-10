"""
build_dataset.py
----------------
Matches every Mobicap pavement image to its corresponding 1km survey segment
in the Excel spreadsheet, then writes a flat dataset.csv that the PyTorch
Dataset class can consume directly.

Matching strategy
-----------------
1. Extract GPS (lat, lon) from each image's OCR header.
2. For images where OCR fails (GPS not recovered), fall back to road-code +
   link matching using the filename.
3. For each image with GPS, find the nearest Excel segment centroid within
   max_match_dist_km using the Haversine distance.
4. Each segment can have many images (all images taken in that 1km interval).

Output CSV columns
------------------
image_path, road_code, road_name, segment_start, segment_end,
lat_centroid, lon_centroid, survey_year, region, station,
vci, vvci,
all_cracking_grade, wide_cracking_grade, ravelling_grade,
bleeding_grade, drainage_road_grade, pothole_grade,
split   (train / val / test)
"""

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .parse_excel import load_excel
from .extract_gps import extract_gps_batch, parse_filename


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Segment lookup structures
# ---------------------------------------------------------------------------

def _build_segment_index(segments: pd.DataFrame) -> tuple:
    """
    Return (lat_arr, lon_arr, idx_arr) numpy arrays for fast nearest-neighbour
    lookup.  idx_arr contains the integer index into `segments`.
    """
    valid = segments.dropna(subset=["lat_centroid", "lon_centroid"])
    lats = valid["lat_centroid"].to_numpy(dtype=float)
    lons = valid["lon_centroid"].to_numpy(dtype=float)
    idxs = valid.index.to_numpy()
    return lats, lons, idxs


def _nearest_segment(
    lat: float,
    lon: float,
    seg_lats: np.ndarray,
    seg_lons: np.ndarray,
    seg_idxs: np.ndarray,
    max_dist_km: float,
) -> int | None:
    """Return the DataFrame index of the nearest segment, or None if too far."""
    if len(seg_lats) == 0:
        return None
    # Vectorised approximate distance (Haversine is expensive; use equirectangular
    # approximation for candidate selection, then exact Haversine for the winner)
    dlat  = np.radians(seg_lats - lat)
    dlon  = np.radians(seg_lons - lon)
    clat  = math.cos(math.radians(lat))
    approx = np.sqrt(dlat ** 2 + (clat * dlon) ** 2) * 6371.0

    best_pos = int(np.argmin(approx))
    exact    = haversine_km(lat, lon, seg_lats[best_pos], seg_lons[best_pos])

    if exact <= max_dist_km:
        return int(seg_idxs[best_pos])
    return None


# ---------------------------------------------------------------------------
# Fallback 1: road-code + link matching (unambiguous roads only)
# ---------------------------------------------------------------------------

def _roadname_to_code(road_name_fragment: str, segments: pd.DataFrame) -> list[int]:
    """
    Return segment indices whose road_name or road_code contains the fragment.
    Used as a coarse fallback when GPS is unavailable.
    """
    frag = road_name_fragment.upper()
    mask = (
        segments["road_name"].str.upper().str.contains(frag, na=False)
        | segments["road_code"].str.upper().str.contains(frag, na=False)
    )
    return segments[mask].index.tolist()


# ---------------------------------------------------------------------------
# Fallback 2: link-sequence matching (for images with no GPS)
# ---------------------------------------------------------------------------

def _link_code_from_meta(road: str, link_num: str) -> str:
    """Build the Excel Link column value from filename road + link number.
    e.g. A001 + 03 → A00103,  A001N1 + 01 → A001N101
    """
    return road + link_num.zfill(2)


def _build_link_seq_map(
    image_paths: list,
    gps_map:     dict,
    segments:    pd.DataFrame,
    verbose:     bool = True,
) -> dict:
    """
    For images that have road+link in their filename but no GPS, assign
    segments by proportional sequence position within the link.

    Strategy
    --------
    For each (road, link) group:
      1. Collect all GPS-less images, sort by sequence number.
      2. Find matching Excel segments (same road_name + link_code), sorted by km.
      3. Map image rank i of N → segment bucket floor(i * S / N) of S segments.

    Returns dict: str(image_path) → segment DataFrame index
    """
    from collections import defaultdict

    if "link_code" not in segments.columns:
        return {}

    groups: dict = defaultdict(list)
    for img_path in image_paths:
        lat, lon = gps_map.get(str(img_path), (None, None))
        if lat is not None:
            continue                              # has GPS — skip
        meta = parse_filename(img_path.name)
        road = meta.get("road")
        link = meta.get("link")
        seq  = meta.get("seq")
        if road and link:
            groups[(road, link)].append((img_path, int(seq or 0)))

    if not groups:
        return {}

    result = {}
    n_assigned = 0

    for (road, link_num), img_seq_list in groups.items():
        lc = _link_code_from_meta(road, link_num)

        # Segments matching this (road, link)
        link_segs = segments[
            (segments["road_name"].str.upper() == road.upper()) &
            (segments["link_code"].str.upper() == lc.upper())
        ].sort_values("segment_start")

        if link_segs.empty:
            # Relax: all segments on this road (coarser but better than nothing)
            link_segs = segments[
                segments["road_name"].str.upper() == road.upper()
            ].sort_values("segment_start")

        if link_segs.empty:
            continue

        seg_list = link_segs.index.tolist()
        S = len(seg_list)
        img_seq_list.sort(key=lambda x: x[1])
        N = len(img_seq_list)

        for rank, (img_path, _) in enumerate(img_seq_list):
            bucket  = min(int(rank * S / N), S - 1)
            result[str(img_path)] = seg_list[bucket]
            n_assigned += 1

    if verbose and n_assigned:
        print(f"  Link-seq matches : {n_assigned}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def _assign_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    temporal:   bool  = True,
    seed:       int   = 42,
) -> pd.Series:
    """
    Assign a split label ('train'/'val'/'test') to each row.

    Temporal strategy: the most recent survey year for each road goes to test,
    the second most recent to val, all earlier to train.  Roads with only one
    year are split randomly.

    Random strategy: stratified by road_code × survey_year group so each road
    is represented in all splits.
    """
    splits = pd.Series("train", index=df.index)

    if temporal and "survey_year" in df.columns and df["survey_year"].notna().any():
        for road, grp in df.groupby("road_code"):
            years = sorted(grp["survey_year"].dropna().unique())
            if len(years) >= 3:
                splits.loc[grp[grp["survey_year"] == years[-1]].index] = "test"
                splits.loc[grp[grp["survey_year"] == years[-2]].index] = "val"
            elif len(years) == 2:
                splits.loc[grp[grp["survey_year"] == years[-1]].index] = "test"
                # val from random 15% of year[-2]
                sub = grp[grp["survey_year"] == years[-2]]
                val_n = max(1, int(len(sub) * val_frac / (train_frac + val_frac)))
                rng = np.random.default_rng(seed)
                val_idx = rng.choice(sub.index, size=val_n, replace=False)
                splits.loc[val_idx] = "val"
            else:
                # Only one year: random split within this road
                rng = np.random.default_rng(seed)
                idx = grp.index.to_numpy().copy()
                rng.shuffle(idx)
                n   = len(idx)
                nv  = max(1, int(n * val_frac))
                nt  = max(1, int(n * (1 - train_frac - val_frac)))
                splits.loc[idx[:nt]]          = "test"
                splits.loc[idx[nt:nt + nv]]   = "val"
        return splits

    # Random split (fallback)
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)
    n   = len(idx)
    nt  = max(1, int(n * (1 - train_frac - val_frac)))
    nv  = max(1, int(n * val_frac))
    for i in idx[:nt]:    splits.loc[i] = "test"
    for i in idx[nt:nt+nv]: splits.loc[i] = "val"
    return splits


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(
    excel_path:      str | Path,
    images_root:     str | Path,
    output_csv:      str | Path | None,
    max_match_dist_km: float = 0.6,
    train_frac:      float = 0.70,
    val_frac:        float = 0.15,
    temporal_split:  bool  = True,
    seed:            int   = 42,
    use_gps_cache:   bool  = True,
    verbose:         bool  = True,
    assign_splits:   bool  = True,
    gps_workers:     int   = 4,
) -> pd.DataFrame:
    """
    Full pipeline: Excel → GPS extraction → matching → CSV.

    Parameters
    ----------
    excel_path        : path to the Mobicap Excel workbook
    images_root       : root directory; all .jpg/.jpeg underneath are scanned
    output_csv        : where to write the final dataset.csv
    max_match_dist_km : images further than this from any segment centroid
                        are discarded
    train_frac, val_frac : split fractions (test = 1 - train - val)
    temporal_split    : use year-based split per road (recommended)
    seed              : random seed for reproducibility
    use_gps_cache     : cache OCR results next to each image

    Returns
    -------
    pd.DataFrame  (the dataset)
    """
    images_root = Path(images_root)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Excel segments
    # ------------------------------------------------------------------
    if verbose:
        print("Loading Excel …", flush=True)
    segments = load_excel(excel_path)
    if verbose:
        print(f"  {len(segments)} segments loaded", flush=True)

    seg_lats, seg_lons, seg_idxs = _build_segment_index(segments)

    # ------------------------------------------------------------------
    # 2. Discover all images
    # ------------------------------------------------------------------
    image_paths = sorted(
        p for p in images_root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg"}
    )
    if verbose:
        print(f"Found {len(image_paths)} images under {images_root}", flush=True)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No .jpg/.jpeg images found under {images_root}")

    # ------------------------------------------------------------------
    # 3. Extract GPS from image headers
    # ------------------------------------------------------------------
    if verbose:
        print("Extracting GPS from image headers (OCR) …", flush=True)
    gps_map = extract_gps_batch(image_paths, use_cache=use_gps_cache, verbose=verbose,
                                workers=gps_workers)

    # ------------------------------------------------------------------
    # 4. Match images → segments
    # ------------------------------------------------------------------
    if verbose:
        print("Matching images to survey segments …", flush=True)

    # Pre-build link-sequence map for GPS-less images (e.g. 2023-24)
    link_seq_map = _build_link_seq_map(image_paths, gps_map, segments, verbose)

    rows = []
    n_gps_match    = 0
    n_fallback     = 0
    n_link_seq     = 0
    n_unmatched    = 0

    for img_path in image_paths:
        lat, lon = gps_map.get(str(img_path), (None, None))
        meta     = parse_filename(img_path.name)
        seg_idx  = None

        if lat is not None and lon is not None:
            road_hint = meta.get("road")  # e.g. "A001N2"

            if road_hint:
                # Road-code-constrained match: prefer segments on the same road.
                # Uses a relaxed distance (1.5x) since the road code already
                # narrows candidates, reducing false matches in GPS collision zones.
                road_segs = segments[
                    segments["road_name"].str.upper() == road_hint.upper()
                ]
                if not road_segs.empty:
                    r_lats, r_lons, r_idxs = _build_segment_index(road_segs)
                    seg_idx = _nearest_segment(
                        lat, lon, r_lats, r_lons, r_idxs,
                        max_match_dist_km * 1.5
                    )

            # Fallback to global GPS match only when road-constrained match fails
            if seg_idx is None:
                seg_idx = _nearest_segment(
                    lat, lon, seg_lats, seg_lons, seg_idxs, max_match_dist_km
                )

            if seg_idx is not None:
                n_gps_match += 1

        # Fallback 1: road code unambiguous (only one segment on this road)
        if seg_idx is None and meta.get("road"):
            candidates = _roadname_to_code(meta["road"], segments)
            if len(candidates) == 1:
                seg_idx = candidates[0]
                n_fallback += 1

        # Fallback 2: link-sequence interpolation (for GPS-less Mobicap images)
        if seg_idx is None:
            seg_idx = link_seq_map.get(str(img_path))
            if seg_idx is not None:
                n_link_seq += 1

        if seg_idx is None:
            n_unmatched += 1
            continue

        seg = segments.loc[seg_idx]
        row = {
            "image_path":   str(img_path),
            "road_code":    seg["road_code"],
            "road_name":    seg["road_name"],
            "segment_start": seg["segment_start"],
            "segment_end":   seg["segment_end"],
            "lat_centroid":  seg["lat_centroid"],
            "lon_centroid":  seg["lon_centroid"],
            "survey_year":   seg["survey_year"],
            "region":        seg["region"],
            "station":       seg["station"],
            "vci":           seg["vci"],
            "vvci":          seg["vvci"],
            "all_cracking_grade":  seg["all_cracking_grade"],
            "wide_cracking_grade": seg["wide_cracking_grade"],
            "ravelling_grade":     seg["ravelling_grade"],
            "bleeding_grade":      seg["bleeding_grade"],
            "drainage_road_grade": seg["drainage_road_grade"],
            "pothole_grade":       seg["pothole_grade"],
        }
        rows.append(row)

    if verbose:
        print(f"  GPS matches   : {n_gps_match}", flush=True)
        print(f"  Road fallback : {n_fallback}", flush=True)
        print(f"  Link-seq      : {n_link_seq}", flush=True)
        print(f"  Unmatched     : {n_unmatched}", flush=True)

    if len(rows) == 0:
        raise RuntimeError(
            "No images could be matched to any segment. "
            "Check images_root and that GPS OCR is working."
        )

    dataset = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Assign train/val/test splits (optional — skip in multi-year mode)
    # ------------------------------------------------------------------
    if assign_splits:
        if verbose:
            print("Assigning train/val/test splits …", flush=True)

        seg_split = _assign_splits(segments, train_frac, val_frac, temporal_split, seed)

        segments["_split"] = seg_split
        split_dict = {
            (str(r["road_code"]), float(r["segment_start"])): r["_split"]
            for _, r in segments.iterrows()
        }

        dataset["split"] = [
            split_dict.get((str(r["road_code"]), float(r["segment_start"])), "train")
            for _, r in dataset.iterrows()
        ]

    # ------------------------------------------------------------------
    # 6. Save (only when output path provided and splits assigned)
    # ------------------------------------------------------------------
    if assign_splits and output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_csv, index=False)
        if verbose:
            split_counts = dataset["split"].value_counts().to_dict()
            print(f"\nDataset saved to {output_csv}", flush=True)
            print(f"  Total rows : {len(dataset)}", flush=True)
            print(f"  Splits     : {split_counts}", flush=True)
            print(f"\nvVCI distribution:\n{dataset['vvci'].describe()}", flush=True)

    return dataset


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--excel",      required=True, help="Path to Mobicap Excel file")
    p.add_argument("--images",     required=True, help="Root folder of images")
    p.add_argument("--output",     default="outputs/dataset.csv")
    p.add_argument("--max-dist",   type=float, default=0.6)
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    build_dataset(
        excel_path=args.excel,
        images_root=args.images,
        output_csv=args.output,
        max_match_dist_km=args.max_dist,
        temporal_split=not args.no_temporal,
        seed=args.seed,
    )
