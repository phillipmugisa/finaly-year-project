"""
api/segments.py
---------------
Loads all survey Excel files at startup and exposes a GPS→segment lookup.
Used by both the /nearest-segment endpoint and the /predict endpoint
(to auto-match a road segment when GPS is available).
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.build_dataset import haversine_km

_ROOT = Path(__file__).resolve().parent.parent

_DATASET_CSV = _ROOT / "outputs" / "dataset.csv"

_EXCEL_PATHS = [
    "/run/media/mugisa/New Volume/Road Condition Data/2025-26/Mobicap Paved Network - Combined  2025-26.xlsx",
    "/run/media/mugisa/New Volume/Road Condition Data/2023-24/Final Data Submitted 2024/Mobicap Paved 2023-2024.xlsx",
    "/run/media/mugisa/New Volume/Road Condition Data/2021-22/Mobicap Paved Network - Combined  2021-22.xlsx",
]

_segments: Optional[pd.DataFrame] = None


def load_segments() -> int:
    """
    Load segment lookup table. Tries sources in order (fastest first):
      1. outputs/dataset.csv  — already built, instant load
      2. Excel files on external drive — slow, only if CSV not available
    Returns number of unique segments loaded.
    """
    global _segments

    # ── Fast path: dataset.csv already built ──────────────────────────────
    if _DATASET_CSV.exists():
        try:
            df = pd.read_csv(_DATASET_CSV)
            _segments = (
                df.groupby(["road_code", "segment_start"], as_index=False)
                .first()
                .sort_values("survey_year", na_position="first")
                .drop_duplicates(subset=["road_code", "segment_start"], keep="last")
                .reset_index(drop=True)
            )
            print(f"  Segments loaded from dataset.csv ({len(_segments):,} unique segments)")
            return len(_segments)
        except Exception as e:
            print(f"  WARNING: dataset.csv load failed ({e}), trying Excel …")

    # ── Slow path: load Excel files from external drive ───────────────────
    from src.data.parse_excel import load_excel
    dfs = []
    for path in _EXCEL_PATHS:
        p = Path(path)
        if p.exists():
            try:
                dfs.append(load_excel(p))
                print(f"  Loaded {p.name}")
            except Exception as e:
                print(f"  WARNING: could not load {p.name}: {e}")

    if not dfs:
        print("  WARNING: no segment data found — /nearest-segment unavailable")
        return 0

    combined  = pd.concat(dfs, ignore_index=True)
    _segments = (
        combined
        .sort_values("survey_year", na_position="first")
        .drop_duplicates(subset=["road_code", "segment_start"], keep="last")
        .reset_index(drop=True)
    )
    return len(_segments)


def find_nearest(
    lat:       float,
    lon:       float,
    road_code: Optional[str] = None,
    max_dist_km: float = 5.0,
) -> Optional[dict]:
    """
    Return the nearest segment dict, or None if nothing within max_dist_km.
    If road_code is given, restrict search to that road first.
    """
    if _segments is None or len(_segments) == 0:
        return None

    df = _segments
    if road_code:
        narrowed = df[df["road_code"].str.upper() == road_code.upper()]
        if not narrowed.empty:
            df = narrowed

    valid = df.dropna(subset=["lat_centroid", "lon_centroid"])
    if valid.empty:
        return None

    dists = valid.apply(
        lambda r: haversine_km(lat, lon, r["lat_centroid"], r["lon_centroid"]),
        axis=1,
    )
    best_pos = dists.idxmin()
    dist     = float(dists[best_pos])

    if dist > max_dist_km:
        return None

    seg = valid.loc[best_pos]
    return {
        "road_code":    str(seg["road_code"]),
        "road_name":    str(seg["road_name"]),
        "km_start":     float(seg["segment_start"]),
        "km_end":       float(seg["segment_end"]),
        "lat_centroid": float(seg["lat_centroid"]),
        "lon_centroid": float(seg["lon_centroid"]),
        "survey_year":  int(seg["survey_year"]) if pd.notna(seg["survey_year"]) else None,
        "vci":          float(seg["vci"])  if pd.notna(seg["vci"])  else None,
        "vvci":         float(seg["vvci"]) if pd.notna(seg["vvci"]) else None,
        "distance_km":  round(dist, 3),
    }
