"""
parse_excel.py
--------------
Loads the Mobicap Paved Network Excel workbook, cleans the data,
computes the Visual VCI (vVCI) from the image-observable defects only,
and returns a tidy DataFrame with one row per 1km survey segment.

vVCI formula
------------
Each visible defect grade (1-5) contributes:
    contribution_i = weight_i * (5 - grade_i) / 4

With grade=1 (no defect) → full weight; grade=5 (worst) → 0.
vVCI = sum of contributions, then normalised to 0-100:
    vVCI_norm = (raw_vVCI / total_visual_weight) * 100

Potholes are stored as raw counts in the Excel; they are binned to a
1-5 grade using configurable thresholds before computing their contribution.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants (mirror config.yaml so this module is usable standalone)
# ---------------------------------------------------------------------------

VISIBLE_DEFECTS = [
    {"name": "all_cracking",   "col": "Cracksall",                "weight": 7.0},
    {"name": "wide_cracking",  "col": "Crackswide",               "weight": 10.0},
    {"name": "ravelling",      "col": "Ravelling/Disintegration",  "weight": 5.0},
    {"name": "bleeding",       "col": "Bleeding",                  "weight": 2.5},
    {"name": "drainage_road",  "col": "Drainage(onRoad)",          "weight": 2.5},
]

POTHOLE_COL     = "NrofPotholes/Failures"
POTHOLE_WEIGHT  = 7.5
POTHOLE_BINS    = [0, 1, 3, 6, 11]   # right-exclusive bin edges
TOTAL_VIS_W     = sum(d["weight"] for d in VISIBLE_DEFECTS) + POTHOLE_WEIGHT  # 34.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Uganda bounding box — used to reject sentinel values (999, -999, etc.)
_LAT_MIN, _LAT_MAX = -1.5,  4.3
_LON_MIN, _LON_MAX = 29.5, 35.1


def _parse_gps_string(gps_str) -> tuple[float | None, float | None]:
    """Parse 'lat lon' string into (lat, lon) floats.
    Returns (None, None) for sentinel values outside Uganda's bounding box.
    """
    if pd.isna(gps_str):
        return None, None
    parts = str(gps_str).strip().split()
    if len(parts) >= 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if _LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX:
                return lat, lon
        except ValueError:
            pass
    return None, None


def _safe_centroid(start: pd.Series, end: pd.Series) -> pd.Series:
    """Element-wise centroid: average if both valid, else whichever is non-null."""
    both   = start.notna() & end.notna()
    result = pd.Series(np.nan, index=start.index)
    result[both]                        = (start[both] + end[both]) / 2.0
    result[~both & start.notna()]       = start[~both & start.notna()]
    result[~both & end.notna()]         = end[~both & end.notna()]
    return result


def _to_numeric_grade(series: pd.Series) -> pd.Series:
    """
    Coerce a defect grade column to integer 1-5.
    Handles: integers, strings like '3', dashes '-' (treated as grade 1 / no defect).
    Values outside [1,5] are clamped.
    """
    s = pd.to_numeric(series, errors="coerce")
    # '-' or blank → treat as 1 (no defect observed / not applicable)
    s = s.fillna(1)
    return s.clip(1, 5).astype(int)


def _pothole_count_to_grade(counts: pd.Series, bins: list[int]) -> pd.Series:
    """
    Convert raw pothole count to ordinal grade 1-5.
    bins = [0, 1, 3, 6, 11] means:
        count == 0          → grade 1
        count in [1, 2]     → grade 2
        count in [3, 5]     → grade 3
        count in [6, 10]    → grade 4
        count >= 11         → grade 5
    """
    counts_num = pd.to_numeric(counts, errors="coerce").fillna(0).astype(int)
    grades = np.digitize(counts_num, bins=bins, right=False)
    grades = np.clip(grades, 1, 5)
    return pd.Series(grades, index=counts.index)


def _defect_contribution(grade: pd.Series, weight: float) -> pd.Series:
    """Compute a single defect's contribution to vVCI (raw, not normalised)."""
    return weight * (5 - grade) / 4.0


def _parse_survey_year(date_series: pd.Series) -> pd.Series:
    """
    Extract 4-digit year from date strings.
    Handles formats:
        '20250719 16:46'  (Mobicap YYYYMMDD HH:MM)
        '2025-09-04'
        '04/09/25'        (DD/MM/YY)
        Excel serial numbers (int/float)
    """
    years = []
    for val in date_series:
        if pd.isna(val):
            years.append(np.nan)
            continue
        s = str(val).strip()

        # Format: YYYYMMDD... — starts with 4-digit year 20xx
        m = re.match(r'^(20\d{2})', s)
        if m:
            years.append(int(m.group(1)))
            continue

        # Format: YYYY-MM-DD or YYYY/MM/DD
        m = re.search(r'(20\d{2})[-/]', s)
        if m:
            years.append(int(m.group(1)))
            continue

        # Format: DD/MM/YY (short year)
        m = re.search(r'\d{2}/\d{2}/(\d{2})$', s)
        if m:
            years.append(2000 + int(m.group(1)))
            continue

        # Excel serial number (days since 1900-01-01)
        try:
            serial = float(s)
            if 30000 < serial < 50000:  # reasonable Excel date range
                from datetime import date, timedelta
                d = date(1899, 12, 30) + timedelta(days=int(serial))
                years.append(d.year)
                continue
        except ValueError:
            pass

        years.append(np.nan)
    return pd.Series(years, index=date_series.index)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_excel(excel_path: str | Path, sheet: str = "Mobicap Paved") -> pd.DataFrame:
    """
    Load and clean the Mobicap Paved sheet.

    Returns
    -------
    pd.DataFrame with columns:
        road_code, road_name, segment_start, segment_end, segment_length,
        lat_start, lon_start, lat_end, lon_end, lat_centroid, lon_centroid,
        survey_year, region, station, vci,
        all_cracking, wide_cracking, ravelling, bleeding, drainage_road,
        pothole_count, pothole_grade,
        <defect>_grade  (for each visible defect),
        vvci_raw, vvci   (normalised 0-100)
    """
    # header=1 uses the short-code row as column names (e.g. 'cral', 'gps1date')
    # which has correctly formatted dates (YYYYMMDD HH:MM).
    # We also read header=0 to get the full English names for defect columns.
    df_short = pd.read_excel(excel_path, sheet_name=sheet, header=1)
    df_full  = pd.read_excel(excel_path, sheet_name=sheet, header=0)

    # Drop the duplicate-header row that appears in header=0 reads
    df_full = df_full[df_full["RoadCode"] != "RoadCode"].copy()
    df_full.reset_index(drop=True, inplace=True)

    # Use df_full for everything EXCEPT the date column (which is better in df_short)
    df = df_full.copy()

    # Graft the correctly formatted date from df_short
    # df_short row 0 corresponds to df_full row 0 (same data rows after header skip)
    df_short = df_short.iloc[:len(df)].reset_index(drop=True)
    if "gps1date" in df_short.columns:
        df["_gps1date_clean"] = df_short["gps1date"].values
    else:
        df["_gps1date_clean"] = np.nan

    # ---- GPS ---------------------------------------------------------------
    df[["lat_start", "lon_start"]] = df["gps1"].apply(
        lambda x: pd.Series(_parse_gps_string(x))
    )
    df[["lat_end", "lon_end"]] = df["gps2"].apply(
        lambda x: pd.Series(_parse_gps_string(x))
    )
    df["lat_centroid"] = _safe_centroid(df["lat_start"], df["lat_end"])
    df["lon_centroid"] = _safe_centroid(df["lon_start"], df["lon_end"])

    # ---- Survey year -------------------------------------------------------
    df["survey_year"] = _parse_survey_year(df["_gps1date_clean"])

    # ---- Segment metadata --------------------------------------------------
    df["road_code"]      = df["RoadCode"].astype(str).str.strip()
    df["road_name"]      = df["RoadName"].astype(str).str.strip()
    df["link_code"]      = df["Link"].astype(str).str.strip() if "Link" in df.columns else ""
    df["segment_start"]  = pd.to_numeric(df["SegmentStart"], errors="coerce")
    df["segment_end"]    = pd.to_numeric(df["SegmentEnd"],   errors="coerce")
    df["region"]         = df["Region"].astype(str).str.strip()
    df["station"]        = df["Station"].astype(str).str.strip()

    seg_len_col = "SegmentLength" if "SegmentLength" in df.columns else None
    df["segment_length"] = (
        pd.to_numeric(df[seg_len_col], errors="coerce")
        if seg_len_col else (df["segment_end"] - df["segment_start"])
    )

    # ---- VCI (full, from Excel) -------------------------------------------
    df["vci"] = pd.to_numeric(df["vci"], errors="coerce")

    # ---- Defect grades -----------------------------------------------------
    for defect in VISIBLE_DEFECTS:
        col_out = f"{defect['name']}_grade"
        df[col_out] = _to_numeric_grade(df[defect["col"]])

    # Potholes: count → grade
    df["pothole_count"] = pd.to_numeric(df[POTHOLE_COL], errors="coerce").fillna(0).astype(int)
    df["pothole_grade"] = _pothole_count_to_grade(df["pothole_count"], POTHOLE_BINS)

    # ---- vVCI computation --------------------------------------------------
    vvci_raw = pd.Series(0.0, index=df.index)
    for defect in VISIBLE_DEFECTS:
        grade_col = f"{defect['name']}_grade"
        vvci_raw += _defect_contribution(df[grade_col], defect["weight"])
    vvci_raw += _defect_contribution(df["pothole_grade"], POTHOLE_WEIGHT)

    df["vvci_raw"] = vvci_raw
    df["vvci"]     = (vvci_raw / TOTAL_VIS_W * 100.0).clip(0, 100)

    # ---- Final column selection --------------------------------------------
    grade_cols = [f"{d['name']}_grade" for d in VISIBLE_DEFECTS] + ["pothole_grade"]
    keep = [
        "road_code", "road_name", "link_code", "segment_start", "segment_end", "segment_length",
        "lat_start", "lon_start", "lat_end", "lon_end",
        "lat_centroid", "lon_centroid",
        "survey_year", "region", "station",
        "vci", "pothole_count",
    ] + grade_cols + ["vvci_raw", "vvci"]

    result = df[keep].dropna(subset=["vci", "lat_centroid", "lon_centroid"])
    result = result[
        (result["lat_centroid"] >= _LAT_MIN) & (result["lat_centroid"] <= _LAT_MAX) &
        (result["lon_centroid"] >= _LON_MIN) & (result["lon_centroid"] <= _LON_MAX)
    ].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Quick sanity check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Mobicap_Paved_Network_Combined.xlsx"
    df = load_excel(path)
    print(f"Loaded {len(df)} segments")
    print(df[["road_name", "station", "survey_year", "vci", "vvci"]].head(10))
    print("\nvVCI stats:\n", df["vvci"].describe())
    print("\nVCI stats:\n", df["vci"].describe())
    print("\nMissing centroids:", df["lat_centroid"].isna().sum())
