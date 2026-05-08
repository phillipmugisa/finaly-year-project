"""
scripts/export_clean_excel.py
-----------------------------
Exports a cleaned Excel workbook from each Mobicap survey year.

Each output file contains only the columns used by the VCI estimator,
with human-readable names, and only rows that have valid GPS + VCI.
A combined workbook (all years) is also written.

Usage
-----
python scripts/export_clean_excel.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.parse_excel import load_excel


# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "year_tag": "2021-22",
        "path": "/run/media/mugisa/New Volume/Road Condition Data/2021-22/"
                "Mobicap Paved Network - Combined  2021-22.xlsx",
    },
    {
        "year_tag": "2023-24",
        "path": "/run/media/mugisa/New Volume/Road Condition Data/2023-24/"
                "Final Data Submitted 2024/Mobicap Paved 2023-2024.xlsx",
    },
    {
        "year_tag": "2025-26",
        "path": "/run/media/mugisa/New Volume/Road Condition Data/2025-26/"
                "Mobicap Paved Network - Combined  2025-26.xlsx",
    },
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "clean_excel"

# ---------------------------------------------------------------------------
# Column rename map  (internal name → human-readable Excel header)
# ---------------------------------------------------------------------------

RENAME = {
    "road_code":            "Road Code",
    "road_name":            "Road Name",
    "link_code":            "Link Code",
    "region":               "Region",
    "station":              "Station",
    "segment_start":        "Segment Start (km)",
    "segment_end":          "Segment End (km)",
    "segment_length":       "Segment Length (km)",
    "survey_year":          "Survey Year",
    "lat_start":            "GPS Lat Start",
    "lon_start":            "GPS Lon Start",
    "lat_end":              "GPS Lat End",
    "lon_end":              "GPS Lon End",
    "lat_centroid":         "GPS Lat (centroid)",
    "lon_centroid":         "GPS Lon (centroid)",
    "vci":                  "VCI (full, from survey)",
    "vvci":                 "vVCI (visual, 0–100)",
    "vvci_raw":             "vVCI raw score",
    "all_cracking_grade":   "All Cracking (grade 1–5)",
    "wide_cracking_grade":  "Wide Cracking (grade 1–5)",
    "ravelling_grade":      "Ravelling / Disintegration (grade 1–5)",
    "bleeding_grade":       "Bleeding (grade 1–5)",
    "drainage_road_grade":  "Drainage on Road (grade 1–5)",
    "pothole_count":        "Pothole Count (raw)",
    "pothole_grade":        "Pothole Grade (1–5)",
}

COLUMN_ORDER = list(RENAME.keys())


def _style_and_write(df: pd.DataFrame, writer: pd.ExcelWriter, sheet: str) -> None:
    """Write df to sheet with frozen header row and auto-width columns."""
    df.to_excel(writer, sheet_name=sheet, index=False)
    ws = writer.sheets[sheet]

    # Auto-fit column widths (approximate)
    for col_idx, col_name in enumerate(df.columns):
        max_len = max(
            len(str(col_name)),
            df[col_name].astype(str).str.len().max() if len(df) > 0 else 0,
        )
        ws.set_column(col_idx, col_idx, min(max_len + 2, 40))

    # Freeze header row
    ws.freeze_panes(1, 0)


def export_year(year_tag: str, source_path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  {year_tag}  →  {Path(source_path).name}")
    print(f"{'='*55}")

    df = load_excel(source_path)
    print(f"  Loaded  : {len(df):,} segments  ({df['road_name'].nunique()} roads)")

    # Keep only columns we use (some may be absent in older files)
    cols_present = [c for c in COLUMN_ORDER if c in df.columns]
    df = df[cols_present].rename(columns=RENAME)

    # Add year tag column at front for combined sheet
    df.insert(0, "Survey Period", year_tag)

    out_path = OUTPUT_DIR / f"segments_{year_tag}.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Drop the Survey Period column for individual files (it's all the same)
        _style_and_write(df.drop(columns=["Survey Period"]), writer, "Segments")

        # Summary sheet
        summary = pd.DataFrame({
            "Metric": [
                "Total segments",
                "Roads",
                "Survey year(s)",
                "VCI mean",
                "VCI std",
                "vVCI mean",
                "vVCI std",
                "Segments with GPS",
            ],
            "Value": [
                len(df),
                df["Road Name"].nunique(),
                ", ".join(str(int(y)) for y in sorted(
                    df["Survey Year"].dropna().unique())),
                f"{df['VCI (full, from survey)'].mean():.1f}",
                f"{df['VCI (full, from survey)'].std():.1f}",
                f"{df['vVCI (visual, 0–100)'].mean():.1f}",
                f"{df['vVCI (visual, 0–100)'].std():.1f}",
                f"{df['GPS Lat (centroid)'].notna().sum():,}",
            ],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"  Saved   : {out_path}")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    year_dfs = []
    for src in SOURCES:
        path = Path(src["path"])
        if not path.exists():
            print(f"WARNING: not found — {path}")
            continue
        df_year = export_year(src["year_tag"], str(path))
        year_dfs.append(df_year)

    if not year_dfs:
        sys.exit("No Excel files were found.")

    # Combined workbook
    combined = pd.concat(year_dfs, ignore_index=True)
    combined_path = OUTPUT_DIR / "segments_combined.xlsx"

    with pd.ExcelWriter(combined_path, engine="xlsxwriter") as writer:
        _style_and_write(combined, writer, "All Years")

        # Per-year sheets
        for df_yr in year_dfs:
            tag = df_yr["Survey Period"].iloc[0]
            _style_and_write(df_yr.drop(columns=["Survey Period"]), writer, tag)

        # Summary across years
        rows = []
        for df_yr in year_dfs:
            tag = df_yr["Survey Period"].iloc[0]
            rows.append({
                "Survey Period":    tag,
                "Segments":         len(df_yr),
                "Roads":            df_yr["Road Name"].nunique(),
                "VCI mean":         round(df_yr["VCI (full, from survey)"].mean(), 1),
                "VCI std":          round(df_yr["VCI (full, from survey)"].std(), 1),
                "vVCI mean":        round(df_yr["vVCI (visual, 0–100)"].mean(), 1),
                "vVCI std":         round(df_yr["vVCI (visual, 0–100)"].std()  , 1),
            })
        rows.append({
            "Survey Period": "COMBINED",
            "Segments":      len(combined),
            "Roads":         combined["Road Name"].nunique(),
            "VCI mean":      round(combined["VCI (full, from survey)"].mean(), 1),
            "VCI std":       round(combined["VCI (full, from survey)"].std(), 1),
            "vVCI mean":     round(combined["vVCI (visual, 0–100)"].mean(), 1),
            "vVCI std":      round(combined["vVCI (visual, 0–100)"].std(), 1),
        })
        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)

    print(f"\nCombined workbook saved → {combined_path}")
    print(f"  Total segments : {len(combined):,}")
    print(f"  Roads          : {combined['Road Name'].nunique()}")

    print("\n--- vVCI distribution (all years combined) ---")
    print(combined["vVCI (visual, 0–100)"].describe().round(2).to_string())

    print("\n--- VCI vs vVCI correlation ---")
    r = combined["VCI (full, from survey)"].corr(combined["vVCI (visual, 0–100)"])
    print(f"  Pearson r = {r:.4f}")


if __name__ == "__main__":
    main()
