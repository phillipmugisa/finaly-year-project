"""
src/models/pci_formula.py
--------------------------
Approximate PCI (Pavement Condition Index) estimation from defect grades.

PCI is defined by ASTM D6433 on a 0-100 scale (100 = perfect, 0 = failed).
Our 6 image-visible defects map to standard PCI distress categories.

Mapping
-------
Defect              PCI distress category           max deduct (high severity)
all_cracking     -> Fatigue cracking (alligator)    ~80
wide_cracking    -> Longitudinal/transverse cracking ~35
ravelling        -> Weathering/raveling              ~40
bleeding         -> Bleeding                         ~20
drainage_road    -> Depression / edge cracking       ~25
potholes         -> Potholes                         ~75

Approach
--------
Each defect grade (1-5) is mapped to a deduct value using a piecewise-linear
lookup calibrated to the ASTM D6433 standard deduct curves at representative
distress densities.  Grade 1 = no defect (deduct 0), grade 5 = worst observed.

The Corrected Deduct Value (CDV) is computed via the standard iterative
procedure: reduce the highest deduct value until only one deduct > 2 remains,
using the CDV correction curves (approximated by a simple formula).

Result: PCI = 100 − CDV, clamped to [0, 100].

Note
----
This is an *approximation* because:
  1. PCI uses continuous density measurements; we use 5-point ordinal grades.
  2. Some PCI distresses (ravelling severity levels, etc.) differ from MoWT
     grade definitions.
Label the output as "Estimated PCI (image-based, approximate)" in the UI.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Deduct value tables: grade (1-5) → deduct value
# Calibrated to ASTM D6433 curves at representative low/med/high densities.
# Grade 1 = absent/negligible, Grade 5 = severe/extensive.
# ---------------------------------------------------------------------------

_DEDUCT_TABLES: dict[str, list[float]] = {
    # Grade:                1     2     3     4     5
    "all_cracking":        [0.0, 12.0, 32.0, 58.0, 78.0],   # fatigue cracking
    "wide_cracking":       [0.0,  6.0, 16.0, 28.0, 38.0],   # L/T cracking
    "ravelling":           [0.0,  5.0, 14.0, 28.0, 42.0],   # weathering
    "bleeding":            [0.0,  3.0,  8.0, 15.0, 22.0],   # bleeding
    "drainage_road":       [0.0,  4.0, 10.0, 20.0, 28.0],   # depression/edge
    "potholes":            [0.0, 20.0, 45.0, 68.0, 80.0],   # potholes
}

_DEFECT_ORDER = [
    "all_cracking", "wide_cracking", "ravelling",
    "bleeding", "drainage_road", "potholes",
]

def _get_deduct(defect: str, grade: int) -> float:
    """Return deduct value for a defect at a given 1-5 grade."""
    grade = max(1, min(5, int(grade)))
    return _DEDUCT_TABLES[defect][grade - 1]


def _cdv_from_deducts(deducts: list[float]) -> float:
    """
    Compute Corrected Deduct Value (CDV) using the ASTM D6433 iterative procedure.

    Algorithm:
    1. Sort deducts descending; keep only values > 2 for q.
    2. At each step compute CDV = TDV / (1 + 0.003 * (q-1) * TDV)
       — a parametric approximation of the ASTM Figure-3 correction curves.
    3. Replace the smallest deduct > 2 with 2 (reducing q by 1).
    4. Repeat; return the maximum CDV observed across all steps.
    """
    active = sorted([v for v in deducts if v > 2.0], reverse=True)
    # Include deducts ≤ 2 in the TDV contribution pool (fixed as 2 from the start)
    passive = sum(min(v, 2.0) for v in deducts if v <= 2.0)

    if not active:
        return 0.0

    best_cdv = 0.0
    working = list(active)

    while True:
        q   = len(working)                    # deducts still > 2
        tdv = sum(working) + passive
        cdv = tdv / (1.0 + 0.003 * (q - 1) * tdv) if q > 1 else tdv
        best_cdv = max(best_cdv, cdv)

        if q == 1:
            break
        # Replace lowest active deduct with 2
        passive    += 2.0
        working[-1] = None
        working     = [v for v in working[:-1] if v > 2.0]
        if not working:
            break

    return best_cdv


def grade_to_pci(grades: dict[str, int] | list[int]) -> float:
    """
    Compute approximate PCI from defect grades.

    Parameters
    ----------
    grades : dict mapping defect name → grade (1-5)
             OR list of 6 ints in DEFECT_ORDER order

    Returns
    -------
    float : estimated PCI in [0, 100]
    """
    if isinstance(grades, (list, tuple, np.ndarray)):
        grades = {name: int(g) for name, g in zip(_DEFECT_ORDER, grades)}

    deducts = [_get_deduct(name, grades.get(name, 1)) for name in _DEFECT_ORDER]
    cdv     = _cdv_from_deducts(deducts)
    return float(np.clip(100.0 - cdv, 0.0, 100.0))


def pci_label(pci: float) -> str:
    """Standard PCI condition labels (ASTM D6433)."""
    if pci >= 85: return "Good"
    if pci >= 70: return "Satisfactory"
    if pci >= 55: return "Fair"
    if pci >= 40: return "Poor"
    if pci >= 25: return "Very Poor"
    if pci >= 10: return "Serious"
    return "Failed"


def grades_tensor_to_pci(grade_indices: list[int]) -> tuple[float, str]:
    """
    Convenience wrapper for inference: takes 0-indexed grade indices (as
    returned by the model) and returns (pci, label).
    """
    grades_1indexed = [g + 1 for g in grade_indices]   # 0-4 → 1-5
    pci = grade_to_pci(grades_1indexed)
    return round(pci, 1), pci_label(pci)


# ---------------------------------------------------------------------------
# Batch version for dataset.csv analysis
# ---------------------------------------------------------------------------

def add_pci_to_dataframe(df, grade_cols: list[str] | None = None):
    """Add 'pci' and 'pci_label' columns to a dataset DataFrame in-place."""
    if grade_cols is None:
        grade_cols = [
            "all_cracking_grade", "wide_cracking_grade", "ravelling_grade",
            "bleeding_grade", "drainage_road_grade", "pothole_grade",
        ]
    defect_names = [c.replace("_grade", "") for c in grade_cols]

    def _row_pci(row):
        grades = {name: int(row[col]) for name, col in zip(defect_names, grade_cols)}
        return grade_to_pci(grades)

    df["pci"]       = df.apply(_row_pci, axis=1)
    df["pci_label"] = df["pci"].apply(pci_label)
    return df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        {"all_cracking": 1, "wide_cracking": 1, "ravelling": 1, "bleeding": 1, "drainage_road": 1, "potholes": 1},
        {"all_cracking": 2, "wide_cracking": 2, "ravelling": 2, "bleeding": 1, "drainage_road": 1, "potholes": 1},
        {"all_cracking": 3, "wide_cracking": 3, "ravelling": 3, "bleeding": 2, "drainage_road": 2, "potholes": 2},
        {"all_cracking": 4, "wide_cracking": 4, "ravelling": 4, "bleeding": 3, "drainage_road": 3, "potholes": 4},
        {"all_cracking": 5, "wide_cracking": 5, "ravelling": 5, "bleeding": 4, "drainage_road": 5, "potholes": 5},
    ]
    for g in tests:
        p = grade_to_pci(g)
        print(f"grades={list(g.values())}  PCI={p:.1f}  ({pci_label(p)})")
