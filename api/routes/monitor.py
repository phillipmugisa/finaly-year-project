"""
api/routes/monitor.py
---------------------
GET /monitor/status  — live pipeline + training status for the dashboard.

Reads log files and /proc to report what's running without requiring
the processes to be tracked by PID (PIDs change each session).
"""

import re
import csv
from pathlib import Path
from typing import Optional

from fastapi import APIRouter

router = APIRouter()

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
_OUTPUTS = _ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helpers — process detection
# ---------------------------------------------------------------------------

def _is_running(keyword: str) -> bool:
    """Return True if any python process has `keyword` in its cmdline."""
    import os
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode(errors="replace")
            if keyword in cmdline:
                return True
        except OSError:
            pass
    return False


# ---------------------------------------------------------------------------
# Helpers — pipeline log parser
# ---------------------------------------------------------------------------

def _parse_pipeline_log(path: Path) -> dict:
    if not path.exists():
        return {"status": "not_started", "sections": []}

    text = path.read_text(errors="replace")
    lines = text.splitlines()

    sections = []
    current: Optional[dict] = None

    gps_re      = re.compile(r"GPS progress:\s*(\d+)/(\d+)\s*\(exif=(\d+),\s*ocr=(\d+),\s*failed=(\d+)\)")
    result_re   = re.compile(r"GPS results:.*?(\d+) EXIF.*?(\d+) OCR.*?(\d+) failed.*?total (\d+)")
    matched_re  = re.compile(r"Matched images\s*:\s*(\d+)")
    section_re  = re.compile(r"^---\s+([\w-]+):\s+(.+)")
    found_re    = re.compile(r"Found (\d+) images under (.+)")
    combined_re = re.compile(r"Combined dataset:\s*(\d+) images")

    for line in lines:
        m = section_re.match(line.strip())
        if m:
            if current:
                sections.append(current)
            current = {"year": m.group(1), "excel": m.group(2).strip(),
                       "image_dirs": [], "gps_done": 0, "gps_total": 0,
                       "matched": 0, "status": "running"}
            continue

        if current is None:
            continue

        m = found_re.search(line)
        if m:
            current["image_dirs"].append({"path": m.group(2).strip(), "count": int(m.group(1))})
            current["gps_total"] += int(m.group(1))
            continue

        m = gps_re.search(line)
        if m:
            current["gps_done"] = int(m.group(1))
            continue

        m = result_re.search(line)
        if m:
            current["gps_done"] = int(m.group(4))
            continue

        m = matched_re.search(line)
        if m:
            current["matched"] += int(m.group(1))
            current["status"] = "done"
            continue

    if current:
        sections.append(current)

    # Combined total
    total_matched = 0
    m = combined_re.search(text)
    if m:
        total_matched = int(m.group(1))
    else:
        total_matched = sum(s.get("matched", 0) for s in sections)

    # De-duplicate sections keeping last occurrence per year (two runs in same log)
    seen: dict[str, dict] = {}
    for s in sections:
        seen[s["year"]] = s
    sections = list(seen.values())

    running = _is_running("prepare_data")
    overall = "running" if running else ("done" if total_matched > 0 else "idle")

    return {
        "status":        overall,
        "total_matched": total_matched,
        "sections":      sections,
    }


# ---------------------------------------------------------------------------
# Helpers — feature extraction log parser
# ---------------------------------------------------------------------------

_TQDM_RE = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s+\[[\d:]+<([\d:]+),\s*([\d.]+)(?:s/it|it/s)\]"
)


def _parse_feature_extraction(log_path: Path) -> dict:
    result: dict = {
        "status":        "not_started",
        "pct":           0,
        "batches_done":  0,
        "batches_total": 0,
        "eta":           None,
        "speed":         None,
    }

    if not log_path.exists():
        return result

    text = log_path.read_text(errors="replace")
    if not text.strip():
        return result

    last_m = None
    for m in _TQDM_RE.finditer(text):
        last_m = m

    if last_m:
        result.update({
            "pct":           int(last_m.group(1)),
            "batches_done":  int(last_m.group(2)),
            "batches_total": int(last_m.group(3)),
            "eta":           last_m.group(4),
            "speed":         float(last_m.group(5)),
        })

    running = _is_running("extract_features")
    if running:
        result["status"] = "running"
    elif last_m and int(last_m.group(1)) >= 100:
        result["status"] = "complete"
    elif text.strip():
        result["status"] = "stopped"

    return result


# ---------------------------------------------------------------------------
# Helpers — training log parser
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\s*(\d+)\s+\((\d+)s\)\s+\|\s+"
    r"train loss ([\d.]+)\s+\|\s+"
    r"val MAE ([\d.]+)\s+RMSE ([\d.]+)\s+defect acc ([\d.]+)\s+\|\s+"
    r"lr ([\d.e+-]+)"
)
_BEST_RE  = re.compile(r"Best checkpoint saved \(val MAE = ([\d.]+)\)")
_STAGE2_RE = re.compile(r"Stage 2 start \(epoch (\d+)\)")


def _parse_training_log(log_path: Path, metrics_csv: Path) -> dict:
    result: dict = {
        "status":       "not_started",
        "current_epoch": None,
        "total_epochs":  None,
        "best_val_mae":  None,
        "best_epoch":    None,
        "stage":         1,
        "stage2_epoch":  None,
        "last_epoch_sec": None,
        "epochs":        [],
    }

    if not log_path.exists():
        return result

    text = log_path.read_text(errors="replace")

    m = _STAGE2_RE.search(text)
    if m:
        result["stage2_epoch"] = int(m.group(1))

    best_mae_val = float("inf")
    best_epoch_val = None
    for m in _BEST_RE.finditer(text):
        v = float(m.group(1))
        if v < best_mae_val:
            best_mae_val = v

    epochs = []
    for m in _EPOCH_RE.finditer(text):
        ep = int(m.group(1))
        total = int(m.group(2))
        epochs.append({
            "epoch":      ep,
            "elapsed_s":  int(m.group(3)),
            "train_loss": float(m.group(4)),
            "val_mae":    float(m.group(5)),
            "val_rmse":   float(m.group(6)),
            "defect_acc": float(m.group(7)),
            "lr":         float(m.group(8)),
        })
        result["total_epochs"] = total

    # Load full metrics from CSV if available (more complete)
    if metrics_csv.exists():
        try:
            with metrics_csv.open() as f:
                rows = list(csv.DictReader(f))
            if rows:
                epochs = [
                    {
                        "epoch":      int(r["epoch"]),
                        "train_loss": float(r.get("train_loss", 0)),
                        "val_mae":    float(r.get("val_mae_vvci", 0)),
                        "val_rmse":   float(r.get("val_rmse_vvci", 0)),
                        "defect_acc": float(r.get("val_acc_defect_mean", 0)),
                    }
                    for r in rows
                ]
        except Exception:
            pass

    if epochs:
        last = epochs[-1]
        result["current_epoch"] = last["epoch"]
        result["last_epoch_sec"] = last.get("elapsed_s")
        if best_mae_val < float("inf"):
            result["best_val_mae"] = best_mae_val
            # find which epoch had that mae
            for e in epochs:
                if abs(e["val_mae"] - best_mae_val) < 0.001:
                    best_epoch_val = e["epoch"]
            result["best_epoch"] = best_epoch_val
        if result["stage2_epoch"] and last["epoch"] >= result["stage2_epoch"]:
            result["stage"] = 2

    running = (
        _is_running("train_model")
        or _is_running("train_features")
        or _is_running("training_2122")
        or _is_running("trainer")
    )

    if running:
        result["status"] = "running"
    elif epochs:
        result["status"] = "complete" if (result["current_epoch"] == (result["total_epochs"] or 0) - 1) else "stopped"
    elif log_path.stat().st_size > 100:
        result["status"] = "starting"

    result["epochs"] = epochs
    return result


# ---------------------------------------------------------------------------
# Helpers — dataset stats
# ---------------------------------------------------------------------------

def _dataset_stats(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"rows": 0, "splits": {}}
    try:
        splits: dict[str, int] = {}
        rows = 0
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows += 1
                s = row.get("split", "unknown")
                splits[s] = splits.get(s, 0) + 1
        return {"rows": rows, "splits": splits}
    except Exception:
        return {"rows": 0, "splits": {}}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/monitor/status", tags=["Monitor"])
def monitor_status():
    """
    Returns live status of the training and data pipeline processes.
    Poll this endpoint (e.g. every 10s) to drive the monitor dashboard.
    """
    pipeline           = _parse_pipeline_log(_OUTPUTS / "prepare_data.log")
    feature_extraction = _parse_feature_extraction(_OUTPUTS / "extract_features.log")
    training           = _parse_training_log(
        _OUTPUTS / "training_features.log",
        _OUTPUTS / "metrics_features.csv",
    )
    retrain            = _parse_training_log(
        _OUTPUTS / "training_full.log",
        _OUTPUTS / "metrics_full.csv",
    )

    dataset_partial = _dataset_stats(_OUTPUTS / "dataset_2122.csv")
    dataset_full    = _dataset_stats(_OUTPUTS / "dataset.csv")

    watcher_running = _is_running("auto_extract")

    return {
        "pipeline":           pipeline,
        "feature_extraction": feature_extraction,
        "training":           training,
        "retrain":            retrain,
        "dataset_partial":    dataset_partial,
        "dataset_full":       dataset_full,
        "watcher_running":    watcher_running,
    }
