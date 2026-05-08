# VCI Estimator — Automated Pavement Condition Assessment for Uganda

Estimates the **Visual Condition Index (vVCI)** and per-defect grades from pavement photographs.
Works with both **Mobicap** survey images (ministry) and **smartphone** images (field engineers).

---

## Project structure

```
vci_estimator/
├── configs/config.yaml          # all hyperparameters
├── src/
│   ├── data/
│   │   ├── parse_excel.py       # load Mobicap Excel, compute vVCI
│   │   ├── extract_gps.py       # unified GPS extractor (EXIF + OCR)
│   │   ├── build_dataset.py     # match images → segments → dataset.csv
│   │   └── dataset.py           # PyTorch Dataset + DataLoaders
│   ├── models/
│   │   └── model.py             # EfficientNet-B3 + dual heads
│   └── training/
│       ├── losses.py            # Huber vVCI loss + weighted CE per defect
│       └── trainer.py           # two-stage training loop
├── scripts/
│   ├── prepare_data.py          # STEP 1: build dataset.csv
│   ├── train_model.py           # STEP 2: train the model
│   ├── evaluate.py              # STEP 3: evaluate best checkpoint
│   ├── baseline.py              # image texture baseline comparison
│   ├── analyse_contributions.py # per-defect contribution analysis
│   ├── cross_road_eval.py       # cross-road generalisation test
│   ├── export_model.py          # export to TorchScript + ONNX
│   └── dataset_report.py        # dataset quality report
└── outputs/                     # generated files (dataset.csv, checkpoints, plots)
```

---

## Setup

```bash
# Python 3.10+ required
pip install torch torchvision timm pandas numpy Pillow \
            pytesseract scikit-learn scipy matplotlib seaborn pyyaml

# tesseract-ocr for Mobicap GPS extraction
sudo apt install tesseract-ocr        # Ubuntu/Debian
brew install tesseract                 # macOS
```

---

## Data layout

```
data/
  Mobicap_Paved_Network_Combined.xlsx   # from ministry
  images/
    jinja/
      2021_2022/   ← A001_LinkXXPAVEXXXXXX.jpg files
      2023_2024/
      2025_2026/
    hoima/
      2025_2026/
```

Smartphone images can go in any subfolder — GPS is read from EXIF automatically.

---

## Step 1 — Build dataset

```bash
python scripts/prepare_data.py \
    --excel  data/Mobicap_Paved_Network_Combined.xlsx \
    --images data/images \
    --output outputs/dataset.csv
```

This will:
1. Load all 6,051 survey segments from the Excel
2. Extract GPS from every image (EXIF for smartphones, OCR for Mobicap)
3. Match each image to its 1km segment using road-code-constrained GPS matching
4. Compute vVCI labels from the 6 visible defect grades
5. Assign temporal train/val/test splits per road

---

## Step 2 — Train

```bash
python scripts/train_model.py \
    --dataset     outputs/dataset.csv \
    --output-dir  outputs/ \
    --epochs      40 \
    --batch-size  32 \
    --device      cuda       # or cpu
```

Training runs in two stages:
- **Stage 1** (epochs 0–9): backbone frozen, heads only trained
- **Stage 2** (epochs 10–39): top 3 EfficientNet blocks unfrozen, lower LR for backbone

Best checkpoint saved to `outputs/checkpoints/best.pt`.

---

## Step 3 — Evaluate

```bash
# Full evaluation on test set
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --dataset    outputs/dataset.csv

# Image texture baseline (compare against CNN)
python scripts/baseline.py --dataset outputs/dataset.csv

# Defect contribution analysis
python scripts/analyse_contributions.py \
    --predictions outputs/evaluation/predictions.csv

# Cross-road generalisation (e.g. train on all, test on Hoima)
python scripts/cross_road_eval.py \
    --checkpoint    outputs/checkpoints/best.pt \
    --dataset       outputs/dataset.csv \
    --hold-out-road A001
```

---

## Step 4 — Export for deployment

```bash
python scripts/export_model.py \
    --checkpoint outputs/checkpoints/best.pt \
    --output-dir outputs/exported
```

Produces:
- `outputs/exported/model.torchscript.pt` — for FastAPI server
- `outputs/exported/model.onnx` — for edge/browser deployment
- `outputs/exported/model_meta.json` — model metadata

---

## Model outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `vvci` | float | 0–100 | Visual VCI from image-observable defects |
| `defect_logits` | 6 × (B,5) | — | Grade logits for each defect (argmax → grade 1–5) |

**Visible defects modelled:**
| # | Defect | VCI weight |
|---|--------|-----------|
| 0 | All cracking | 7% |
| 1 | Wide cracking | 10% |
| 2 | Ravelling / Disintegration | 5% |
| 3 | Bleeding | 2.5% |
| 4 | Drainage on road | 2.5% |
| 5 | Potholes / Failures | 7.5% |

**Note:** vVCI covers 34.5% of the full VCI formula. Rutting, roughness (IRI), and FWD are excluded as they require instrumented measurement.

---

## GPS sources

| Image source | GPS method |
|---|---|
| Smartphone (Android/iOS) | EXIF metadata (automatic) |
| Mobicap survey system | OCR of blue header text overlay |

The system auto-detects the source and applies the correct extraction method.

---

## Expected performance targets

| Metric | Target |
|--------|--------|
| vVCI test MAE | < 8.0 |
| vVCI test RMSE | < 12.0 |
| Defect grade accuracy (mean) | > 0.65 |
| Defect within-1 accuracy | > 0.90 |

These will be updated after training on the full dataset.
