# VCI Estimator — Automated Pavement Condition Assessment for Uganda

Estimates the **Visual Condition Index (vVCI)** and per-defect grades from pavement photographs.
Works with both **Mobicap** survey images (ministry) and **smartphone** images (field engineers).

---

## Project structure

```
vci_estimator/
├── configs/config.yaml              # all hyperparameters
├── colab_train.ipynb                # Colab notebook for GPU training
├── src/
│   ├── data/
│   │   ├── parse_excel.py           # load Mobicap Excel, compute vVCI
│   │   ├── extract_gps.py           # unified GPS extractor (EXIF + OCR)
│   │   ├── build_dataset.py         # match images → segments → dataset.csv
│   │   ├── dataset.py               # PyTorch Dataset + DataLoaders
│   │   └── rdd2022_dataset.py       # RDD2022 dataset for PCI pre-training
│   ├── models/
│   │   ├── model.py                 # EfficientNet-B3 + task heads (+ HeadsModel)
│   │   └── pci_formula.py           # ASTM D6433 PCI formula from grade predictions
│   └── training/
│       ├── losses.py                # Huber vVCI loss + weighted CE per defect
│       └── trainer.py               # two-stage training loop
├── scripts/
│   ├── prepare_data.py              # STEP 1: build dataset.csv from images + Excel
│   ├── extract_features.py          # STEP 2a: extract backbone features for Colab
│   ├── train_model.py               # STEP 2b: full model training (needs GPU + images)
│   ├── train_features.py            # STEP 2c: heads-only training on pre-extracted features
│   ├── pretrain_pci.py              # pre-train PCI head on RDD2022
│   ├── evaluate.py                  # STEP 3: evaluate best checkpoint on test set
│   ├── baseline.py                  # image texture baseline comparison
│   ├── analyse_contributions.py     # per-defect contribution analysis
│   ├── cross_road_eval.py           # cross-road generalisation test
│   ├── export_model.py              # export to TorchScript + ONNX
│   ├── dataset_report.py            # dataset quality report
│   ├── auto_extract.sh              # auto-runs feature extraction after pipeline
│   └── prepare_colab_upload.sh      # packages everything for Drive upload
├── api/                             # FastAPI inference server
│   ├── main.py
│   ├── inference.py
│   ├── image_utils.py
│   ├── routes/
│   │   ├── predict.py
│   │   ├── segments.py
│   │   └── monitor.py
│   └── schemas.py
└── web/                             # browser-based field app
    ├── index.html
    ├── css/main.css
    └── js/
        ├── api.js
        ├── field.js
        ├── batch.js
        ├── map.js
        ├── monitor.js
        └── visualizer.js
```

---

## Setup

```bash
# Python 3.10+ required
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# tesseract-ocr is required for Mobicap GPS extraction
sudo apt install tesseract-ocr        # Ubuntu/Debian
brew install tesseract                 # macOS
```

---

## Data layout

The pipeline expects Excel files and image directories in this structure:

```
Road Condition Data/
  2021-22/
    Mobicap Paved Network - Combined  2021-22.xlsx
  2023-24/
    Final Data Submitted 2024/
      Mobicap Paved 2023-2024.xlsx
  2025-26/
    Mobicap Paved Network - Combined  2025-26.xlsx

Jinja road/
  2021-22/    ←  6,973 images
  2023-24/    ← 42,789 images
  2025-26/    ← 59,492 images

Hoima road/
  2021-22/    ← 29,596 images
  2023-24/    ← 19,308 images
  2025-26/    ←  9,077 images
```

Total: ~167,000 images across 6 road-year combinations.

---

## Step 1 — Build dataset.csv

### Full multi-year command

```bash
source venv/bin/activate

python scripts/prepare_data.py \
    --excel  "/path/to/Road Condition Data/2021-22/Mobicap Paved Network - Combined  2021-22.xlsx" \
    --excel  "/path/to/Road Condition Data/2023-24/Final Data Submitted 2024/Mobicap Paved 2023-2024.xlsx" \
    --excel  "/path/to/Road Condition Data/2025-26/Mobicap Paved Network - Combined  2025-26.xlsx" \
    --images "/path/to/Hoima road" \
    --images "/path/to/Jinja road" \
    --output outputs/dataset.csv \
    --num-workers 8
```

> **`--images` order matters:** List `Hoima road` before `Jinja road` so that
> Jinja 2025-26 (the largest batch, ~60k images) is processed last. This way
> all other years are done first if you need to stop early.

> **`--num-workers`:** Set to the number of CPU cores on your machine
> (`nproc` on Linux). More cores = faster OCR. Default is 4.

### To run in the background and log output

```bash
nohup python scripts/prepare_data.py \
    --excel  "/path/to/.../2021-22.xlsx" \
    --excel  "/path/to/.../2023-2024.xlsx" \
    --excel  "/path/to/.../2025-26.xlsx" \
    --images "/path/to/Hoima road" \
    --images "/path/to/Jinja road" \
    --output outputs/dataset.csv \
    --num-workers 8 \
    >> outputs/prepare_data.log 2>&1 &

echo "Pipeline PID: $!"    # save this to monitor or kill later
tail -f outputs/prepare_data.log
```

### What it does (processing order)

| Step | Year | Road | Images | Notes |
|------|------|------|--------|-------|
| 1 | 2021-22 | Hoima | 29,596 | GPS via OCR |
| 2 | 2021-22 | Jinja | 6,973 | GPS via OCR |
| 3 | 2023-24 | Hoima | 19,308 | GPS via OCR |
| 4 | 2023-24 | Jinja | 42,789 | GPS via OCR (many fail → link-seq fallback) |
| 5 | 2025-26 | Hoima | 9,077 | GPS via OCR |
| 6 | 2025-26 | Jinja | 59,492 | GPS via OCR — **processed last** |

### Speed & output

| | Value |
|-|-------|
| Total images | ~167,000 |
| Fresh run (no cache) | ~8–10 hours with 8 workers |
| Repeat run (GPS cached) | ~30 minutes (reads `.gps.json` sidecars) |
| Output | `outputs/dataset.csv` (~2 MB, ~100k labelled image rows) |

**GPS cache:** Each image gets a `.gps.json` sidecar file written next to it on
first run. If you copy these sidecars alongside the images to another machine,
the pipeline finishes in ~30 min instead of hours.

---

## Model architecture

### Backbone
**EfficientNet-B3** (pretrained ImageNet, from `timm`). Input: 224×224 RGB.
Output: 1536-d feature vector via Global Average Pooling.

### Task heads (1.19M trainable parameters)

| Head | Architecture | Output |
|------|-------------|--------|
| **Defect** | Dropout → Linear(1536→256) → BN+ReLU → 6×[Dropout → Linear(256→5)] | 6 grade logits (1–5 per defect) |
| **vVCI** | Dropout → Linear(1536→256) → BN+ReLU → Dropout → Linear(256→1) → Sigmoid×100 | Scalar [0, 100] |
| **PCI** | Same as vVCI | Scalar [0, 100] |

The DefectHead uses a **shared bottleneck** (1536→256) so correlated defects
(cracking, ravelling) share a representation before branching.

### Loss function (4 terms)

```
L = λ_vvci   × Huber(pred_vvci,  true_vvci)       [λ=1.0]
  + λ_defect × Σ CrossEntropy(pred_grade_i, grade_i) [λ=0.5]
  + λ_pci    × Huber(pred_pci,   formula_pci)       [λ=0.5]
  + λ_consist× Huber(pred_vvci,  formula_vvci)      [λ=0.3]
```

**Consistency loss**: soft grade predictions (softmax × expected grade values)
are passed through the MoWT vVCI formula and compared with the vVCI head output.
This prevents the two heads from diverging during training.

---

## Step 2 — Train the model

There are two training paths depending on whether you have a GPU and access to
the raw images.

### Option A — Colab (recommended, no local GPU needed)

This path extracts backbone features locally (~2 hours on CPU), uploads ~500 MB
to Google Drive, then trains on a free Colab T4 GPU (~30–45 min).

**1. Extract backbone features locally:**
```bash
source venv/bin/activate

python scripts/extract_features.py \
    --dataset    outputs/dataset.csv \
    --output     outputs/features.npz \
    --batch-size 32 \
    --num-workers 4 \
    --resume
```

Output: `outputs/features.npz` (~500 MB, float16 1536-d vectors for every image).

**2. Package project scripts:**
```bash
zip -r outputs/vci_estimator_scripts.zip \
    src/ configs/ \
    scripts/train_features.py \
    scripts/pretrain_pci.py \
    -x "**/__pycache__/*" "**/*.pyc"
```

**3. Upload to Google Drive:**
```
MyDrive/vci_estimator/
  ├── features.npz              (~500 MB)
  ├── dataset.csv               (~2 MB)
  └── vci_estimator_scripts.zip (~small)
```

**4. Open `colab_train.ipynb` in Google Colab** with a T4 GPU runtime and run
all cells. The notebook will:
- Visualize the dataset (vVCI distribution, split breakdown, defect distributions)
- Download RDD2022 Japan from FigShare (~3 GB) for PCI head pre-training
- Pre-train the PCI head on international road damage data
- Train all three heads (Defect + vVCI + PCI) on Uganda features with consistency loss
- Generate training curves and post-training analysis plots
- Save `best.pt` back to Google Drive

**Key training flags used in the notebook:**
```bash
--resplit           # random 70/15/15 segment split (408 train vs 231 with temporal)
--lambda-consist 0.3  # consistency loss weight
--epochs 60         # more epochs on GPU since feature-based training is fast
```

**5. Download `best.pt`** from Drive to `outputs/checkpoints/best.pt`.

Or use the automated helper (runs feature extraction + packaging after pipeline):
```bash
bash scripts/auto_extract.sh <pipeline_pid>
```

### Option B — Full local training (requires GPU + raw images)

```bash
python scripts/train_model.py \
    --dataset     outputs/dataset.csv \
    --output-dir  outputs/ \
    --epochs      40 \
    --batch-size  32 \
    --device      cuda
```

Two-stage training:
- **Stage 1** (epochs 0–9): backbone frozen, heads only
- **Stage 2** (epochs 10–39): top 3 EfficientNet-B3 blocks unfrozen, lower backbone LR

Best checkpoint saved to `outputs/checkpoints/best.pt`.

---

## Step 3 — Evaluate

```bash
# Full evaluation on test set
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --dataset    outputs/dataset.csv

# Image texture baseline comparison
python scripts/baseline.py --dataset outputs/dataset.csv

# Per-defect contribution analysis
python scripts/analyse_contributions.py \
    --predictions outputs/evaluation/predictions.csv

# Cross-road generalisation test
python scripts/cross_road_eval.py \
    --checkpoint    outputs/checkpoints/best.pt \
    --dataset       outputs/dataset.csv \
    --hold-out-road A001
```

Works with both full-model and Colab feature-based checkpoints automatically.

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

## Step 5 — Run the API and web app

```bash
# Start inference API (port 8000)
uvicorn api.main:app --reload --port 8000

# Serve web app (port 3000)
cd web && python -m http.server 3000
# Open http://localhost:3000
```

API endpoints:
- `GET  /health` — liveness + model_ready flag
- `POST /predict` — 1+ images → vvci, pci, defect grades, GPS, road segment
- `POST /predict-batch` — ZIP of images → per-image CSV results
- `GET  /nearest-segment?lat=&lon=` — GPS → nearest road segment lookup
- `GET  /monitor/status` — pipeline and training progress (used by web Monitor tab)

---

## Model outputs

The `/predict` endpoint returns:

```json
{
  "model_ready": true,
  "vvci": 74.3,
  "vvci_label": "Fair",
  "pci": 61.2,
  "pci_label": "Satisfactory",
  "defects": [
    {
      "name": "all_cracking",
      "predicted_grade": 2,
      "confidence": 0.74,
      "grade_probs": [0.08, 0.74, 0.12, 0.04, 0.02]
    }
  ],
  "gps_used": {"lat": 0.4457, "lon": 33.199, "source": "ocr"},
  "road_matched": {"road_name": "A001N2", "km_start": 86.0, "km_end": 87.0},
  "images_used": 3
}
```

`grade_probs` is the full 5-class softmax distribution (grades 1–5), displayed
as probability bars in the web app.

**Visible defects modelled:**

| # | Defect | VCI weight |
|---|--------|-----------|
| 0 | All cracking | 7% |
| 1 | Wide cracking | 10% |
| 2 | Ravelling / Disintegration | 5% |
| 3 | Bleeding | 2.5% |
| 4 | Drainage on road | 2.5% |
| 5 | Potholes / Failures | 7.5% |

**Note:** vVCI covers 34.5% of the full VCI formula. Rutting, roughness (IRI),
and structural capacity (FWD) are excluded — they require instrumented measurement
and are not visible in images.

---

## GPS extraction

| Image source | GPS method |
|---|---|
| Smartphone (Android/iOS) | EXIF metadata (automatic) |
| Mobicap survey vehicle | OCR of blue text header (top 80px of image) |

The system auto-detects the source per image and applies the correct method.

---

## Current performance (feature-based training, 755 segments)

| Metric | Value |
|--------|-------|
| **Test MAE (vVCI)** | **16.6 points** (0–100 scale) |
| **Test RMSE (vVCI)** | 19.7 points |
| **Defect accuracy (mean)** | **44.6%** (5-class; random=20%) |
| Best val MAE | 14.9 |

Per-defect accuracy:

| Defect | Exact acc | Within-1 acc |
|--------|-----------|-------------|
| Potholes / Failures | ~85% | ~95% |
| Ravelling | ~39% | ~78% |
| Bleeding | ~30% | ~72% |
| Wide cracking | ~21% | ~65% |
| All cracking | ~18% | ~62% |
| Drainage on road | ~10% | ~58% |

**Key limitation:** only 755 unique survey segments (Jinja + Hoima roads). More
roads or end-to-end Colab training on raw images is expected to improve results.
