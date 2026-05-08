# CLAUDE.md — VCI Estimator Project Context

Read this file at the start of every session. It contains the full project
history, every key decision made, the current state, and what to build next.

---

## ⚠ CRITICAL PENDING TASKS (unblock before final submission)

These must be completed and checked off before the project is considered done.
Surface these at the start of every session until resolved.

- [ ] **RDD2022 PCI pre-training** — The PCI regression head currently trains
  from scratch on Uganda data only. Pre-training on RDD2022 (international road
  damage images) would significantly improve direct image→PCI accuracy.
  Blocked by: 13.3 GB download impractical on local machine; no GPU locally.
  **Action required:** On a machine with GPU + fast internet (e.g. Google Colab):
  ```bash
  python scripts/download_rdd2022.py --output data/rdd2022   # 13.3 GB from FigShare
  python scripts/pretrain_pci.py --data data/rdd2022 --device cuda \
      --epochs-stage1 10 --epochs-stage2 10 --batch-size 32
  # Then retrain the full model with pre-trained PCI weights:
  python scripts/train_model.py --dataset outputs/dataset.csv \
      --pci-pretrain outputs/pci_pretrain/pci_head.pt --device cuda --epochs 40
  ```
  Infrastructure is fully built: `scripts/download_rdd2022.py`,
  `scripts/pretrain_pci.py`, `src/data/rdd2022_dataset.py` are all complete.

---

## What this project is

An automated pavement condition assessment system for Uganda's road network.
Built for two users:

1. **Ministry of Works & Transport (MoWT)** — already uses Mobicap, an
   expensive vehicle-mounted survey system that captures GPS-tagged pavement
   images and records defect grades in Excel. The ministry wants to automate
   the visual rating step so they do not need to send large field teams.

2. **Field engineers without Mobicap** — engineers who only have a smartphone.
   They photograph pavement, upload images to a web app, and get an automated
   condition estimate. This dramatically reduces survey cost (Mobicap costs
   millions of UGX per survey; a smartphone costs nothing extra).

Both users share one model and one API. The image intake layer handles both
Mobicap images (GPS burned into blue text header) and smartphone images
(GPS in EXIF metadata).

---

## Key decisions made (do not revisit without good reason)

### VCI, not PCI
The ministry uses **VCI (Visual Condition Index)**, not PCI. Field teams go
out every 1km and assign defect grades (1–5 scale). VCI is a weighted linear
formula of those grades. The project originally proposed PCI but was corrected
after interaction with the ministry.

### Visual VCI (vVCI), not full VCI
Full VCI = weighted sum of ALL defect grades including:
- Rutting (L + R): 10% combined weight → **measured instrumentally, excluded**
- Roughness (IRI): 20% weight → **measured instrumentally, excluded**
- FWD (structural): 15% weight → **measured instrumentally, excluded**

The model predicts **vVCI** — VCI from image-observable defects only.
Total visual weight = **34.5% of full VCI**.

vVCI formula per defect:
```
contribution_i = weight_i × (5 - grade_i) / 4
vVCI_raw = sum of contributions
vVCI = (vVCI_raw / 34.5) × 100   ← normalised to 0-100
```

The web app labels output as "Visual Condition Index (image-based)" and notes
it excludes rutting, roughness, and structural components. This is honest and
consistent with how the ministry already separates visual from measured surveys.

### Six image-observable defects
After excluding measured defects AND defects not visible in images (side
drainage, edge drop, width loss, shoulder condition), only these remain:

| Defect | Excel column | VCI weight |
|--------|-------------|-----------|
| All cracking | Cracksall | 7% |
| Wide cracking | Crackswide | 10% |
| Ravelling / Disintegration | Ravelling/Disintegration | 5% |
| Bleeding | Bleeding | 2.5% |
| Drainage on road | Drainage(onRoad) | 2.5% |
| Potholes / Failures | NrofPotholes/Failures | 7.5% |

**Important:** Drainage on road only uses values 1, 3, 5 (odd-only scale) in
the Mobicap system. This is a data entry convention, not a bug.

**Important:** Potholes column is a raw count (0, 1, 2, 25...), not a grade.
It is binned to 1–5 using thresholds [0, 1, 3, 6, 11] before training.

### PCI is also a target (secondary)
The handwritten proposal includes PCI estimation using pretrained weights from
external datasets (Image2PCI, RDD2022, CrackForest). This is a secondary goal
for Phase 2 extension — the primary goal is vVCI from local labelled data.

### Multi-task architecture
One shared EfficientNet-B3 backbone feeds two heads:
- **Defect classification head**: 6 × Linear(5) — one per visible defect,
  outputs grade logits (1–5, stored as 0–4 class indices)
- **vVCI regression head**: MLP → sigmoid × 100 → scalar in [0, 100]

Training uses combined loss:
```
Total = λ_vvci × Huber(pred_vvci, true_vvci)
      + λ_defect × Σ CrossEntropy(pred_grade_i, true_grade_i)
```
Default: λ_vvci = 1.0, λ_defect = 0.5

Two training stages:
- Stage 1 (epochs 0–9): backbone frozen, heads only
- Stage 2 (epochs 10–39): unfreeze top 3 EfficientNet blocks, backbone LR × 0.1

### Road-code-constrained GPS matching
GPS collision problem: 104 zones in Uganda where multiple roads share the same
0.01° grid cell. Pure GPS proximity caused A001N2 (Jinja) images to match
C023 (Kampala) segments — a silent mislabelling bug.

Fix: when a filename encodes a road code (e.g. A001N2), restrict matching to
segments on that road first (using 1.5× relaxed distance). Only fall back to
global GPS match when road-constrained match fails.

Filename pattern: `{RoadCode}_{LinkXX}_{Side}PAVE{seq}.jpg`
Example: `A001N2_LINK01_RHSPAVE002334.jpg` → road=A001N2, link=01, side=RHS

### GPS extraction: EXIF first, OCR fallback
- Smartphone images: GPS in EXIF (GPSLatitude, GPSLongitude) → parse directly
- Mobicap images: GPS burned as text in blue header (top 80px) → pytesseract OCR
- Single function `get_gps(path)` returns `(lat, lon, source)` where source
  is `'exif' | 'ocr' | None`
- OCR sometimes drops decimal point: fixed by trying all insertion positions
  and validating against Uganda bounding box (lat -1.5–4.3, lon 29.5–35.1)

### Temporal train/val/test split
Since the same roads are surveyed in multiple years (Jinja: 2021-22, 2023-24,
2025-26; Hoima: 2025-26), splits are assigned per-road by year:
- Most recent year → test
- Second most recent → val
- Earlier years → train
Roads with only one year (like the current 2025-26 Excel) fall back to random
70/15/15 split at the segment level.

### Header crop in all image paths
Mobicap images have an 80px blue text overlay at the top. This is cropped in:
- `dataset.py __getitem__` (training)
- `api/image_utils.py` (inference) — must be applied here too

Smartphone images have no such header — the crop is Mobicap-specific.

---

## Data

### Excel file
`Mobicap_Paved_Network_Combined.xlsx` (sheet: "Mobicap Paved")
- 6,051 rows, one per 1km survey segment
- 136 roads, all surveyed in 2025 (gps1date format: `20250719 16:46`)
- GPS stored in `gps1` / `gps2` columns as `"lat lon"` string pairs
- Short-code column names (header row 1): cral, crwd, sdet, blee, pnpo, etc.
- Full-name column names (header row 0): Cracksall, Crackswide, etc.
- `load_excel()` uses header=0 for defect columns, header=1 for date columns

### Images
Mobicap images: 1288×952px, RGB, no EXIF GPS, blue text header with GPS.
Image density: ~10–30m apart within a segment. Multiple images per 1km segment.

Available datasets:
- Jinja road (A001, A001N1, A001N2): 2021-22, 2023-24, 2025-26
  - 2021-22: 6,973 images  (blue-header Mobicap, GPS via OCR)
  - 2023-24: 42,789 images (no GPS — link-seq matching)
  - 2025-26: 59,492 images (blue-header Mobicap, GPS via OCR)
- Hoima road (A009, A009N1, A009N2): 2021-22, 2023-24, 2025-26
  - 2021-22: 29,596 images (BUSUNJU-LWAMATA-* style names, GPS via OCR)
  - 2023-24: 19,308 images (A009_LinkXX style, no GPS — link-seq matching)
  - 2025-26:  9,077 images (A009LINKXX style no underscore, GPS via OCR)
  - Excel segments (A009/A009N1/A009N2) are already inside the Combined Excels
  - Filename fix: A009LINK02 (no underscore) handled by _? in _FNAME_ROAD regex

Actual image locations (external drive):
  /run/media/mugisa/New Volume/Jinja road/2021-22/
  /run/media/mugisa/New Volume/Jinja road/2023-24/
  /run/media/mugisa/New Volume/Jinja road/2025-26/
  /run/media/mugisa/New Volume/Hoima road/2021-22/
  /run/media/mugisa/New Volume/Hoima road/2023-24/
  /run/media/mugisa/New Volume/Hoima road/2025-26/

### Segment-level baseline (for research comparison)
Ridge regression on defect grades → vVCI: MAE=0, R²=1 (it's a closed-form
formula). The meaningful baseline is image texture features → vVCI, which
gives the lower bound a CNN must beat.

Correlation: Pearson r(VCI, vVCI) = 0.95 on the full dataset.

---

## Project structure

```
vci_estimator/
├── CLAUDE.md                        ← this file
├── README.md                        ← setup and usage guide
├── requirements.txt
├── configs/config.yaml              ← all hyperparameters
├── src/
│   ├── data/
│   │   ├── parse_excel.py           ← load Excel, compute vVCI
│   │   ├── extract_gps.py           ← unified GPS (EXIF + OCR)
│   │   ├── build_dataset.py         ← image→segment matching → dataset.csv
│   │   └── dataset.py               ← PyTorch Dataset + DataLoaders
│   ├── models/
│   │   └── model.py                 ← EfficientNet-B3 + dual heads
│   └── training/
│       ├── losses.py                ← Huber + weighted CE, metrics
│       └── trainer.py               ← two-stage training loop
├── scripts/
│   ├── prepare_data.py              ← STEP 1: build dataset.csv
│   ├── train_model.py               ← STEP 2: train
│   ├── evaluate.py                  ← STEP 3: test set evaluation
│   ├── baseline.py                  ← texture baseline comparison
│   ├── analyse_contributions.py     ← per-defect analysis
│   ├── cross_road_eval.py           ← generalisation test
│   ├── export_model.py              ← TorchScript + ONNX export
│   └── dataset_report.py            ← dataset quality plots
└── outputs/
    ├── dataset.csv                  ← built by prepare_data.py
    ├── checkpoints/best.pt          ← best model checkpoint
    ├── exported/model.torchscript.pt
    ├── exported/model.onnx
    └── exported/model_meta.json
```

---

## Phase completion status

### Phase 1 — Data pipeline ✅ COMPLETE
- [x] 1.1 Road-code-constrained GPS matching (bug fixed)
- [x] 1.2 Smartphone EXIF GPS extractor
- [x] 1.3 Unified get_gps() — EXIF first, OCR fallback
- [x] 1.4 Pothole bin validation (bins [0,1,3,6,11] confirmed)
- [x] 1.5 prepare_data.py tested end-to-end on sample images
- [x] 1.6 dataset_report.py — quality plots
- [x] 1.7 Header crop confirmed in dataset.py and documented for inference

### Phase 2 — Model development ✅ CODE COMPLETE (needs GPU + full dataset to train)
- [x] 2.1 Model verified: 6×(B,5) defect heads, (B,1) vVCI in [0,100]
- [x] 2.2-2.3 Two-stage trainer written and smoke-tested end-to-end (scripts/smoke_test.py)
- [x] 2.4 Hyperparameters in configs/config.yaml
- [x] 2.5 analyse_contributions.py written
- [x] 2.6 baseline.py — formula gap + image texture ridge baseline
- [x] 2.7 evaluate.py — scatter, residuals, confusion matrices + PCI metrics
- [x] 2.8 export_model.py — TorchScript + ONNX + metadata JSON
- [x] 2.9 PCI estimation: src/models/pci_formula.py — ASTM D6433 formula from
        predicted defect grades. Outputs approx PCI [0-100] + 7-level label.
        Integrated into inference.py and evaluate.py.

**To run training on your machine:**
```bash
pip install torch torchvision timm pandas numpy Pillow pytesseract \
            scikit-learn scipy matplotlib pyyaml

# Step 1: build dataset (Jinja + Hoima, all 3 years)
python scripts/prepare_data.py \
    --excel  "/run/media/mugisa/New Volume/Road Condition Data/2021-22/Mobicap Paved Network - Combined  2021-22.xlsx" \
    --excel  "/run/media/mugisa/New Volume/Road Condition Data/2023-24/Final Data Submitted 2024/Mobicap Paved 2023-2024.xlsx" \
    --excel  "/run/media/mugisa/New Volume/Road Condition Data/2025-26/Mobicap Paved Network - Combined  2025-26.xlsx" \
    --images "/run/media/mugisa/New Volume/Jinja road" \
    --images "/run/media/mugisa/New Volume/Hoima road" \
    --output outputs/dataset.csv

# Step 2: train (GPU recommended, ~2h for 40 epochs)
python scripts/train_model.py \
    --dataset    outputs/dataset.csv \
    --device     cuda \
    --epochs     40 \
    --batch-size 32

# Step 3: evaluate
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --dataset    outputs/dataset.csv

# Step 4: export
python scripts/export_model.py --checkpoint outputs/checkpoints/best.pt
```

### Phase 3 — Inference API ✅ COMPLETE
All endpoints implemented and importable. Start with:
```bash
uvicorn api.main:app --reload --port 8000
```
- `GET  /health` — liveness + model_ready flag
- `POST /predict` — 1+ images → vvci, vvci_label, pci, pci_label, defects, gps_used, road_matched
- `POST /predict-batch` — ZIP → per-image results + downloadable CSV
- `GET  /nearest-segment?lat=&lon=&road_code=` — GPS segment lookup

PCI is included in all responses (derived from predicted defect grades via ASTM formula).
Auth: X-API-Key header. Default dev key: `dev-key-change-in-production`. Set `VCI_API_KEY` env var for production.

### Phase 3 original spec (archived)
Build a FastAPI server. Key requirements:

**Endpoints needed:**
- `GET  /health` — liveness check
- `POST /predict` — single or multiple images for one segment
  - Input: multipart/form-data with 1+ images
  - Detects image source (EXIF GPS → smartphone, blue header → Mobicap)
  - Applies Mobicap header crop only for Mobicap images
  - If multiple images: average backbone feature maps, then heads
  - Returns: vvci, vvci_label, defect grades + confidence, gps_used, road_matched
- `POST /predict-batch` — zip of images + optional Excel path for Mobicap archives
  - Runs full pipeline: GPS extract → segment match → predict → CSV results
  - For ministry batch processing use case
- `GET  /nearest-segment?lat=&lon=&road_code=` — GPS to road segment lookup
  - Returns road name, km start/end, existing survey data for that segment
  - Used by web app to show context before/after prediction

**Response schema (define in api/schemas.py):**
```python
{
  "vvci": 84.1,
  "vvci_label": "Good",          # Good(>80) Fair(60-80) Poor(40-60) Bad(<40)
  "defects": [
    {"name": "all_cracking", "predicted_grade": 2, "confidence": 0.87},
    ...
  ],
  "gps_used": {"lat": 0.4457, "lon": 33.199, "source": "ocr"},
  "road_matched": {"name": "A001N2", "km_start": 86.0, "km_end": 87.0}
}
```

**Model loading:** Load TorchScript from `outputs/exported/model.torchscript.pt`
at server startup. Keep in memory for fast inference.

**Auth:** Simple API key via `X-API-Key` header. Rate limit: 60 req/min.

**Structure to build:**
```
api/
  main.py           ← FastAPI app, startup model load
  schemas.py        ← Pydantic request/response models
  inference.py      ← model inference + multi-image aggregation
  image_utils.py    ← source detection, header crop, transform
  auth.py           ← API key validation, rate limiting
  routes/
    predict.py      ← /predict and /predict-batch
    segments.py     ← /nearest-segment
```

### Phase 4 — Web application ✅ COMPLETE (frontend only, requires API running)
Single-page app at `web/index.html`. Open directly in browser or serve with any static file server.

Files:
```
web/
  index.html        ← main app (two-tab: Field Survey + Batch Review)
  css/main.css      ← condition colour bands, defect bars, tab styles
  js/
    api.js          ← fetch wrappers for /predict, /predict-batch, /nearest-segment
    field.js        ← field capture logic: file upload, GPS, results, session log, CSV export
    batch.js        ← batch upload: ZIP → API → results table + summary stats + CSV export
    map.js          ← Leaflet maps (field pin + batch multi-pin, colour-coded by condition)
```

Both vVCI and PCI are displayed in the results panel. Field mode has a persistent session log
that can be exported to CSV at the end of a field day.

To serve locally:
```bash
cd web && python -m http.server 3000
# then open http://localhost:3000
```

### Phase 5 — Evaluation, docs, deployment 🔲 NOT STARTED
- Cross-road generalisation: train on Jinja, test on Hoima (scripts/cross_road_eval.py ready)
- Multi-year consistency: compare predictions across 2021-22 / 2023-24 / 2025-26
- Cost comparison: smartphone vs Mobicap vs manual (UGX 5-10M/100km manual)
- MC Dropout for prediction confidence intervals
- Docker deployment: FastAPI server + web app in docker-compose
- Research paper sections: methodology, results, cost analysis

---

## Important technical notes for future sessions

- **vVCI is normalised to 0-100**: `(raw_sum / 34.5) × 100`
- **Grade encoding in PyTorch**: grades stored as 0-indexed (grade 1 → class 0,
  grade 5 → class 4). Add 1 when displaying to users.
- **Drainage grade oddity**: only values 1, 3, 5 appear in the data. Not a bug.
- **Single survey year**: the 2025-26 Excel has only year 2025. Temporal splits
  require the multi-year image folders (Jinja 2021-22, 2023-24, 2025-26).
- **GPS cache**: each image gets a `.gps.json` sidecar after first extraction.
  Delete these to force re-OCR. Use `--no-cache` flag in prepare_data.py.
- **Batch size guidance** (EfficientNet-B3 @ 224px):
  - batch 32 → ~6GB VRAM
  - batch 16 → ~4GB VRAM
  - batch 8  → ~3GB VRAM (CPU training: use 8)
- **timm model name**: `efficientnet_b3` (all lowercase, underscore)
- **ONNX wrapper**: the export wraps model to return (vvci, stacked_logits)
  since ONNX requires fixed-structure outputs, not dicts or lists.

---

## Contacts / domain context
- Client: Uganda Ministry of Works and Transport (MoWT)
- Survey system: Mobicap (vehicle-mounted, expensive)
- Standard: MoWT Visual Condition Assessment Manual
- Country: Uganda (lat -1.5 to 4.3, lon 29.5 to 35.1)
- Road codes: A001 (Jinja highway), A001N2 (Jinja variant), C023 (Kampala), etc.
