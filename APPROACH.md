# Automated Pavement Condition Assessment — Approach & Methodology

**Authors:** Mugisa Phillip & Ater Maluac Ater  
**Client:** Uganda Ministry of Works & Transport (MoWT)  
**Dataset:** Mobicap vehicle-mounted survey images — Jinja (A001) + Hoima (A009), 2021–2026

---

## 1. Problem Statement

Uganda's MoWT surveys paved roads every two years using a Mobicap vehicle that captures GPS-tagged images and records defect grades in Excel. The rating step — where engineers visually assign grades 1–5 to each defect type — is expensive and slow. This project automates that step.

Two user groups:

| User | Input | Goal |
|------|-------|------|
| MoWT (Mobicap) | GPS-tagged images with blue header overlay | Replace manual rating; output vVCI + PCI per segment |
| Field engineer | Smartphone photos, GPS from EXIF | Low-cost survey without Mobicap equipment |

---

## 2. Target Metric — Visual VCI (vVCI)

The ministry uses **VCI (Visual Condition Index)**, a weighted sum of defect grades (scale 0–100, higher = better). Full VCI includes rutting, roughness (IRI), and structural capacity (FWD) — all measured by instruments, not visible in images.

This project predicts **vVCI** — VCI computed from image-observable defects only:

| Defect | VCI weight |
|--------|-----------|
| All cracking | 7% |
| Wide cracking | 10% |
| Ravelling / Disintegration | 5% |
| Bleeding | 2.5% |
| Drainage on road | 2.5% |
| Potholes / Failures | 7.5% |

**Total visual weight = 34.5%** of full VCI.

Formula:
```
contribution_i = weight_i × (5 − grade_i) / 4
vVCI_raw       = Σ contributions
vVCI           = (vVCI_raw / 34.5) × 100     ← normalised to [0, 100]
```

PCI (ASTM D6433) is a secondary output derived analytically from predicted defect grades. The PCI regression head will be pre-trained on the RDD2022 international road damage dataset (Colab) before Uganda training.

---

## 3. Data Pipeline

### 3.1 Sources

- **Excel:** `Mobicap_Paved_Network_Combined.xlsx` — 6,051 rows, one per 1 km segment. Contains defect grades, GPS, and road codes.
- **Images:** ~167,000 images across Jinja + Hoima, three survey years (2021-22, 2023-24, 2025-26).
  - Mobicap images: 1288×952 px, GPS burned in blue 80 px header — extracted via pytesseract OCR.
  - Smartphone images: GPS in EXIF metadata.

### 3.2 GPS Extraction

```
get_gps(image_path) → (lat, lon, source)
  source = 'exif'   — smartphone image, GPS in EXIF
  source = 'ocr'    — Mobicap image, GPS parsed from blue header
  source = None     — no GPS available, use link-sequence matching
```

OCR post-processing fixes a common decimal-point drop by trying all insertion positions and validating against Uganda's bounding box (lat −1.5–4.3, lon 29.5–35.1).

### 3.3 Image-to-Segment Matching

Each image is matched to its 1 km Excel segment using GPS proximity. A key bug was fixed: 104 grid cells in Uganda contain roads from multiple routes. Pure GPS proximity caused Jinja (A001N2) images to silently match Kampala (C023) segments.

**Fix:** when a filename encodes a road code (e.g. `A001N2_LINK01_RHSPAVE002334.jpg`), restrict matching to segments on that road first using 1.5× relaxed distance. Only fall back to global GPS match on failure.

### 3.4 Dataset

After matching, `dataset.csv` contains 132,544 rows — one per image — with columns:
`image_path`, `road_code`, `segment_start`, `segment_end`, `survey_year`, `vvci`, `all_cracking_grade`, `wide_cracking_grade`, `ravelling_grade`, `bleeding_grade`, `drainage_road_grade`, `pothole_grade`, `split`.

**Potholes** are a raw count (0, 1, 2, 25 ...) in the Excel. Binned to 1–5 using thresholds `[0, 1, 3, 6, 11]` before training.

**Drainage** uses only odd values (1, 3, 5) — a MoWT data entry convention, not a bug.

---

## 4. Model Architecture

### 4.1 Backbone

**EfficientNet-B3** (pretrained on ImageNet, from `timm`).  
- Input: 224×224 RGB (Mobicap header cropped before resize)
- Output: 1536-d feature vector via Global Average Pooling

### 4.2 Task Heads

Three heads attached to the shared feature vector:

#### Defect Head (updated)
Shared bottleneck + six independent classifiers:
```
Dropout(0.3)
Linear(1536 → 256)
BatchNorm1d(256) + ReLU
── per defect (×6) ──
  Dropout(0.15)
  Linear(256 → 5)          ← grade logits [0–4], displayed as [1–5]
```
The shared bottleneck allows correlated defects (e.g. cracking and ravelling co-occur) to share a representation before branching. **401,694 parameters.**

#### vVCI Head
```
Dropout(0.3)
Linear(1536 → 256)
BatchNorm1d(256) + ReLU
Dropout(0.15)
Linear(256 → 1)
Sigmoid × 100              ← output in [0, 100]
```
**394,497 parameters.**

#### PCI Head
Identical architecture to vVCI head, separate weights.  
Pre-trained on RDD2022 (~47,000 Japanese road damage images, 13.3 GB FigShare) using ASTM D6433 formula-derived PCI targets. Pre-training runs on Google Colab T4 — no local GPU required.  
**394,497 parameters.**

**Total trainable (heads only): ~1.19M parameters.**

---

## 5. Training Strategy

### 5.1 Loss Function

```
L_total = λ_vvci   × Huber(pred_vvci,   true_vvci)       [λ = 1.0]
        + λ_defect × Σ_i CE(pred_grade_i, true_grade_i)  [λ = 0.5]
        + λ_pci    × Huber(pred_pci,    true_pci)         [λ = 0.5]
        + λ_consist × Huber(pred_vvci, formula_vvci)      [λ = 0.3]
```

**Defect class weights:** inverse-frequency per defect, computed from the training split. Upweights rare grades (4, 5) to counter the heavy skew towards grades 1–2 on Uganda's generally good roads.

**Consistency loss** (new): `formula_vvci` is computed by passing the defect head's soft grade predictions (softmax probabilities × expected grade values) through the MoWT vVCI formula. This enforces that both heads stay in agreement with the known formula and prevents them from diverging during training.

### 5.2 Feature-Based Training (current, CPU)

Running end-to-end training on 132k images on CPU would take days. Instead:

1. **Extract features once:** run frozen EfficientNet-B3 over all 132,544 images → 1536-d vectors → `outputs/features.npz` (~400 MB, float16).
2. **Aggregate by segment:** mean-pool all image feature vectors belonging to the same 1 km survey segment. This is the key alignment step — labels (vVCI, grades) are assigned per segment, so training per image repeats the same label for every image in the segment (30–100 images per km), creating noise and redundancy.
3. **Train heads only:** `scripts/train_features.py` trains the 1.19M-parameter HeadsModel on 755 segment-level vectors. One epoch takes ~1 second on CPU.

**Segment split (random 70/15/15):**

| Split | Segments | Note |
|-------|----------|------|
| Train | 408 | All years contribute |
| Val   | 87  | Early stopping |
| Test  | 89  | Held out |

A random split (rather than temporal) is used because the temporal split assigned the largest year (2025-26) to test, leaving only 231 training segments. The random split gives 77% more training data and improves defect accuracy.

### 5.3 End-to-End Training (planned, Colab T4)

For full backbone fine-tuning:

- **Stage 1 (epochs 0–9):** backbone frozen, only heads trained. Fast convergence.
- **Stage 2 (epochs 10–39):** top 3 EfficientNet blocks unfrozen at LR × 0.1. Fine-tunes high-level pavement-specific features while keeping lower ImageNet representations intact.

PCI head is initialised from RDD2022 pre-training weights before Stage 2.

---

## 6. Accuracy Results

| Metric | Value |
|--------|-------|
| **Test MAE (vVCI)** | **16.6 points** (on 0–100 scale) |
| **Test RMSE (vVCI)** | 19.7 points |
| **Defect accuracy (mean)** | **44.6%** (5-class; random = 20%) |
| Best val MAE | 14.9 points |

Per-defect accuracy:

| Defect | Accuracy |
|--------|----------|
| Potholes / Failures | ~85% |
| Ravelling | ~39% |
| All cracking | ~18% |
| Wide cracking | ~21% |
| Bleeding | ~30% |
| Drainage on road | ~10% |

**Key limitation:** only 755 unique survey segments (Jinja + Hoima roads). More roads would significantly reduce MAE. End-to-end Colab training on raw images is expected to improve results further.

---

## 7. Inference API

FastAPI server (`api/main.py`). Loaded at startup: TorchScript model from `outputs/exported/model.torchscript.pt`.

| Endpoint | Description |
|----------|-------------|
| `GET  /health` | Liveness + `model_ready` flag |
| `POST /predict` | 1+ images → vVCI, vVCI label, PCI, PCI label, defect grades, GPS, road segment |
| `POST /predict-batch` | ZIP archive → per-image results + downloadable CSV |
| `GET  /nearest-segment` | `?lat=&lon=&road_code=` → nearest Excel segment info |
| `GET  /monitor/status` | Live pipeline + training progress |

**Auth:** `X-API-Key` header. Default dev key: `dev-key-change-in-production`. Set `VCI_API_KEY` env var for production.

**Multi-image inference:** feature maps from multiple images of the same segment are averaged before passing to heads. This matches the training-time segment aggregation.

---

## 8. Web Application

Single-page app at `web/index.html`. Four tabs:

| Tab | Description |
|-----|-------------|
| **Field Survey** | Drag-drop or live camera upload, GPS auto-detect, results with defect bars, session log, CSV export, Leaflet map |
| **Batch Review** | ZIP upload for Mobicap archives, results table, summary stats, CSV export, multi-pin map |
| **Monitor** | Live training/pipeline status from `/monitor/status`, loss + MAE charts |
| **Visualizer** | Interactive project phase tracker + CNN architecture diagram with forward-pass animation |

---

## 9. Pending: RDD2022 PCI Pre-training (Colab)

The PCI regression head currently trains from scratch on Uganda data only, using formula-derived PCI labels. Pre-training on RDD2022 — 47,000 international road damage images with expert ratings — would provide a much stronger PCI initialisation.

**Steps (run on Google Colab with T4 GPU):**
```bash
# 1. Download RDD2022 (13.3 GB, ~15 min on Colab)
python scripts/download_rdd2022.py --output data/rdd2022

# 2. Pre-train PCI head on RDD2022
python scripts/pretrain_pci.py \
    --data data/rdd2022 --device cuda \
    --epochs-stage1 10 --epochs-stage2 10 --batch-size 32

# 3. Train full model with pre-trained PCI head
python scripts/train_features.py \
    --features  outputs/features.npz \
    --dataset   outputs/dataset.csv \
    --pci-pretrain outputs/pci_pretrain/pci_head.pt \
    --device cuda --epochs 60 --resplit
```

All infrastructure (`scripts/download_rdd2022.py`, `scripts/pretrain_pci.py`, `src/data/rdd2022_dataset.py`) is complete. Only requires a machine with GPU + fast internet.

---

## 10. Project Structure

```
vci_estimator/
├── CLAUDE.md                    ← session context + key decisions
├── APPROACH.md                  ← this file
├── colab_train.ipynb            ← Colab notebook for GPU training
├── configs/config.yaml          ← all hyperparameters
├── src/
│   ├── data/
│   │   ├── parse_excel.py       ← load Excel, compute vVCI
│   │   ├── extract_gps.py       ← EXIF + OCR GPS unified extractor
│   │   ├── build_dataset.py     ← image→segment matching → dataset.csv
│   │   └── dataset.py           ← PyTorch Dataset + DataLoaders
│   ├── models/
│   │   ├── model.py             ← EfficientNet-B3 + DefectHead + regression heads
│   │   └── pci_formula.py       ← ASTM D6433 PCI from defect grades
│   └── training/
│       ├── losses.py            ← Huber + weighted CE + consistency loss
│       └── trainer.py           ← two-stage training loop
├── scripts/
│   ├── prepare_data.py          ← STEP 1: build dataset.csv
│   ├── extract_features.py      ← STEP 2: extract backbone features (CPU)
│   ├── train_features.py        ← STEP 3: train heads on features (CPU/Colab)
│   ├── train_model.py           ← STEP 3 alt: end-to-end (GPU only)
│   ├── evaluate.py              ← STEP 4: test set metrics + plots
│   ├── download_rdd2022.py      ← RDD2022 downloader (Colab)
│   └── pretrain_pci.py          ← PCI head pre-training on RDD2022
├── api/
│   ├── main.py                  ← FastAPI app
│   ├── inference.py             ← model inference + multi-image aggregation
│   └── routes/
│       ├── predict.py           ← /predict + /predict-batch
│       ├── segments.py          ← /nearest-segment
│       └── monitor.py           ← /monitor/status
├── web/
│   ├── index.html               ← single-page app
│   ├── css/main.css
│   └── js/
│       ├── field.js             ← field survey logic
│       ├── batch.js             ← batch upload logic
│       ├── monitor.js           ← training monitor
│       └── visualizer.js        ← project + model architecture viz
└── outputs/
    ├── dataset.csv              ← 132,544 image rows
    ├── features.npz             ← 132,544 × 1536 feature vectors
    └── checkpoints/best.pt      ← best trained checkpoint
```
