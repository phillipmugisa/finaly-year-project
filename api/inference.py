"""
api/inference.py
----------------
Model loading and inference.

Supports two checkpoint formats:
  1. Full model (PavementVCIModel) — backbone + heads in one checkpoint.
     Produced by train_model.py (local GPU training).
  2. Feature-based (HeadsModel) — heads only, backbone loaded from timm.
     Produced by train_features.py (Colab training on pre-extracted features).
     Detected by checkpoint key "feature_based": True.

Model search order:
  1. outputs/checkpoints/best.pt   (PyTorch checkpoint — preferred)
  2. outputs/exported/model.torchscript.pt  (TorchScript fallback)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import DEFECT_COLS, N_GRADES
from src.models.pci_formula import grades_tensor_to_pci

_ROOT         = Path(__file__).resolve().parent.parent
_CKPT_PATH    = _ROOT / "outputs" / "checkpoints" / "best.pt"
_TS_PATH      = _ROOT / "outputs" / "exported"    / "model.torchscript.pt"

DEFECT_NAMES  = [c.replace("_grade", "") for c in DEFECT_COLS]
N_DEFECTS     = len(DEFECT_COLS)

_model    = None   # PavementVCIModel  OR  HeadsModel
_backbone = None   # frozen EfficientNet-B3 (only set when _model is HeadsModel)
_ts_model = None   # TorchScript fallback
_device   = "cpu"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_model() -> bool:
    """
    Load model at server startup.  Returns True if a model was found.
    Detects feature-based checkpoints automatically.
    """
    global _model, _backbone, _ts_model, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    if _CKPT_PATH.exists():
        try:
            ckpt = torch.load(_CKPT_PATH, map_location=_device)

            if ckpt.get("feature_based"):
                # ── Colab-trained heads-only checkpoint ──────────────────
                from src.models.model import HeadsModel
                import timm
                _model = HeadsModel().to(_device)
                _model.load_state_dict(ckpt["model"])
                _model.eval()

                # Load frozen backbone separately for feature extraction
                _backbone = timm.create_model(
                    "efficientnet_b3", pretrained=True,
                    num_classes=0, global_pool="avg",
                )
                _backbone.eval().to(_device)
                for p in _backbone.parameters():
                    p.requires_grad_(False)
                print(f"  Loaded feature-based checkpoint: {_CKPT_PATH}")
            else:
                # ── Full model checkpoint ─────────────────────────────────
                from src.models.model import PavementVCIModel
                _model = PavementVCIModel(pretrained=False).to(_device)
                state  = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
                _model.load_state_dict(state)
                _model.eval()
                print(f"  Loaded full checkpoint: {_CKPT_PATH}")

            return True
        except Exception as e:
            print(f"  WARNING: checkpoint load failed ({e}), trying TorchScript …")
            _model = _backbone = None

    if _TS_PATH.exists():
        try:
            _ts_model = torch.jit.load(str(_TS_PATH), map_location=_device)
            _ts_model.eval()
            print(f"  Loaded TorchScript: {_TS_PATH}")
            return True
        except Exception as e:
            print(f"  WARNING: TorchScript load failed ({e})")

    return False


def is_ready() -> bool:
    return _model is not None or _ts_model is not None


# ---------------------------------------------------------------------------
# vVCI label
# ---------------------------------------------------------------------------

def vvci_label(vvci: float) -> str:
    if vvci > 80: return "Good"
    if vvci > 60: return "Fair"
    if vvci > 40: return "Poor"
    return "Bad"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(img_tensors: list) -> dict:
    """
    Run inference on one or more preprocessed image tensors.

    Multi-image strategy: average backbone feature vectors, then pass
    through the heads once.  Falls back to averaging final predictions
    when using TorchScript (which doesn't expose backbone features).

    Parameters
    ----------
    img_tensors : list of (1, 3, 224, 224) float tensors

    Returns
    -------
    dict with keys: model_ready, vvci, vvci_label, defects
    """
    if not is_ready():
        return {"model_ready": False, "vvci": None, "vvci_label": None, "defects": []}

    with torch.no_grad():
        tensors = [t.to(_device) for t in img_tensors]

        # ── Full PyTorch model — feature-level averaging ──────────────────
        if _model is not None:
            # Feature-based model: extract features via separate frozen backbone
            bb = _backbone if _backbone is not None else _model.backbone
            feats = torch.cat(
                [bb(t) for t in tensors], dim=0
            ).mean(dim=0, keepdim=True)              # (1, 1536)

            defect_logits = _model.defect_head(feats)  # list of 6 × (1, 5)
            vvci_val      = float(_model.vvci_head(feats).squeeze())
            pci_head_raw  = float(_model.pci_head(feats).squeeze())

        # ── TorchScript fallback — prediction-level averaging ─────────────
        else:
            pci_head_raw = None
            vvcis, all_logits = [], []
            for t in tensors:
                ts_out = _ts_model(t)
                # TorchScript export returns (vvci_tensor, stacked_logits)
                vvcis.append(float(ts_out[0].squeeze()))
                all_logits.append(ts_out[1])           # (1, 6, 5)
            vvci_val      = sum(vvcis) / len(vvcis)
            stacked       = torch.stack(all_logits).mean(0)  # (1, 6, 5)
            defect_logits = [stacked[:, i, :] for i in range(N_DEFECTS)]

        # ── Build defect predictions ──────────────────────────────────────
        defects    = []
        grade_idxs = []
        for i, name in enumerate(DEFECT_NAMES):
            logits     = defect_logits[i]          # (1, 5)
            probs      = F.softmax(logits, dim=-1).squeeze()  # (5,)
            grade_idx  = int(probs.argmax().item())
            confidence = float(probs[grade_idx].item())
            grade_idxs.append(grade_idx)
            defects.append({
                "name":            name,
                "predicted_grade": grade_idx + 1,  # 0-4 → 1-5
                "confidence":      round(confidence, 4),
                "grade_probs":     [round(float(p), 4) for p in probs.tolist()],
            })

        # ── PCI: use trained head if value is in range, else formula ──────
        if pci_head_raw is not None and 0 <= pci_head_raw <= 100:
            pci_val = round(pci_head_raw, 1)
            pci_lbl = _pci_label_from_formula(pci_head_raw)
        else:
            # Formula fallback (always available, derived from predicted grades)
            pci_val, pci_lbl = grades_tensor_to_pci(grade_idxs)

    return {
        "model_ready": True,
        "vvci":        round(vvci_val, 2),
        "vvci_label":  vvci_label(vvci_val),
        "pci":         pci_val,
        "pci_label":   pci_lbl,
        "defects":     defects,
    }


def _pci_label_from_formula(pci: float) -> str:
    from src.models.pci_formula import pci_label
    return pci_label(pci)
