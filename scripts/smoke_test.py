"""
scripts/smoke_test.py
---------------------
Quick structural test of the full pipeline — no real images needed.

Creates a tiny synthetic dataset.csv with fake image paths, patches the
Dataset to return random tensors instead of loading files, and runs 2
training epochs + evaluation to confirm nothing crashes.

Usage
-----
python scripts/smoke_test.py

Should complete in under 60s on CPU.
"""

import sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

# ── 1. Build a synthetic dataset.csv ─────────────────────────────────────────

N = 40   # rows

rng = np.random.default_rng(0)
grades = rng.integers(1, 6, size=(N, 6))   # 6 defect grades, 1-5

# vVCI formula
weights = np.array([7, 10, 5, 2.5, 2.5, 7.5])
raw     = (weights * (5 - grades) / 4).sum(axis=1)
vvci    = raw / 34.5 * 100

splits  = (["train"] * 28) + (["val"] * 6) + (["test"] * 6)

df = pd.DataFrame({
    "image_path":          [f"/tmp/fake_img_{i:04d}.jpg" for i in range(N)],
    "road_code":           ["A001N2"] * N,
    "road_name":           ["A001N2"] * N,
    "segment_start":       rng.uniform(0, 100, N),
    "segment_end":         rng.uniform(0, 100, N),
    "lat_centroid":        rng.uniform(0, 2, N),
    "lon_centroid":        rng.uniform(32, 34, N),
    "survey_year":         [2025] * N,
    "region":              ["Eastern"] * N,
    "station":             ["Jinja"] * N,
    "vci":                 vvci,
    "vvci":                vvci,
    "all_cracking_grade":  grades[:, 0],
    "wide_cracking_grade": grades[:, 1],
    "ravelling_grade":     grades[:, 2],
    "bleeding_grade":      grades[:, 3],
    "drainage_road_grade": grades[:, 4],
    "pothole_grade":       grades[:, 5],
    "split":               splits,
})

tmp_dir = Path(tempfile.mkdtemp())
csv_path = tmp_dir / "smoke_dataset.csv"
df.to_csv(csv_path, index=False)
print(f"[smoke] synthetic dataset: {N} rows → {csv_path}")

# ── 2. Patch Dataset.__getitem__ to skip disk reads ───────────────────────────

from src.data.dataset import PavementDataset, DEFECT_COLS
import torchvision.transforms as T

_fake_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _fake_getitem(self, idx):
    row    = self.df.iloc[idx]
    image  = torch.randn(3, 224, 224)   # synthetic
    grades = torch.tensor(
        [int(row[c]) - 1 for c in DEFECT_COLS], dtype=torch.long
    )
    vvci   = torch.tensor(float(row["vvci"]), dtype=torch.float32)
    return {
        "image":      image,
        "grades":     grades,
        "vvci":       vvci,
        "image_path": str(row["image_path"]),
    }

PavementDataset.__getitem__ = _fake_getitem

# ── 3. Run training (2 epochs, tiny batch) ────────────────────────────────────

from src.training.trainer import train

print("[smoke] running 2-epoch training …")
result = train(
    csv_path      = csv_path,
    output_dir    = tmp_dir,
    backbone      = "efficientnet_b3",
    pretrained    = False,       # no download
    image_size    = 224,
    batch_size    = 8,
    num_workers   = 0,
    epochs        = 2,
    freeze_epochs = 1,
    base_lr       = 1e-3,
    device        = "cpu",
    seed          = 0,
)

print(f"[smoke] best_val_mae={result['best_val_mae']:.2f}  best_epoch={result['best_epoch']}")

# ── 4. PCI formula test ───────────────────────────────────────────────────────

from src.models.pci_formula import grade_to_pci, pci_label, add_pci_to_dataframe

df_test = pd.read_csv(csv_path)
add_pci_to_dataframe(df_test)
print(f"[smoke] PCI computed for {len(df_test)} rows — "
      f"mean={df_test['pci'].mean():.1f}, labels={df_test['pci_label'].value_counts().to_dict()}")

# ── 5. Inference (no model file needed — use freshly trained) ─────────────────

from src.models.model import PavementVCIModel
from src.models.pci_formula import grades_tensor_to_pci
import torch.nn.functional as F

model = PavementVCIModel(pretrained=False)
ckpt  = torch.load(result["checkpoint_path"], map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out          = model(x)
    vvci_val     = float(out["vvci"].squeeze())
    grade_idxs   = [int(F.softmax(lg, dim=-1).argmax().item()) for lg in out["defect_logits"]]
    pci_val, lbl = grades_tensor_to_pci(grade_idxs)

print(f"[smoke] inference: vVCI={vvci_val:.1f}  PCI={pci_val}  ({lbl})")

# ── Cleanup ───────────────────────────────────────────────────────────────────

shutil.rmtree(tmp_dir)
print("\n✓ Smoke test passed — pipeline is structurally correct.")
