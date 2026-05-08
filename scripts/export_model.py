"""
scripts/export_model.py
-----------------------
Export trained model to TorchScript and ONNX for deployment.

TorchScript → used by the FastAPI inference server (full Python env).
ONNX        → used by ONNX Runtime (CPU-only, lightweight, web/edge).

Usage
-----
python scripts/export_model.py \
    --checkpoint outputs/checkpoints/best.pt \
    --output-dir outputs/exported
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    import torch
    from src.models.model import PavementVCIModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/exported")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg  = ckpt.get("config", {})
    model = PavementVCIModel(
        backbone=cfg.get("backbone", "efficientnet_b3"),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded from epoch {ckpt['epoch']}")

    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    # ── TorchScript ──────────────────────────────────────────────────────
    try:
        # Use torch.jit.trace (more compatible than script for timm models)
        traced = torch.jit.trace(model, dummy, strict=False)
        ts_path = out_dir / "model.torchscript.pt"
        traced.save(str(ts_path))
        # Verify
        loaded = torch.jit.load(str(ts_path))
        out    = loaded(dummy)
        print(f"TorchScript saved → {ts_path}")
        print(f"  Verification: vvci shape {out['vvci'].shape}, range [{out['vvci'].min():.1f}, {out['vvci'].max():.1f}]")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # ── ONNX ─────────────────────────────────────────────────────────────
    try:
        onnx_path = out_dir / "model.onnx"
        # ONNX export: wrap model to return a single tensor (vvci only)
        # for a simple export; defect logits can be added as needed
        class _VCIOnly(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                out = self.m(x)
                stacked = torch.stack(out["defect_logits"], dim=1)  # (B, 6, 5)
                return out["vvci"], stacked

        wrapper = _VCIOnly(model)
        wrapper.eval()
        torch.onnx.export(
            wrapper, dummy, str(onnx_path),
            input_names=["image"],
            output_names=["vvci", "defect_logits"],
            dynamic_axes={"image": {0: "batch"}, "vvci": {0: "batch"}, "defect_logits": {0: "batch"}},
            opset_version=17,
        )
        print(f"ONNX saved → {onnx_path}")

        # Verify with onnxruntime if available
        try:
            import onnxruntime as ort
            sess   = ort.InferenceSession(str(onnx_path))
            result = sess.run(None, {"image": dummy.numpy()})
            print(f"  ONNX verification: vvci={result[0]}, defect_logits shape={result[1].shape}")
        except ImportError:
            print("  (onnxruntime not installed — skipping verification)")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    # ── Save metadata ─────────────────────────────────────────────────────
    import json
    meta = {
        "backbone":   cfg.get("backbone", "efficientnet_b3"),
        "image_size": args.image_size,
        "n_defects":  6,
        "n_grades":   5,
        "defect_names": ["all_cracking","wide_cracking","ravelling","bleeding","drainage","potholes"],
        "vvci_range": [0, 100],
        "epoch":      int(ckpt["epoch"]),
        "val_mae":    float(ckpt["val_mae"]),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved → {out_dir}/model_meta.json")

if __name__ == "__main__":
    main()
