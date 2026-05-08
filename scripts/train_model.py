"""
scripts/train_model.py
----------------------
Step 2 of the pipeline.  Run after prepare_data.py.

Usage (basic)
-------------
python scripts/train_model.py --dataset outputs/dataset.csv

Usage (full options)
--------------------
python scripts/train_model.py \
    --dataset     outputs/dataset.csv \
    --output-dir  outputs/ \
    --epochs      40 \
    --batch-size  32 \
    --freeze-epochs 10 \
    --lr          1e-3 \
    --backbone    efficientnet_b3 \
    --device      cuda

GPU memory guide (batch size vs backbone)
------------------------------------------
efficientnet_b3 @ 224px:
    batch 32  →  ~6 GB VRAM
    batch 16  →  ~4 GB VRAM
    batch 8   →  ~3 GB VRAM
If you only have a CPU, use --device cpu and --batch-size 8.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train the pavement VCI estimator")

    # I/O
    parser.add_argument("--dataset",     required=True,
                        help="Path to dataset.csv from prepare_data.py")
    parser.add_argument("--output-dir",  default="outputs",
                        help="Directory for checkpoints and logs")

    # Model
    parser.add_argument("--backbone",    default="efficientnet_b3",
                        help="timm backbone name (default: efficientnet_b3)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not load ImageNet weights (not recommended)")
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--vvci-hidden",  type=int,   default=256,
                        help="Hidden size of vVCI regression head")
    parser.add_argument("--pci-hidden",   type=int,   default=256,
                        help="Hidden size of PCI regression head")
    parser.add_argument("--pci-pretrain", default=None,
                        help="Path to PCI pre-train checkpoint from pretrain_pci.py")

    # Training
    parser.add_argument("--epochs",          type=int,   default=40)
    parser.add_argument("--freeze-epochs",   type=int,   default=10,
                        help="Epochs to train with frozen backbone (stage 1)")
    parser.add_argument("--unfreeze-blocks", type=int,   default=3,
                        help="Number of top backbone blocks to unfreeze in stage 2")
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--num-workers",     type=int,   default=4)
    parser.add_argument("--image-size",      type=int,   default=224)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1,
                        help="LR multiplier for backbone in stage 2")

    # Loss
    parser.add_argument("--lambda-vvci",   type=float, default=1.0)
    parser.add_argument("--lambda-defect", type=float, default=0.5)
    parser.add_argument("--lambda-pci",    type=float, default=0.5,
                        help="Weight on PCI regression loss (0 to disable)")
    parser.add_argument("--huber-delta",   type=float, default=5.0,
                        help="Huber loss delta (vVCI/PCI units, 0-100 scale)")

    # Misc
    parser.add_argument("--device", default=None,
                        help="'cuda', 'cpu', or 'mps' (auto-detected if omitted)")
    parser.add_argument("--seed",   type=int, default=42)

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        sys.exit(f"Dataset CSV not found: {args.dataset}")

    result = train(
        csv_path             = args.dataset,
        output_dir           = args.output_dir,
        backbone             = args.backbone,
        pretrained           = not args.no_pretrained,
        image_size           = args.image_size,
        batch_size           = args.batch_size,
        num_workers          = args.num_workers,
        epochs               = args.epochs,
        freeze_epochs        = args.freeze_epochs,
        unfreeze_top_blocks  = args.unfreeze_blocks,
        base_lr              = args.lr,
        weight_decay         = args.weight_decay,
        backbone_lr_factor   = args.backbone_lr_factor,
        lambda_vvci          = args.lambda_vvci,
        lambda_defect        = args.lambda_defect,
        lambda_pci           = args.lambda_pci,
        huber_delta          = args.huber_delta,
        dropout              = args.dropout,
        vvci_hidden          = args.vvci_hidden,
        pci_hidden           = args.pci_hidden,
        pci_pretrain         = args.pci_pretrain,
        seed                 = args.seed,
        device               = args.device,
    )

    print("\n=== Final results ===")
    print(f"  Best val MAE   : {result['best_val_mae']:.2f}")
    print(f"  Best epoch     : {result['best_epoch']}")
    print(f"  Checkpoint     : {result['checkpoint_path']}")
    tm = result["test_metrics"]
    print(f"  Test MAE       : {tm['mae_vvci']:.2f}")
    print(f"  Test RMSE      : {tm['rmse_vvci']:.2f}")
    print(f"  Test defect acc: {tm['acc_defect_mean']:.3f}")


if __name__ == "__main__":
    main()
