"""
model.py
--------
Multi-task pavement condition model.

Architecture
------------
                    ┌─────────────────────────────────┐
  Input image  ──▶  │  EfficientNet-B3 backbone (timm) │  ──▶  feature vector (1536-d)
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌─────────────────┐           ┌──────────────────┐
           │ Defect head     │           │ vVCI head        │
           │ 6 × Linear(5)   │           │ Linear → ReLU    │
           │ (one per defect)│           │ Linear → sigmoid │
           │ outputs: logits │           │ output: 0-100    │
           └─────────────────┘           └──────────────────┘

Training stages
---------------
Stage 1 (epochs 0 … freeze_epochs-1):
    Backbone frozen → only heads trained.
Stage 2 (epochs freeze_epochs … end):
    Last `unfreeze_top_blocks` backbone blocks unfrozen with a lower LR.
"""

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False


# ---------------------------------------------------------------------------
# Constants (mirror config)
# ---------------------------------------------------------------------------

N_DEFECTS = 6   # all_cracking, wide_cracking, ravelling, bleeding, drainage, potholes
N_GRADES  = 5   # grade classes 1-5 → 0-4


# ---------------------------------------------------------------------------
# Defect classification head
# ---------------------------------------------------------------------------

class DefectHead(nn.Module):
    """
    One linear classifier per defect type.
    Input : feature vector of dimension `in_features`
    Output: list of N_DEFECTS logit tensors, each of shape (B, N_GRADES)
    """

    def __init__(self, in_features: int, n_defects: int = N_DEFECTS, n_grades: int = N_GRADES, dropout: float = 0.3):
        super().__init__()
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, n_grades),
            )
            for _ in range(n_defects)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns list of (B, N_GRADES) logit tensors."""
        return [clf(x) for clf in self.classifiers]


# ---------------------------------------------------------------------------
# Shared regression head (used for both vVCI and PCI)
# ---------------------------------------------------------------------------

class RegressionHead(nn.Module):
    """
    Two-layer MLP regressor, sigmoid-scaled to [0, 100].
    Shared architecture for vVCI and PCI heads — trained with separate weights.
    """

    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) tensor in [0, 100]."""
        return self.net(x) * 100.0


# Keep alias so existing code that imports VVCIHead still works
VVCIHead = RegressionHead


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PavementVCIModel(nn.Module):
    """
    Multi-task pavement condition estimator.

    Parameters
    ----------
    backbone     : timm model name (default: 'efficientnet_b3')
    pretrained   : load ImageNet weights
    dropout      : dropout probability in heads
    vvci_hidden  : hidden size of vVCI regression MLP
    """

    def __init__(
        self,
        backbone:    str   = "efficientnet_b3",
        pretrained:  bool  = True,
        dropout:     float = 0.3,
        vvci_hidden: int   = 256,
        pci_hidden:  int   = 256,
        n_defects:   int   = N_DEFECTS,
        n_grades:    int   = N_GRADES,
    ):
        super().__init__()

        if not _TIMM_OK:
            raise ImportError("timm is required: pip install timm")

        # ── Backbone ────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # remove classifier
            global_pool="avg",      # global average pool → feature vector
        )
        in_features = self.backbone.num_features  # 1536 for efficientnet_b3

        # ── Heads ────────────────────────────────────────────────────────
        self.defect_head = DefectHead(in_features, n_defects, n_grades, dropout)
        self.vvci_head   = RegressionHead(in_features, vvci_hidden, dropout)
        self.pci_head    = RegressionHead(in_features, pci_hidden,  dropout)

        # Track which blocks exist for selective unfreezing
        self._backbone_name = backbone

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list]:
        """
        Parameters
        ----------
        x : (B, 3, H, W) image tensor

        Returns
        -------
        dict with:
            'defect_logits' : list of N_DEFECTS tensors, each (B, N_GRADES)
            'vvci'          : (B, 1) tensor in [0, 100]
        """
        feats = self.backbone(x)
        return {
            "defect_logits": self.defect_head(feats),
            "vvci":          self.vvci_head(feats),
            "pci":           self.pci_head(feats),
        }

    # ------------------------------------------------------------------
    # Staged freezing / unfreezing
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Stage 1: freeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_top_blocks(self, n_blocks: int = 3) -> None:
        """
        Stage 2: unfreeze the last `n_blocks` blocks of the backbone.
        Works for EfficientNet-style models from timm (which expose `.blocks`).
        Falls back to unfreezing the entire backbone if `.blocks` is absent.
        """
        # First unfreeze the classifier norm / head layers (always present)
        for name in ["bn2", "conv_head", "classifier"]:
            module = getattr(self.backbone, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(True)

        if hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks  # list-like of Sequential blocks
            n = len(blocks)
            for block in blocks[max(0, n - n_blocks):]:
                for p in block.parameters():
                    p.requires_grad_(True)
        else:
            # Generic fallback
            for p in self.backbone.parameters():
                p.requires_grad_(True)

    def get_parameter_groups(self, backbone_lr_factor: float = 0.1, base_lr: float = 1e-3) -> list[dict]:
        """
        Return parameter groups for the optimiser with two LR levels.
        Call this after unfreeze_top_blocks() in stage 2.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params     = (
            list(self.defect_head.parameters())
            + list(self.vvci_head.parameters())
            + list(self.pci_head.parameters())
        )
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_factor},
            {"params": head_params,     "lr": base_lr},
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_vvci(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns (B,) vVCI tensor."""
        out = self.forward(x)
        return out["vvci"].squeeze(1)


# ---------------------------------------------------------------------------
# Heads-only model (for Colab feature-based training)
# ---------------------------------------------------------------------------

class HeadsModel(nn.Module):
    """
    The three task heads without the backbone.
    Used when training on pre-extracted feature vectors (train_features.py).
    The backbone is loaded separately (frozen timm EfficientNet-B3) for inference.
    """

    def __init__(self, feat_dim: int = 1536, n_defects: int = N_DEFECTS,
                 n_grades: int = N_GRADES, vvci_hidden: int = 256,
                 pci_hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.defect_head = DefectHead(feat_dim, n_defects, n_grades, dropout)
        self.vvci_head   = RegressionHead(feat_dim, vvci_hidden, dropout)
        self.pci_head    = RegressionHead(feat_dim, pci_hidden,  dropout)

    def forward(self, x: torch.Tensor) -> dict:
        return {
            "defect_logits": self.defect_head(x),
            "vvci":          self.vvci_head(x),
            "pci":           self.pci_head(x),
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = PavementVCIModel(pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("defect_logits:", [t.shape for t in out["defect_logits"]])
    print("vvci shape   :", out["vvci"].shape)
    print("vvci values  :", out["vvci"].squeeze())

    print("\n--- Stage 1: freeze backbone ---")
    model.freeze_backbone()
    print("Trainable params:", model.n_trainable_params)

    print("\n--- Stage 2: unfreeze top 3 blocks ---")
    model.unfreeze_top_blocks(n_blocks=3)
    print("Trainable params:", model.n_trainable_params)
