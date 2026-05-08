"""
losses.py
---------
Combined loss for the multi-task pavement model.

    Total loss = λ_vvci × L_vvci  +  λ_defect × Σ_i L_defect_i

L_vvci   : Huber loss (smooth L1) between predicted and true vVCI.
           Huber is less sensitive to outlier segments than MSE.

L_defect : Cross-entropy loss for each defect grade (ordinal, but treated
           as categorical).  Summed (not averaged) across the 6 defects so
           each defect contributes equally to the gradient regardless of weight.

Defect grade imbalance
----------------------
In the dataset grades skew heavily towards 1 and 2 (good road).  Class weights
are computed per defect from the training split to upweight rare grades.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-defect class weights from dataset
# ---------------------------------------------------------------------------

def compute_class_weights(
    csv_path: str | Path,
    defect_cols: list[str],
    n_grades: int = 5,
    split: str = "train",
) -> list[torch.Tensor]:
    """
    Compute inverse-frequency class weights for each defect column.

    Returns list of (n_grades,) float tensors, one per defect.
    """
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split]
    weights_list = []

    for col in defect_cols:
        counts = np.zeros(n_grades, dtype=float)
        for grade in range(1, n_grades + 1):
            counts[grade - 1] = (df[col] == grade).sum()
        counts = np.clip(counts, 1, None)  # avoid divide-by-zero
        weights = counts.sum() / (n_grades * counts)
        weights_list.append(torch.tensor(weights, dtype=torch.float32))

    return weights_list


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

class PavementLoss(nn.Module):
    """
    Parameters
    ----------
    lambda_vvci   : weight on the vVCI regression loss
    lambda_defect : weight on each defect classification loss
    lambda_pci    : weight on the PCI regression loss (0 disables it)
    class_weights : list of (n_grades,) tensors for each defect, or None
    huber_delta   : delta for Huber loss (in index units, 0-100 scale)
    """

    def __init__(
        self,
        lambda_vvci:   float = 1.0,
        lambda_defect: float = 0.5,
        lambda_pci:    float = 0.5,
        class_weights: list | None = None,
        huber_delta:   float = 5.0,
        n_defects:     int   = 6,
    ):
        super().__init__()
        self.lambda_vvci   = lambda_vvci
        self.lambda_defect = lambda_defect
        self.lambda_pci    = lambda_pci
        self.huber_delta   = huber_delta
        self.n_defects     = n_defects

        # Register class weights as buffers (move to device automatically)
        if class_weights is not None:
            for i, w in enumerate(class_weights):
                self.register_buffer(f"cw_{i}", w)
            self._has_weights = True
        else:
            self._has_weights = False

    def _get_cw(self, i: int) -> torch.Tensor | None:
        if self._has_weights:
            return getattr(self, f"cw_{i}", None)
        return None

    def forward(
        self,
        pred_vvci:     torch.Tensor,              # (B, 1) or (B,)
        true_vvci:     torch.Tensor,              # (B,)
        pred_logits:   list[torch.Tensor],        # list of N_DEFECTS × (B, 5)
        true_grades:   torch.Tensor,              # (B, N_DEFECTS)  0-indexed grades
        pred_pci:      torch.Tensor | None = None,  # (B, 1) or (B,) — optional
        true_pci:      torch.Tensor | None = None,  # (B,) — optional
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with keys: 'total', 'vvci', 'defect', 'pci', and per-defect losses.
        PCI loss is only computed when both pred_pci and true_pci are provided.
        """
        # ── vVCI regression loss ─────────────────────────────────────────
        pred_vvci = pred_vvci.squeeze(1)
        l_vvci = F.huber_loss(pred_vvci, true_vvci, delta=self.huber_delta, reduction="mean")

        # ── Defect classification loss ───────────────────────────────────
        l_defect_total = torch.tensor(0.0, device=pred_vvci.device)
        defect_losses  = {}

        for i, logits in enumerate(pred_logits):
            targets = true_grades[:, i]                  # (B,)
            cw      = self._get_cw(i)
            if cw is not None:
                cw = cw.to(logits.device)
            l_i = F.cross_entropy(logits, targets, weight=cw, reduction="mean")
            defect_losses[f"defect_{i}"] = l_i
            l_defect_total = l_defect_total + l_i

        # ── PCI regression loss (optional) ───────────────────────────────
        l_pci = torch.tensor(0.0, device=pred_vvci.device)
        if pred_pci is not None and true_pci is not None and self.lambda_pci > 0:
            l_pci = F.huber_loss(
                pred_pci.squeeze(1), true_pci.float(),
                delta=self.huber_delta, reduction="mean",
            )

        # ── Combined ─────────────────────────────────────────────────────
        total = (self.lambda_vvci   * l_vvci
               + self.lambda_defect * l_defect_total
               + self.lambda_pci    * l_pci)

        return {
            "total":  total,
            "vvci":   l_vvci,
            "defect": l_defect_total,
            "pci":    l_pci,
            **defect_losses,
        }


# ---------------------------------------------------------------------------
# Evaluation metrics (no grad needed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    pred_vvci:   torch.Tensor,   # (B,) or (B,1)
    true_vvci:   torch.Tensor,   # (B,)
    pred_logits: list[torch.Tensor],
    true_grades: torch.Tensor,   # (B, N_DEFECTS)
) -> dict[str, float]:
    """
    Returns dict with:
        mae_vvci, rmse_vvci,
        acc_defect_mean, acc_defect_<i>   (per-defect accuracy)
    """
    pred_vvci = pred_vvci.squeeze(1).float()
    true_vvci = true_vvci.float()

    mae  = (pred_vvci - true_vvci).abs().mean().item()
    rmse = ((pred_vvci - true_vvci) ** 2).mean().sqrt().item()

    defect_accs = []
    for i, logits in enumerate(pred_logits):
        preds   = logits.argmax(dim=1)
        correct = (preds == true_grades[:, i]).float().mean().item()
        defect_accs.append(correct)

    return {
        "mae_vvci":        mae,
        "rmse_vvci":       rmse,
        "acc_defect_mean": float(np.mean(defect_accs)),
        **{f"acc_defect_{i}": a for i, a in enumerate(defect_accs)},
    }
