from __future__ import annotations

import torch
import torch.nn as nn

from .standardizer import FeatureStandardiser


class LayerWeightedPooledHead(nn.Module):
    """
    Layer-weighted classification head over pre-extracted pooled stats.

    Architecture:
        FeatureStandardiser (per-position z-score, fitted on train cache)
        -> softmax-weighted sum over backbone layers
        -> Linear + BatchNorm + GELU + Dropout (x2)
        -> Linear classifier

    The standardiser is the key to making training actually work: raw pooled
    stats have per-position stds spanning ~4 orders of magnitude (mean blocks
    dominate skew blocks by 1000x), which collapses the MLP to majority-class
    before the layer weights ever get useful gradient signal. Fit with
    head.scaler.fit(train_loader) before training.

    Input:  [B, n_layers, stat_dim]
    Output: (logits [B, n_classes], embedding [B, proj_dim])
    """

    def __init__(
        self,
        n_layers: int,
        stat_dim: int,
        proj_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.stat_dim = stat_dim
        self.proj_dim = proj_dim

        self.scaler = FeatureStandardiser(n_layers, stat_dim)
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

        self.proj = nn.Sequential(
            nn.Linear(stat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(proj_dim, n_classes)

    def layer_softmax(self) -> torch.Tensor:
        return torch.softmax(self.layer_weights, dim=0)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.scaler(pooled.to(torch.float32))
        w = self.layer_softmax().view(1, -1, 1)
        fused = (x * w).sum(dim=1)
        z = self.proj(fused)
        logits = self.classifier(z)
        return logits, z

    def param_groups(self, base_lr: float = 1e-3) -> list[dict]:
        """s3prl convention: layer weights get 0.1 x the head LR. Scaler
        buffers are not learnable and excluded automatically."""
        head_params = list(self.proj.parameters()) + list(self.classifier.parameters())
        return [
            {"params": [self.layer_weights], "lr": base_lr * 0.1, "name": "layer_weights"},
            {"params": head_params, "lr": base_lr, "name": "head"},
        ]
