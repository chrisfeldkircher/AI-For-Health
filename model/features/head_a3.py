"""
A3 manner-aware head: A2 pooled-stats stream + manner-pooled stream + indicator.

Two layer-weighted streams concatenated with a 3-d category-presence indicator,
then a single MLP. Each stream has its own per-position standardiser; the manner
standardiser is fit only over non-empty buckets via the indicator so zero-filled
empty categories don't deflate per-position stds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .standardizer import FeatureStandardiser


class MannerStandardiser(nn.Module):
    """Per-position z-score over [B, n_layers, n_cats, dim], fit using the
    indicator so empty buckets (zero-filled) are excluded from the running mean
    and std. Persists with the model via buffers."""

    def __init__(self, n_layers: int, n_cats: int, dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.n_cats = n_cats
        self.dim = dim
        self.register_buffer("mean", torch.zeros(n_layers, n_cats, dim))
        self.register_buffer("std", torch.ones(n_layers, n_cats, dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, loader: DataLoader, verbose: bool = True) -> None:
        # Per-category counts because empty buckets are excluded; per-(L, c, d)
        # would just multiply work by L*d for no gain (indicator is per-cat).
        n_per_cat = torch.zeros(self.n_cats, dtype=torch.float64)
        sum_x = torch.zeros(self.n_layers, self.n_cats, self.dim, dtype=torch.float64)
        sum_x2 = torch.zeros_like(sum_x)

        try:
            from tqdm.auto import tqdm
            it = tqdm(loader, desc="fit_manner_scaler", leave=False) if verbose else loader
        except ImportError:
            it = loader

        for batch in it:
            x = batch["pooled_manner"].to(torch.float64)               # [B, L, C, D]
            ind = batch["indicator"].to(torch.float64)                  # [B, C]
            # Mask: 1 where bucket is present, broadcast to [B, L, C, D]
            m = ind.unsqueeze(1).unsqueeze(-1)                          # [B, 1, C, 1]
            x = x * m                                                    # zero-out (already zero, but explicit)
            sum_x  += x.sum(dim=0)
            sum_x2 += (x ** 2).sum(dim=0)
            n_per_cat += ind.sum(dim=0)                                  # [C]

        if (n_per_cat <= 1).any():
            bad = (n_per_cat <= 1).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"MannerStandardiser.fit: cat(s) {bad} had <=1 non-empty bucket "
                f"in fit set (counts={n_per_cat.tolist()}); cannot estimate std"
            )

        n = n_per_cat.view(1, self.n_cats, 1)                            # [1, C, 1]
        mean = sum_x / n
        var = (sum_x2 / n) - mean ** 2
        var.clamp_(min=0.0)
        std = var.sqrt().clamp(min=1e-6)

        self.mean.copy_(mean.to(self.mean.dtype))
        self.std.copy_(std.to(self.std.dtype))
        self.fitted.fill_(True)

        if verbose:
            print(
                f"[manner_scaler] fit  n_per_cat={n_per_cat.tolist()}  "
                f"mean_abs_avg={mean.abs().mean().item():.4f}  "
                f"std range=[{std.min().item():.4e}, {std.max().item():.4e}]"
            )

    def forward(self, x: torch.Tensor, indicator: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C, D] fp16/float; indicator: [B, C] uint8 -> returns
        standardised x with empty buckets re-zeroed (so they contribute zero
        to the layer-weighted sum after standardisation)."""
        x = (x.to(torch.float32) - self.mean) / self.std
        m = indicator.to(x.dtype).unsqueeze(1).unsqueeze(-1)              # [B, 1, C, 1]
        return x * m


class MannerAwareHead(nn.Module):
    """
    Two-stream head:
      - A2 stream: [B, L_a2, stat_a2] -> standardise -> softmax(L_a2 weights) sum
                   -> [B, stat_a2]
      - Manner stream: [B, L_m, C, D_m] -> standardise (mask empty buckets)
                   -> softmax(L_m weights) sum (shared across cats)
                   -> [B, C, D_m] -> flatten -> [B, C*D_m]
      - Indicator [B, C] is appended raw (cheap, lets the head learn a small
        per-utterance bias when a category is missing).

    Final MLP is intentionally narrow (proj_dim=128) and high-dropout (0.6) given
    the input dim jump (10243 vs A2's 4096). Weight decay 3e-3 is the matching
    suggestion in the attack plan.
    """

    def __init__(
        self,
        n_layers_a2: int,
        stat_dim_a2: int,
        n_layers_m: int,
        n_cats: int,
        manner_dim: int,
        proj_dim: int = 128,
        n_classes: int = 2,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.n_layers_a2 = n_layers_a2
        self.stat_dim_a2 = stat_dim_a2
        self.n_layers_m = n_layers_m
        self.n_cats = n_cats
        self.manner_dim = manner_dim
        self.proj_dim = proj_dim

        self.scaler_a2 = FeatureStandardiser(n_layers_a2, stat_dim_a2)
        self.scaler_m = MannerStandardiser(n_layers_m, n_cats, manner_dim)

        self.layer_weights_a2 = nn.Parameter(torch.zeros(n_layers_a2))
        self.layer_weights_m = nn.Parameter(torch.zeros(n_layers_m))

        in_dim = stat_dim_a2 + n_cats * manner_dim + n_cats
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(proj_dim, n_classes)

    def layer_softmax_a2(self) -> torch.Tensor:
        return torch.softmax(self.layer_weights_a2, dim=0)

    def layer_softmax_m(self) -> torch.Tensor:
        return torch.softmax(self.layer_weights_m, dim=0)

    def fit_scalers(self, loader: DataLoader, verbose: bool = True) -> None:
        """Fits both per-stream standardisers in one pass over the loader."""
        # A2 stream uses the existing FeatureStandardiser (expects key "pooled")
        self.scaler_a2.fit(loader, verbose=verbose)
        # Manner stream needs the indicator-aware fit
        self.scaler_m.fit(loader, verbose=verbose)

    def forward(self, batch_or_pooled, pooled_manner=None, indicator=None):
        """Accepts either a batch dict (so it slots into the existing eval loop
        with a small wrapper) or the three tensors directly."""
        if isinstance(batch_or_pooled, dict):
            pooled = batch_or_pooled["pooled"]
            pooled_manner = batch_or_pooled["pooled_manner"]
            indicator = batch_or_pooled["indicator"]
        else:
            pooled = batch_or_pooled

        # A2 stream
        x_a2 = self.scaler_a2(pooled.to(torch.float32))                  # [B, L_a2, stat]
        w_a2 = self.layer_softmax_a2().view(1, -1, 1)
        fused_a2 = (x_a2 * w_a2).sum(dim=1)                               # [B, stat]

        # Manner stream
        x_m = self.scaler_m(pooled_manner, indicator)                     # [B, L_m, C, D]
        w_m = self.layer_softmax_m().view(1, -1, 1, 1)
        fused_m = (x_m * w_m).sum(dim=1)                                  # [B, C, D]
        fused_m = fused_m.flatten(1)                                      # [B, C*D]

        ind_f = indicator.to(fused_m.dtype)                               # [B, C]
        feat = torch.cat([fused_a2, fused_m, ind_f], dim=-1)              # [B, in_dim]
        z = self.proj(feat)
        logits = self.classifier(z)
        return logits, z

    def param_groups(self, base_lr: float = 1e-3) -> list[dict]:
        head_params = list(self.proj.parameters()) + list(self.classifier.parameters())
        return [
            {"params": [self.layer_weights_a2, self.layer_weights_m],
             "lr": base_lr * 0.1, "name": "layer_weights"},
            {"params": head_params, "lr": base_lr, "name": "head"},
        ]
