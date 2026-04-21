from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FeatureStandardiser(nn.Module):
    """
    Per-position z-score standardiser for pooled backbone stats.

    Fit once on the training cache; persists with the model via buffers so
    the checkpoint is self-contained (no external scaler file). Defaults to
    identity (mean=0, std=1) until fit() is called.

    Shape: expects [B, n_layers, stat_dim]; standardises each of the
    n_layers * stat_dim feature positions independently.
    """

    def __init__(self, n_layers: int, stat_dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.stat_dim = stat_dim
        self.register_buffer("mean", torch.zeros(n_layers, stat_dim))
        self.register_buffer("std", torch.ones(n_layers, stat_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, loader: DataLoader, verbose: bool = True) -> None:
        n = 0
        sum_x = torch.zeros(self.n_layers, self.stat_dim, dtype=torch.float64)
        sum_x2 = torch.zeros(self.n_layers, self.stat_dim, dtype=torch.float64)

        try:
            from tqdm.auto import tqdm
            it = tqdm(loader, desc="fit_scaler", leave=False) if verbose else loader
        except ImportError:
            it = loader

        for batch in it:
            x = batch["pooled"].to(torch.float64)
            sum_x  += x.sum(dim=0)
            sum_x2 += (x ** 2).sum(dim=0)
            n += x.shape[0]

        if n == 0:
            raise RuntimeError("FeatureStandardiser.fit: loader produced zero samples")

        mean = sum_x / n
        var = (sum_x2 / n) - mean ** 2
        var.clamp_(min=0.0)
        std = var.sqrt().clamp(min=1e-6)

        self.mean.copy_(mean.to(self.mean.dtype))
        self.std.copy_(std.to(self.std.dtype))
        self.fitted.fill_(True)

        if verbose:
            ratio = (std.max() / std.min()).item()
            print(
                f"[scaler] fit on n={n}  "
                f"mean_abs_avg={mean.abs().mean().item():.4f}  "
                f"std range=[{std.min().item():.4e}, {std.max().item():.4e}]  "
                f"ratio={ratio:.2e}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
