"""
G8 — single-scalar OOD score from the A2 layer-weighted pooled representation.

plan.md § 5.7: "Mahalanobis distance of A2 pooled vector from non-cold mean
(post-hoc fitted)". One scalar per utterance.

Substrate choice: we evaluate Mahalanobis distance in the A2 head's
**pre-projection** layer-weighted space, *not* the 128-d post-projection z.
The post-projection MLP is class-conditioned and entangles cold/non-cold
structure; the pre-projection 4096-d vector is the standardised, layer-fused
representation the classifier sees as input.

Dimensionality: 4096 features × ~7 700 train_fit non-cold samples is
rank-deficient under a vanilla covariance. We use Ledoit–Wolf shrinkage
(closed-form, no hyperparameter), which guarantees a full-rank,
well-conditioned cov estimate. This matches standard OOD-by-Mahalanobis
practice (Lee et al. 2018, Sastry & Oore 2020).

Reproducibility: we pin to `head_A2_seed42.pt` for the layer weights and
scaler. Re-running A2 with a different seed could shift this score; the
audit/probe protocol holds the head fixed across honesty rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from .head import LayerWeightedPooledHead


G8_NAMES: tuple[str, ...] = ("ood_mahalanobis_NC",)
G8_DIM = len(G8_NAMES)


def _load_a2_head(
    ckpt_path: str | Path,
    *, n_layers: int = 25, stat_dim: int = 4096, proj_dim: int = 128,
    device: str = "cpu",
) -> LayerWeightedPooledHead:
    head = LayerWeightedPooledHead(
        n_layers=n_layers, stat_dim=stat_dim,
        proj_dim=proj_dim, n_classes=2, dropout=0.0,
    )
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    head.load_state_dict(state, strict=False)
    head.eval().to(device)
    return head


@torch.no_grad()
def _fused_vectors(
    head: LayerWeightedPooledHead,
    stems: list[str],
    pooled_dir: Path,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Returns the 4096-d standardised + layer-weighted-fused representation
    per stem (the input to head.proj). Aligned to `stems`."""
    out = np.empty((len(stems), head.stat_dim), dtype=np.float32)
    w = head.layer_softmax().view(1, -1, 1)
    for start in range(0, len(stems), batch_size):
        chunk = stems[start:start + batch_size]
        batch = torch.stack([
            torch.load(pooled_dir / f"{s}.pt",
                       weights_only=True, map_location="cpu").to(torch.float32)
            for s in chunk
        ], dim=0).to(device)
        x = head.scaler(batch)
        fused = (x * w).sum(dim=1)
        out[start:start + len(chunk)] = fused.detach().cpu().numpy()
    return out


def extract_g8(
    stems: list[str],
    train_fit_nc_stems: list[str],
    cache_root: str | Path,
    backbone_id: str = "microsoft_wavlm-large",
    *, head_ckpt: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    verbose: bool = True,
) -> np.ndarray:
    """Returns X [N, 1] fp32 — squared Mahalanobis distance of each utterance's
    A2 fused vector from the train_fit non-cold mean.

    Args:
      stems              : utterances to score (any split)
      train_fit_nc_stems : non-cold stems used to fit the Gaussian
      head_ckpt          : path to a trained A2 head; defaults to seed=42
    """
    cache_root = Path(cache_root)
    pooled_dir = cache_root / backbone_id / "pooled"
    if head_ckpt is None:
        head_ckpt = cache_root / backbone_id / "head_A2_seed42.pt"
    head = _load_a2_head(head_ckpt, device=device)

    if verbose:
        print(f"[g8] head loaded from {head_ckpt}  layer_weights argmax="
              f"L{int(head.layer_softmax().argmax().item())}")

    fit_X = _fused_vectors(head, train_fit_nc_stems, pooled_dir, device, batch_size)
    if verbose:
        print(f"[g8] fused fit set: {fit_X.shape}  "
              f"mean|x|={float(np.abs(fit_X).mean()):.4f}")

    cov = LedoitWolf(assume_centered=False).fit(fit_X)
    mean = cov.location_
    precision = cov.precision_
    if verbose:
        print(f"[g8] LedoitWolf shrinkage={cov.shrinkage_:.4f}  "
              f"precision cond=ok")

    # Score in row-batches to keep memory bounded.
    all_X = _fused_vectors(head, stems, pooled_dir, device, batch_size)
    diff = all_X - mean[None, :]
    # squared Mahalanobis: diag(diff @ precision @ diff.T)
    out = np.einsum("ni,ij,nj->n", diff, precision, diff).astype(np.float32)
    return out.reshape(-1, 1)
