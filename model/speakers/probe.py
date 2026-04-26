"""
Speaker probe: how much pseudo-speaker identity is recoverable from the
classification head's pre-logit embedding z?

A high probe accuracy / NMI means z still carries speaker information, which
is the signature of the Huckvale-trap shortcut: the classifier is using
"who is speaking" as a proxy for "do they have a cold." The same probe is
re-run after every de-confounding rung (A5.5 / A6 / A7) — we want the
numbers to drop.

Protocol:
  - Train the probe on train_fit z (same samples the classifier trained on,
    since we're asking about representation geometry, not generalisation of
    the main task).
  - Evaluate on devel z (unseen speakers by URTIC construction).
  - Targets are pseudo-speaker cluster IDs from speakers/cluster.py TSVs.
  - Report top-1 accuracy and NMI between probe predictions and true
    cluster labels. NMI is more informative at high k (degrades gracefully
    when the probe is guessing a close-but-wrong cluster).

Chance level for top-1 at k=210 is ~0.5%. A speaker-free z would score
near chance; a speaker-encoding z will score 20-80%.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, Dataset


class SpeakerProbe(nn.Module):
    """2-layer MLP from z (proj_dim) to pseudo-speaker logits."""
    def __init__(self, z_dim: int, n_clusters: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_clusters),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@torch.no_grad()
def extract_z(
    head: nn.Module,
    ds: Dataset,
    device: str,
    batch_size: int = 256,
) -> tuple[torch.Tensor, list[str]]:
    """Run `head` over `ds`, return (z [N, proj_dim] on CPU, file_name list)."""
    from features.train import _pooled_collate
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_pooled_collate)
    head.eval().to(device)
    zs: list[torch.Tensor] = []
    names: list[str] = []
    for batch in loader:
        pooled = batch["pooled"].to(device)
        _, z = head(pooled)
        zs.append(z.detach().cpu())
        names.extend(batch["file_name"])
    return torch.cat(zs, dim=0), names


@torch.no_grad()
def extract_z_joint(
    head: nn.Module,
    ds: Dataset,
    device: str,
    batch_size: int = 256,
) -> tuple[torch.Tensor, list[str]]:
    """Variant for two-stream heads (e.g. MannerAwareHead) — uses the joint
    collate and passes the batch dict to the head."""
    from features.train import _joint_collate, _to_device_joint
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_joint_collate)
    head.eval().to(device)
    zs: list[torch.Tensor] = []
    names: list[str] = []
    for batch in loader:
        feats = _to_device_joint(batch, device)
        _, z = head(feats)
        zs.append(z.detach().cpu())
        names.extend(batch["file_name"])
    return torch.cat(zs, dim=0), names


@dataclass
class ProbeResult:
    top1: float
    nmi: float
    train_top1: float
    n_train: int
    n_eval: int
    k: int
    epochs_trained: int
    best_eval_top1_epoch: int


def _align_labels(names: list[str], assignments: dict[str, int]) -> np.ndarray:
    """Convert file_name-keyed assignments into a label array aligned to `names`."""
    # PooledCacheDataset returns file_name with the .wav suffix; cluster TSV
    # is keyed by stem. Strip suffix to match.
    out = np.empty(len(names), dtype=np.int64)
    for i, n in enumerate(names):
        stem = n[:-4] if n.endswith(".wav") else n
        out[i] = assignments[stem]
    return out


def train_probe(
    z_train: torch.Tensor, y_train: np.ndarray,
    z_eval:  torch.Tensor, y_eval:  np.ndarray,
    n_clusters: int,
    *,
    device: str = "cuda",
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    verbose: bool = True,
) -> ProbeResult:
    """
    Train the speaker probe on (z_train, y_train), evaluate on (z_eval, y_eval).
    Tracks best eval top-1 across epochs; reports that best value as the probe
    score (the probe is a measurement tool, not a model we deploy).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    probe = SpeakerProbe(z_dim=z_train.shape[1], n_clusters=n_clusters,
                         hidden_dim=hidden_dim, dropout=dropout).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    z_train_d = z_train.to(device)
    y_train_d = torch.from_numpy(y_train).long().to(device)
    z_eval_d  = z_eval.to(device)
    y_eval_d  = torch.from_numpy(y_eval).long().to(device)

    n = z_train_d.shape[0]
    best_top1 = -1.0
    best_epoch = -1
    best_pred: Optional[np.ndarray] = None
    last_train_top1 = 0.0

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(n, device=device)
        running_loss, correct, seen = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = probe(z_train_d[idx])
            loss = loss_fn(logits, y_train_d[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item() * idx.numel()
            correct += int((logits.argmax(-1) == y_train_d[idx]).sum().item())
            seen += idx.numel()
        last_train_top1 = correct / max(seen, 1)

        probe.eval()
        with torch.no_grad():
            eval_logits = probe(z_eval_d)
            eval_pred = eval_logits.argmax(-1)
            eval_top1 = float((eval_pred == y_eval_d).float().mean().item())
        if eval_top1 > best_top1:
            best_top1 = eval_top1
            best_epoch = epoch
            best_pred = eval_pred.detach().cpu().numpy()

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"  [probe ep {epoch:02d}/{epochs}] "
                  f"loss={running_loss/seen:.4f}  train_top1={last_train_top1:.4f}  "
                  f"eval_top1={eval_top1:.4f}  best={best_top1:.4f}@{best_epoch}")

    nmi = float(normalized_mutual_info_score(y_eval, best_pred))
    return ProbeResult(
        top1=best_top1, nmi=nmi,
        train_top1=last_train_top1,
        n_train=z_train.shape[0], n_eval=z_eval.shape[0],
        k=n_clusters, epochs_trained=epochs, best_eval_top1_epoch=best_epoch,
    )
