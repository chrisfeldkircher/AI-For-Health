from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .backbone import Backbone
from .cache import CacheManifest, save_pooled


def pooled_stats(x: torch.Tensor) -> torch.Tensor:
    """
    Mean, std, skewness, kurtosis over the time axis.

    x:       [B, L, T, D] any float dtype
    returns: [B, L, 4*D]  fp16
    """
    x32 = x.to(torch.float32)
    mean = x32.mean(dim=2)
    centered = x32 - mean.unsqueeze(2)
    var = (centered ** 2).mean(dim=2)
    std = var.sqrt()
    std_safe = std.clamp(min=1e-6).unsqueeze(2)
    skew = ((centered / std_safe) ** 3).mean(dim=2)
    kurt = ((centered / std_safe) ** 4).mean(dim=2) - 3.0
    return torch.cat([mean, std, skew, kurt], dim=-1).to(torch.float16)


def pooled_stats_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked mean, std, skew, kurtosis. Padded frames are excluded.

    x:       [B, L, T, D]
    mask:    [B, T] bool, True = valid
    returns: [B, L, 4*D] fp16
    """
    x32 = x.to(torch.float32)
    m = mask.to(torch.float32).unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
    n = m.sum(dim=2).clamp(min=1.0)                         # [B, 1, 1]

    mean = (x32 * m).sum(dim=2) / n
    centered = (x32 - mean.unsqueeze(2)) * m
    var = (centered ** 2).sum(dim=2) / n
    std = var.sqrt()
    std_safe = std.clamp(min=1e-6).unsqueeze(2)
    skew = ((centered / std_safe) ** 3).sum(dim=2) / n
    kurt = ((centered / std_safe) ** 4).sum(dim=2) / n - 3.0
    return torch.cat([mean, std, skew, kurt], dim=-1).to(torch.float16)


def _pad_collate(batch: list[dict]) -> dict:
    audios = [b["audio"] for b in batch]
    lens = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    T_max = int(lens.max())
    B = len(audios)

    padded = torch.zeros(B, T_max, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.long)
    for i, a in enumerate(audios):
        padded[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = 1

    return {
        "file_name": [b["file_name"] for b in batch],
        "audio": padded,
        "attention_mask": mask,
        "label": torch.stack([b["label"] for b in batch]),
    }


def extract_pooled(
    backbone: Backbone,
    dataset: Dataset,
    cache_root: str,
    batch_size: int = 4,
    num_workers: int = 0,
    skip_existing: bool = True,
    progress: bool = True,
) -> CacheManifest:
    """
    Runs the backbone over `dataset` and writes per-utterance pooled stats
    to `{cache_root}/{backbone_id}/pooled/{file_stem}.pt`.

    Each file holds a tensor of shape [n_layers, 4*hidden_dim] in fp16.
    """
    root = Path(cache_root) / backbone.backbone_id / "pooled"
    root.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_pad_collate,
    )

    if progress:
        try:
            from tqdm.auto import tqdm
            loader_iter = tqdm(loader, desc=f"extract[{backbone.backbone_id}]")
        except ImportError:
            loader_iter = loader
    else:
        loader_iter = loader

    n_written = 0
    for batch in loader_iter:
        file_names = batch["file_name"]
        remaining = []
        for i, fn in enumerate(file_names):
            stem = fn[:-4] if fn.endswith(".wav") else fn
            target = root / f"{stem}.pt"
            if skip_existing and target.exists():
                continue
            remaining.append((i, stem, target))
        if not remaining:
            continue

        hidden, out_mask = backbone(batch["audio"], batch["attention_mask"])
        stats = pooled_stats_masked(hidden, out_mask)  # [B, L, 4D]

        for i, stem, target in remaining:
            save_pooled(target, stats[i].contiguous().cpu())
            n_written += 1

    manifest = CacheManifest.create(
        backbone=backbone,
        stat_dim=4 * backbone.hidden_dim,
        n_chunks=n_written,
    )
    manifest.save(Path(cache_root) / backbone.backbone_id / "manifest.json")
    return manifest
