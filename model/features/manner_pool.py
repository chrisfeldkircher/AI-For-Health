"""
Manner-category pooling over cached WavLM frames.

For each utterance, reads:
  {cache_root}/{backbone_id}/frames/L{N}/{stem}.pt   [T, D] fp16
  {cache_root}/manner_labels/{stem}.pt               [T'] int8 in {0, 1, 2}

and writes a single bundle:
  {cache_root}/{backbone_id}/manner_pooled/{stem}.pt
      pooled    : [len(layers), n_cats, 2*D] fp16   (mean, std per category per layer)
      indicator : [n_cats] uint8                    (1 if that category had >=1 frame)

Design notes:
  - Only mean + std per category. Third/fourth moments are too noisy on the smallest
    bucket per utterance (unvoiced ~20% × 399 frames ~ 80 frames, sometimes < 20).
  - Label / frame length may differ by 1-3 frames (librosa center=True vs WavLM CNN
    stride arithmetic). We truncate to min(T_frames, T_labels) — the pYIN labeller
    already pads with silence to frame count, so the truncation just drops trailing
    padding in the rare long-labels case.
  - Empty buckets are zero-filled and their indicator flag is 0 so the head can mask
    them without NaN propagation.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .extract import DEFAULT_FRAME_LAYERS
from .manner import MANNER_CATEGORIES


def pool_manner_one(
    frames_by_layer: dict[int, torch.Tensor],
    labels: torch.Tensor,
    n_cats: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    frames_by_layer : {L: [T, D] fp16/float}
    labels          : [T'] int8/int in {0..n_cats-1}
    returns         : pooled [L, n_cats, 2*D] fp16, indicator [n_cats] uint8
    """
    layers = sorted(frames_by_layer.keys())
    D = frames_by_layer[layers[0]].shape[-1]
    T_frames = frames_by_layer[layers[0]].shape[0]
    T = min(int(T_frames), int(labels.shape[0]))

    labels = labels[:T].to(torch.long)

    pooled = torch.zeros(len(layers), n_cats, 2 * D, dtype=torch.float16)
    indicator = torch.zeros(n_cats, dtype=torch.uint8)

    for c in range(n_cats):
        mask = labels == c
        n = int(mask.sum().item())
        if n == 0:
            continue
        indicator[c] = 1
        for li, L in enumerate(layers):
            f = frames_by_layer[L][:T][mask].to(torch.float32)
            mu = f.mean(dim=0)
            # unbiased=False so n=1 still gives std=0 instead of NaN
            sd = f.std(dim=0, unbiased=False)
            pooled[li, c, :D] = mu.to(torch.float16)
            pooled[li, c, D:] = sd.to(torch.float16)

    return pooled, indicator


def extract_manner_pooled(
    dataset: Dataset,
    cache_root: str,
    backbone_id: str = "microsoft_wavlm-large",
    layers: tuple[int, ...] = DEFAULT_FRAME_LAYERS,
    num_workers: int = 0,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict:
    """
    Walks `dataset` for its file_name stream, reads cached frames + manner labels,
    writes per-utterance manner-pooled bundles.
    """
    from .extract import _pad_collate

    cache_root_p = Path(cache_root)
    frames_root = cache_root_p / backbone_id / "frames"
    labels_root = cache_root_p / "manner_labels"
    out_dir = cache_root_p / backbone_id / "manner_pooled"
    out_dir.mkdir(parents=True, exist_ok=True)

    for L in layers:
        if not (frames_root / f"L{L}").exists():
            raise FileNotFoundError(
                f"frame cache missing for layer {L} at {frames_root / f'L{L}'}"
            )
    if not labels_root.exists():
        raise FileNotFoundError(f"manner labels missing at {labels_root}")

    n_cats = len(MANNER_CATEGORIES)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=_pad_collate,
    )
    if progress:
        try:
            from tqdm.auto import tqdm
            loader = tqdm(loader, desc="manner_pool")
        except ImportError:
            pass

    n_written = 0
    n_skipped_missing = 0
    for batch in loader:
        fn = batch["file_name"][0]
        stem = fn[:-4] if fn.endswith(".wav") else fn
        target = out_dir / f"{stem}.pt"
        if skip_existing and target.exists():
            continue

        label_path = labels_root / f"{stem}.pt"
        if not label_path.exists():
            n_skipped_missing += 1
            continue

        frames_by_layer: dict[int, torch.Tensor] = {}
        missing_layer = False
        for L in layers:
            fp = frames_root / f"L{L}" / f"{stem}.pt"
            if not fp.exists():
                missing_layer = True
                break
            frames_by_layer[L] = torch.load(fp, map_location="cpu", weights_only=True)
        if missing_layer:
            n_skipped_missing += 1
            continue

        labels = torch.load(label_path, map_location="cpu", weights_only=True)
        pooled, indicator = pool_manner_one(frames_by_layer, labels, n_cats=n_cats)
        torch.save({"pooled": pooled, "indicator": indicator}, target)
        n_written += 1

    return {
        "n_written": n_written,
        "n_skipped_missing": n_skipped_missing,
        "out_dir": str(out_dir),
        "layers": list(layers),
        "n_cats": n_cats,
    }
