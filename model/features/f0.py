"""
F0 contour extraction (pYIN), cached per stem at cache/f0/{stem}.npy.

Same pYIN parameters as features/manner.py so the F0 contours align frame-for-
frame with the cached manner labels (50 Hz @ hop 320, sr 16k). Voicing
decisions still come from manner_labels — F0 here is purely the pitch
contour, with NaN at unvoiced/silence frames (pYIN's natural output).

This is a separate extractor (does not modify manner.py) so the existing
manner_labels cache is not invalidated. Cost is one full pYIN re-run
(~22 h CPU at hop=320 on 19 101 chunks) — we accept the cost rather than
mutating a validated cache.

If the runtime is unacceptable we can switch to librosa.yin (deterministic,
~10x faster, no probabilistic step) and re-run as a sensitivity check; v1
sticks with pYIN for consistency with the manner labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def extract_f0(
    dataset: Dataset,
    cache_root: str,
    *, sr: int = 16000,
    hop_length: int = 320,
    frame_length: int = 2048,
    fmin: float = 65.0,
    fmax: float = 400.0,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict:
    """
    Walks `dataset` (AudioDataset-like: yields {"file_name", "audio",
    "attention_mask"}), runs pYIN, writes per-stem F0 contour to
    cache/f0/{stem}.npy as float32 with NaN at unvoiced frames.
    """
    import librosa
    from .extract import _pad_collate

    out_dir = Path(cache_root) / "f0"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=_pad_collate,
    )
    if progress:
        try:
            from tqdm.auto import tqdm
            loader = tqdm(loader, desc="f0[pYIN]")
        except ImportError:
            pass

    n_written = 0
    n_skipped = 0
    for batch in loader:
        fn = batch["file_name"][0]
        stem = fn[:-4] if fn.endswith(".wav") else fn
        target = out_dir / f"{stem}.npy"
        if skip_existing and target.exists():
            n_skipped += 1
            continue

        audio = batch["audio"][0].numpy().astype(np.float32, copy=False)
        valid = int(batch["attention_mask"][0].sum().item())
        x = audio[:valid]

        f0, _voiced_flag, _voiced_prob = librosa.pyin(
            x, sr=sr, fmin=fmin, fmax=fmax,
            frame_length=frame_length, hop_length=hop_length, center=True,
        )
        np.save(target, f0.astype(np.float32))
        n_written += 1

    return {
        "n_written": n_written,
        "n_skipped_existing": n_skipped,
        "out_dir": str(out_dir),
    }
