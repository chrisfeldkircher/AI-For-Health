"""
G1 — voicing scalars from cached pYIN+RMS manner labels.

Free group (no waveform re-read): reads cache/manner_labels/{stem}.pt and
derives ~9 low-dim scalars per utterance. plan.md § 5.7 lists this as the
zero-cost group.

Features (in order):
  0  voiced_fraction
  1  unvoiced_fraction
  2  silence_fraction
  3  voicing_dropout_rate          # voiced -> non-voiced transitions / sec
  4  mean_voiced_segment_sec
  5  mean_unvoiced_segment_sec
  6  mean_silence_segment_sec
  7  voiced_to_unvoiced_per_sec    # specific transition (cold-relevant)
  8  long_silence_rate_per_sec     # silences >= 80 ms

All durations in seconds. Frame rate matches WavLM (50 Hz at hop 320, sr 16k).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


G1_NAMES: tuple[str, ...] = (
    "voiced_fraction",
    "unvoiced_fraction",
    "silence_fraction",
    "voicing_dropout_per_sec",
    "mean_voiced_seg_sec",
    "mean_unvoiced_seg_sec",
    "mean_silence_seg_sec",
    "voiced_to_unvoiced_per_sec",
    "long_silence_rate_per_sec",
)
G1_DIM = len(G1_NAMES)


def _runs(mask: np.ndarray) -> np.ndarray:
    """Run-lengths of contiguous True regions in `mask`."""
    if mask.size == 0:
        return np.array([], dtype=np.int64)
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return ends - starts


def voicing_scalars(labels: np.ndarray, frame_rate: float = 50.0) -> np.ndarray:
    """labels: [T] int in {0=silence, 1=voiced, 2=unvoiced}. Returns [G1_DIM] fp32."""
    T = int(labels.shape[0])
    duration_s = max(T / frame_rate, 1e-3)

    silence  = labels == 0
    voiced   = labels == 1
    unvoiced = labels == 2

    vr = _runs(voiced)
    ur = _runs(unvoiced)
    sr = _runs(silence)

    if T >= 2:
        prev = labels[:-1]
        nxt  = labels[1:]
        v_to_other = int(((prev == 1) & (nxt != 1)).sum())
        v_to_uv    = int(((prev == 1) & (nxt == 2)).sum())
    else:
        v_to_other = 0
        v_to_uv = 0

    long_silence_rate = float((sr >= 4).sum()) / duration_s  # 4 frames @ 50 Hz = 80 ms

    return np.array([
        float(voiced.mean())   if T else 0.0,
        float(unvoiced.mean()) if T else 0.0,
        float(silence.mean())  if T else 0.0,
        v_to_other / duration_s,
        float(vr.mean() / frame_rate) if vr.size else 0.0,
        float(ur.mean() / frame_rate) if ur.size else 0.0,
        float(sr.mean() / frame_rate) if sr.size else 0.0,
        v_to_uv / duration_s,
        long_silence_rate,
    ], dtype=np.float32)


def extract_g1(
    stems: list[str],
    cache_root: str | Path,
    *, frame_rate: float = 50.0,
) -> np.ndarray:
    """Returns X [N, G1_DIM] fp32, aligned to `stems`."""
    labels_dir = Path(cache_root) / "manner_labels"
    out = np.zeros((len(stems), G1_DIM), dtype=np.float32)
    for i, stem in enumerate(stems):
        labels = torch.load(
            labels_dir / f"{stem}.pt",
            weights_only=True, map_location="cpu",
        ).numpy().astype(np.int8, copy=False)
        out[i] = voicing_scalars(labels, frame_rate=frame_rate)
    return out
