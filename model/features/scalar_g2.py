"""
G2 — F0 / prosody scalars from cached pYIN F0 contour + manner labels.

~10 low-dim features per utterance (plan.md § 5.7). F0 contour read from
cache/f0/{stem}.npy (NaN at unvoiced); voicing mask from manner_labels.

Features (in order):
   0  f0_voiced_fraction         # fraction of frames with finite F0 (sanity)
   1  f0_mean_hz                 # mean F0 over voiced frames
   2  f0_std_hz
   3  f0_p10_hz
   4  f0_p90_hz
   5  f0_range_hz                # p90 - p10 (robust range)
   6  f0_log_mean_st             # mean of log2(F0) in semitones-from-100Hz
   7  f0_jitter_local            # mean |df/dt| / mean F0 (cycle-to-cycle frac)
   8  f0_missingness_in_voiced   # voiced-by-manner but pYIN says unvoiced
   9  f0_voiced_run_count_per_sec # number of contiguous voiced-F0 runs / sec
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


G2_NAMES: tuple[str, ...] = (
    "f0_voiced_fraction",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_p10_hz",
    "f0_p90_hz",
    "f0_range_hz",
    "f0_log_mean_st",
    "f0_jitter_local",
    "f0_missingness_in_voiced",
    "f0_voiced_run_count_per_sec",
)
G2_DIM = len(G2_NAMES)


def _runs(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=np.int64)
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def prosody_scalars(
    f0: np.ndarray,           # [T] float32, NaN at unvoiced
    labels: np.ndarray,       # [T] int8, 0=silence, 1=voiced, 2=unvoiced
    *, frame_rate: float = 50.0,
) -> np.ndarray:
    T = int(min(f0.shape[0], labels.shape[0]))
    f0 = f0[:T].astype(np.float32, copy=False)
    labels = labels[:T]
    duration_s = max(T / frame_rate, 1e-3)

    finite = np.isfinite(f0)
    voiced_by_manner = labels == 1

    f0_voiced_fraction = float(finite.mean()) if T else 0.0
    f0_missingness_in_voiced = (
        float((voiced_by_manner & ~finite).sum() / max(voiced_by_manner.sum(), 1))
        if voiced_by_manner.any() else 0.0
    )

    # Use F0 only where it's finite — pYIN already gates on voicing.
    f0_v = f0[finite]
    if f0_v.size >= 2:
        mean_hz   = float(f0_v.mean())
        std_hz    = float(f0_v.std())
        p10, p90  = np.percentile(f0_v, [10, 90])
        range_hz  = float(p90 - p10)
        log_mean_st = float(12.0 * np.log2(mean_hz / 100.0)) if mean_hz > 0 else 0.0
        # Local jitter: mean abs frame-to-frame difference / mean F0, on the
        # voiced contour only (gaps treated as breaks via diff).
        df = np.abs(np.diff(f0_v))
        jitter_local = float(df.mean() / max(mean_hz, 1e-3))
    else:
        mean_hz = std_hz = float(f0_v.mean()) if f0_v.size else 0.0
        p10 = p90 = mean_hz
        range_hz = 0.0
        log_mean_st = 0.0
        jitter_local = 0.0

    voiced_runs = _runs(finite)
    runs_per_sec = float(voiced_runs.size) / duration_s

    return np.array([
        f0_voiced_fraction,
        mean_hz,
        std_hz,
        float(p10),
        float(p90),
        range_hz,
        log_mean_st,
        jitter_local,
        f0_missingness_in_voiced,
        runs_per_sec,
    ], dtype=np.float32)


def extract_g2(
    stems: list[str],
    cache_root: str | Path,
    *, frame_rate: float = 50.0,
) -> np.ndarray:
    """Returns X [N, G2_DIM] fp32, aligned to `stems`. Requires cache/f0/
    populated by features.f0.extract_f0(...)."""
    cache_root = Path(cache_root)
    f0_dir     = cache_root / "f0"
    labels_dir = cache_root / "manner_labels"
    if not f0_dir.exists():
        raise FileNotFoundError(
            f"no F0 cache at {f0_dir} — run features.f0.extract_f0() first"
        )

    out = np.zeros((len(stems), G2_DIM), dtype=np.float32)
    for i, stem in enumerate(stems):
        f0 = np.load(f0_dir / f"{stem}.npy")
        labels = torch.load(
            labels_dir / f"{stem}.pt",
            weights_only=True, map_location="cpu",
        ).numpy().astype(np.int8, copy=False)
        out[i] = prosody_scalars(f0, labels, frame_rate=frame_rate)
    return out
