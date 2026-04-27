"""
G6 — spectral shape slice of eGeMAPSv02 (low-order MFCC + spectral flux).

plan.md § 5.7 G6: low-order MFCC stats (1–6), spectral_centroid, rolloff,
flux, high/low band ratios. eGeMAPSv02 covers MFCC 1–4 and spectralFlux;
centroid/rolloff are not in eGeMAPS so we accept the narrower MFCC + flux
slice for v1 (~30-d). If a richer G6 turns out to matter we can reach back
to ComParE_2016 or compute centroid/rolloff via librosa.

Includes (each prefix typically matches V / UV / no-suffix variants × _amean
+ _stddevNorm functionals):
  spectralFlux*           # frame-to-frame spectral change (all regimes)
  mfcc1*  mfcc2*  mfcc3*  mfcc4*

Excluded:
  formants F1/F2/F3       # speaker-identifying (plan § 5.2)
  loudness*               # overlaps G4 energy
  alphaRatio*, hammarbergIndex*, slopeV/UV  # in G3
  F0semitone*             # in G2
"""
from __future__ import annotations

import numpy as np

from .opensmile_extract import extract_egemaps


G6_PREFIXES: tuple[str, ...] = (
    "spectralFlux",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4",
)


def _select_columns(columns: list[str], prefixes: tuple[str, ...]) -> list[int]:
    return [i for i, c in enumerate(columns) if any(c.startswith(p) for p in prefixes)]


def carve_g6(X_egemaps: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    idx = _select_columns(columns, G6_PREFIXES)
    if not idx:
        raise ValueError(
            f"no eGeMAPS columns matched G6 prefixes {G6_PREFIXES}. "
            f"Got first 5 columns: {columns[:5]}"
        )
    return X_egemaps[:, idx], [columns[i] for i in idx]


def extract_g6(
    stems: list[str],
    cache_root: str,
    wav_dir: str,
    *, skip_existing: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Returns (X [N, |G6|] fp32, names list)."""
    X, columns = extract_egemaps(
        stems, cache_root, wav_dir,
        skip_existing=skip_existing, progress=progress,
    )
    return carve_g6(X, columns)
