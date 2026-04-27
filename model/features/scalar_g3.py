"""
G3 — voice quality slice of eGeMAPSv02 (jitter, shimmer, HNR, spectral tilts).

plan.md § 5.7 G3: jitter, shimmer, HNR, harmonicity, spectral tilt[voiced],
CPP if available. eGeMAPSv02 covers all of these except CPP (CPP is in
ComParE_2016 only).

Carving by name prefix on eGeMAPS column names — exclusive with G6. The
carving lives here, not at extraction time, so we can re-run carving without
recomputing eGeMAPS.

Includes (each prefix typically matches _amean + _stddevNorm functionals):
  jitterLocal_*           # cycle-to-cycle F0 perturbation
  shimmerLocaldB_*        # cycle-to-cycle amplitude perturbation (dB)
  HNRdBACF_*              # harmonics-to-noise ratio (autocorrelation)
  alphaRatioV_*           # voiced spectral tilt 50-1k vs 1-5k
  hammarbergIndexV_*      # voiced spectral peak ratio
  slopeV0-500_*           # voiced spectral slope, low band
  slopeV500-1500_*        # voiced spectral slope, mid band

Excluded (per plan.md § 5.2 explicit excludes):
  formants F1/F2/F3       # speaker-identifying
"""
from __future__ import annotations

import numpy as np

from .opensmile_extract import extract_egemaps


G3_PREFIXES: tuple[str, ...] = (
    "jitterLocal_",
    "shimmerLocaldB_",
    "HNRdBACF_",
    "alphaRatioV_",
    "hammarbergIndexV_",
    "slopeV0-500_",
    "slopeV500-1500_",
)


def _select_columns(columns: list[str], prefixes: tuple[str, ...]) -> list[int]:
    return [i for i, c in enumerate(columns) if any(c.startswith(p) for p in prefixes)]


def carve_g3(X_egemaps: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    idx = _select_columns(columns, G3_PREFIXES)
    if not idx:
        raise ValueError(
            f"no eGeMAPS columns matched G3 prefixes {G3_PREFIXES}. "
            f"Got first 5 columns: {columns[:5]}"
        )
    return X_egemaps[:, idx], [columns[i] for i in idx]


def extract_g3(
    stems: list[str],
    cache_root: str,
    wav_dir: str,
    *, skip_existing: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Returns (X [N, |G3|] fp32, names list)."""
    X, columns = extract_egemaps(
        stems, cache_root, wav_dir,
        skip_existing=skip_existing, progress=progress,
    )
    return carve_g3(X, columns)
