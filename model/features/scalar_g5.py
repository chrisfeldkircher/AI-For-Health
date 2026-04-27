"""
G5 — modulation spectrogram group wrapper.

Reads cached 64-d modulation features (cache/handcrafted/modulation/{stem}.npy)
written by features.modulation.extract_modulation(). Layout:

    4 acoustic super-bands (a0..a3, low → high mel) ×
    8 modulation bands     (m0..m7, log-spaced 1-20 Hz) ×
    {amean, stddev}
  = 64-d.

The naming convention mirrors openSMILE's `_amean` / `_stddevNorm` suffixes
so downstream tooling (audit pretty-printers, correlation diagnostics) does
not need a special case for G5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .modulation import N_ACOUSTIC_BANDS, N_MOD_BANDS, extract_modulation


def _build_names() -> tuple[str, ...]:
    names: list[str] = []
    for ab in range(N_ACOUSTIC_BANDS):
        for mb in range(N_MOD_BANDS):
            for stat in ("amean", "stddev"):
                names.append(f"mod_a{ab}_m{mb}_{stat}")
    return tuple(names)


G5_NAMES: tuple[str, ...] = _build_names()
G5_DIM: int = len(G5_NAMES)


def extract_g5(
    stems: list[str],
    cache_root: str | Path,
) -> np.ndarray:
    """Returns X [N, G5_DIM] fp32, aligned to `stems`. Requires
    cache/handcrafted/modulation/ populated by
    features.modulation.extract_modulation()."""
    cache_root = Path(cache_root)
    out_dir    = cache_root / "handcrafted" / "modulation"
    if not out_dir.exists():
        raise FileNotFoundError(
            f"no modulation cache at {out_dir} — run "
            "features.modulation.extract_modulation() first"
        )
    out = np.zeros((len(stems), G5_DIM), dtype=np.float32)
    for i, stem in enumerate(stems):
        out[i] = np.load(out_dir / f"{stem}.npy")
    return out


__all__ = ["G5_NAMES", "G5_DIM", "extract_g5", "extract_modulation"]
