"""
openSMILE eGeMAPSv02 functional extractor (88 features per utterance).

Source for G3 (voice quality) and G6 (spectral shape) groups in A5a; cached
once per stem at cache/handcrafted/egemaps/{stem}.npy. Feature column names
saved alongside as cache/handcrafted/egemaps/_columns.json so per-group
slicing in scalar_g3 / scalar_g6 can match by name without re-running
openSMILE.

eGeMAPSv02 is the 2016 Eyben et al. minimalistic feature set. We choose it
over ComParE_2016 (6 373 features) for two reasons:
  - 88 functionals fit comfortably under the v1 per-group dimensionality
    ceiling once we carve into G3 (~14) + G6 (~30).
  - The set is curated and physiologically motivated (voice quality, spectral
    tilt, MFCC), which matches A5a's "low-d, audited" framing better than
    the kitchen-sink ComParE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def extract_egemaps(
    stems: list[str],
    cache_root: str | Path,
    wav_dir: str | Path,
    *, skip_existing: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Returns (X [N, 88] fp32 aligned to `stems`, column-name list).

    Caches per stem at cache_root/handcrafted/egemaps/{stem}.npy and writes
    the column-name list to cache_root/handcrafted/egemaps/_columns.json on
    the first run.
    """
    import opensmile

    cache_root = Path(cache_root)
    out_dir    = cache_root / "handcrafted" / "egemaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir    = Path(wav_dir)
    cols_path  = out_dir / "_columns.json"

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    iterator = stems
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(stems, desc="egemaps")
        except ImportError:
            pass

    out = None  # lazily allocated once we know the column count
    columns: list[str] | None = None
    for i, stem in enumerate(iterator):
        cache_path = out_dir / f"{stem}.npy"
        if skip_existing and cache_path.exists():
            arr = np.load(cache_path)
        else:
            df = smile.process_file(str(wav_dir / f"{stem}.wav"))
            arr = df.values[0].astype(np.float32, copy=False)
            np.save(cache_path, arr)
            if columns is None:
                columns = list(df.columns)
                cols_path.write_text(
                    json.dumps(columns, indent=2), encoding="utf-8",
                )

        if out is None:
            out = np.empty((len(stems), arr.shape[0]), dtype=np.float32)
        out[i] = arr

    if columns is None:
        # Everything was cached — recover column names from disk.
        if not cols_path.exists():
            raise FileNotFoundError(
                f"{cols_path} missing — re-extract at least one stem to "
                "populate column names."
            )
        columns = json.loads(cols_path.read_text(encoding="utf-8"))

    assert out is not None
    return out, columns


def load_egemaps(
    stems: list[str],
    cache_root: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Read-only loader over an already-populated egemaps cache."""
    cache_root = Path(cache_root)
    out_dir    = cache_root / "handcrafted" / "egemaps"
    cols_path  = out_dir / "_columns.json"
    columns = json.loads(cols_path.read_text(encoding="utf-8"))
    out = np.empty((len(stems), len(columns)), dtype=np.float32)
    for i, stem in enumerate(stems):
        out[i] = np.load(out_dir / f"{stem}.npy")
    return out, columns
