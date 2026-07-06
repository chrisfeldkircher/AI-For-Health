"""
G9 — constant-Q transform (CQT) spectral group.

Fusion candidate from the team work plan ("CQT-based features ... will pass
through the same honesty gate before consideration"). The CQT gives constant
frequency resolution per octave, so the low-frequency region (F0 region,
subglottal coupling, nasal murmur below ~500 Hz) is resolved much more finely
than in the mel front end that G5 and the WavLM features already see. Cold
symptoms (congestion, hoarseness) plausibly shift narrow low-frequency
structure that a 40-mel front end blurs.

Aggregation kept narrow on purpose, mirroring G5's design rule so the
linear-only A5a probe cannot launder speaker identity through capacity:

    84 CQT bins (C1 ~ 32.7 Hz to ~4.2 kHz, 12 bins/octave, 7 octaves)
  x {amean, stddev} of the per-bin dB trajectory over time
  = 168-d per utterance.

Cache layout: cache/handcrafted/cqt/{stem}.npy (one 168-d fp32 vector),
matching the modulation cache convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CQT_N_BINS = 84
CQT_BINS_PER_OCTAVE = 12
CQT_FMIN_HZ = 32.703195662574764   # C1; librosa.note_to_hz("C1")
CQT_HOP = 160                       # 100 Hz frame rate at 16 kHz, matches G5


def _build_names() -> tuple[str, ...]:
    names: list[str] = []
    for b in range(CQT_N_BINS):
        for stat in ("amean", "stddev"):
            names.append(f"cqt_b{b:02d}_{stat}")
    return tuple(names)


G9_NAMES: tuple[str, ...] = _build_names()
G9_DIM: int = len(G9_NAMES)


def cqt_features(
    audio: np.ndarray,
    sr: int = 16000,
    *, n_bins: int = CQT_N_BINS,
    bins_per_octave: int = CQT_BINS_PER_OCTAVE,
    fmin: float = CQT_FMIN_HZ,
    hop_length: int = CQT_HOP,
) -> np.ndarray:
    """168-d CQT feature vector for one waveform.

    Steps:
      1. Constant-Q transform (84 bins, 12/octave from C1, hop 160).
      2. dB conversion of the magnitude.
      3. Per bin: (mean, std) of the dB trajectory over time. Flatten.
    """
    import librosa

    audio = audio.astype(np.float32, copy=False)
    out = np.zeros((n_bins, 2), dtype=np.float32)
    # CQT needs enough samples for the longest (lowest-frequency) filter.
    if audio.shape[0] < hop_length * 8:
        return out.reshape(-1)

    C = librosa.cqt(
        y=audio, sr=sr,
        n_bins=n_bins, bins_per_octave=bins_per_octave,
        fmin=fmin, hop_length=hop_length,
    )
    mag_db = librosa.amplitude_to_db(np.abs(C) + 1e-10).astype(np.float32)  # [n_bins, T]

    out[:, 0] = mag_db.mean(axis=1)
    out[:, 1] = mag_db.std(axis=1)
    return out.reshape(-1)


@torch.no_grad()
def extract_cqt(
    dataset: Dataset,
    cache_root: str | Path,
    *, sr: int = 16000,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict:
    """Walk `dataset` (AudioDataset-like), write per-stem 168-d CQT vector to
    cache_root/handcrafted/cqt/{stem}.npy. Serial reference implementation;
    the notebook cell fans the same cqt_features() out over CPU cores."""
    from .extract import _pad_collate

    out_dir = Path(cache_root) / "handcrafted" / "cqt"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=_pad_collate,
    )
    if progress:
        try:
            from tqdm.auto import tqdm
            loader = tqdm(loader, desc="cqt")
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

        feat = cqt_features(x, sr=sr)
        np.save(target, feat.astype(np.float32))
        n_written += 1

    return {
        "n_written": n_written,
        "n_skipped_existing": n_skipped,
        "out_dir": str(out_dir),
    }


def extract_g9(
    stems: list[str],
    cache_root: str | Path,
) -> np.ndarray:
    """Returns X [N, G9_DIM] fp32, aligned to `stems`. Requires
    cache/handcrafted/cqt/ populated by extract_cqt() (or the notebook
    cell's parallel fan-out of cqt_features())."""
    cache_root = Path(cache_root)
    out_dir    = cache_root / "handcrafted" / "cqt"
    if not out_dir.exists():
        raise FileNotFoundError(
            f"no cqt cache at {out_dir} — run features.scalar_g9_cqt.extract_cqt() "
            "(or the A5b_k3_cqt notebook cell's extraction step) first"
        )
    out = np.zeros((len(stems), G9_DIM), dtype=np.float32)
    for i, stem in enumerate(stems):
        out[i] = np.load(out_dir / f"{stem}.npy")
    return out


__all__ = ["G9_NAMES", "G9_DIM", "cqt_features", "extract_cqt", "extract_g9"]
