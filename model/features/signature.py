"""
G10 - path log-signature features (rough-path branch, dependency-free).

The team work plan calls for "the log-signature at low truncation depth ... to
keep dimensionality under control while retaining cross-dimensional interactions
and order-sensitivity that standard pooling discards", applied to RAW acoustic
features (not WavLM embeddings) to test whether the embedding step suppresses
fine-grained temporal signal.

We implement depth-2 log-signature in pure numpy (no iisignature/signatory
dependency, which would need a team decision + install):

  level 1 : path increment  S^i = X_T^i - X_0^i                      (d terms)
  level 2 : Levy areas       A^ij = 1/2 * sum_k [(X_k^i - X_0^i) dX_k^j
                                                 - (X_k^j - X_0^j) dX_k^i]
                             for i < j                               (d(d-1)/2 terms)

Total depth-2 log-sig dim = d + d(d-1)/2. The Levy area is the antisymmetric
part of the second iterated integral; it is exactly the order-sensitive term
(right-then-up vs up-then-right flip its sign) and it is translation-invariant.

Path: a compact 4-channel frame series per utterance built straight from the
waveform (log-RMS energy, log-F0 interpolated, spectral centroid, zero-crossing
rate), each z-scored WITHIN the utterance so increments are scale-free and the
Levy areas are dimensionless. 4 channels -> 4 + 6 = 10-d. Narrow by design so
the linear-only honesty probe cannot launder speaker identity through capacity.

Cache layout: cache/handcrafted/signature/{stem}.npy (one 10-d fp32 vector),
matching the modulation / cqt convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


N_PATH_CHANNELS = 4          # log-RMS, log-F0(interp), spectral centroid, ZCR
G10_HOP = 160                # 100 Hz frame rate at 16 kHz, matches G5/G9


def logsig_depth2(path: np.ndarray) -> np.ndarray:
    """Depth-2 log-signature of a path X of shape [T, d].

    Returns a vector of length d + d*(d-1)/2:
      [increments (d)] ++ [Levy areas A^ij for i<j (d(d-1)/2)].
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2:
        raise ValueError(f"path must be [T, d], got {path.shape}")
    T, d = path.shape
    lvl1 = path[-1] - path[0] if T >= 1 else np.zeros(d)
    n_area = d * (d - 1) // 2
    areas = np.zeros(n_area, dtype=np.float64)
    if T >= 2:
        dX = np.diff(path, axis=0)                 # [T-1, d]
        prefix = path[:-1] - path[0]               # [T-1, d], X_k - X_0 (left points)
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                # 1/2 sum_k [ prefix_i * dX_j - prefix_j * dX_i ]
                areas[idx] = 0.5 * float(
                    np.sum(prefix[:, i] * dX[:, j] - prefix[:, j] * dX[:, i]))
                idx += 1
    return np.concatenate([lvl1, areas]).astype(np.float32)


def _logsig_dim(d: int) -> int:
    return d + d * (d - 1) // 2


G10_DIM: int = _logsig_dim(N_PATH_CHANNELS)   # 10


def _build_names() -> tuple[str, ...]:
    ch = ("logRMS", "logF0", "centroid", "zcr")
    names = [f"sig_incr_{c}" for c in ch]
    for i in range(N_PATH_CHANNELS):
        for j in range(i + 1, N_PATH_CHANNELS):
            names.append(f"sig_area_{ch[i]}_{ch[j]}")
    return tuple(names)


G10_NAMES: tuple[str, ...] = _build_names()


def acoustic_path(
    audio: np.ndarray,
    sr: int = 16000,
    *, hop_length: int = G10_HOP,
    n_fft: int = 512,
) -> np.ndarray:
    """Build the [T, 4] per-frame acoustic path, z-scored per channel.

    Channels: log-RMS energy, log-F0 (voiced-interpolated), spectral centroid,
    zero-crossing rate. Per-channel z-scoring within the utterance makes the
    signature scale-free (a global gain change leaves the standardized path,
    hence the log-sig, unchanged)."""
    import librosa

    audio = audio.astype(np.float32, copy=False)
    if audio.shape[0] < hop_length * 4:
        return np.zeros((0, N_PATH_CHANNELS), dtype=np.float64)

    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    log_rms = np.log(rms + 1e-8)

    cent = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=n_fft, hop_length=hop_length)[0]

    # F0 via pYIN; interpolate over unvoiced (NaN) frames, log-scale.
    f0, _, _ = librosa.pyin(
        audio, sr=sr, fmin=65.0, fmax=400.0,
        frame_length=n_fft * 4, hop_length=hop_length)
    f0 = np.asarray(f0, dtype=np.float64)
    if np.isnan(f0).all():
        f0 = np.full_like(f0, 120.0)
    else:
        good = ~np.isnan(f0)
        f0 = np.interp(np.arange(f0.size), np.flatnonzero(good), f0[good])
    log_f0 = np.log(f0 + 1e-8)

    T = min(len(log_rms), len(log_f0), len(cent), len(zcr))
    P = np.stack([log_rms[:T], log_f0[:T], cent[:T], zcr[:T]], axis=1).astype(np.float64)

    mu = P.mean(axis=0, keepdims=True)
    sd = P.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (P - mu) / sd


def signature_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """10-d depth-2 log-signature of the 4-channel acoustic path."""
    P = acoustic_path(audio, sr=sr)
    if P.shape[0] < 2:
        return np.zeros(G10_DIM, dtype=np.float32)
    return logsig_depth2(P)


@torch.no_grad()
def extract_signature(
    dataset: Dataset,
    cache_root: str | Path,
    *, sr: int = 16000,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict:
    """Walk `dataset` (AudioDataset-like), write per-stem 10-d log-sig vector to
    cache_root/handcrafted/signature/{stem}.npy. Serial reference; the notebook
    cell fans signature_features() out over CPU cores."""
    from .extract import _pad_collate

    out_dir = Path(cache_root) / "handcrafted" / "signature"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=_pad_collate)
    if progress:
        try:
            from tqdm.auto import tqdm
            loader = tqdm(loader, desc="signature")
        except ImportError:
            pass

    n_written = n_skipped = 0
    for batch in loader:
        fn = batch["file_name"][0]
        stem = fn[:-4] if fn.endswith(".wav") else fn
        target = out_dir / f"{stem}.npy"
        if skip_existing and target.exists():
            n_skipped += 1
            continue
        audio = batch["audio"][0].numpy().astype(np.float32, copy=False)
        valid = int(batch["attention_mask"][0].sum().item())
        np.save(target, signature_features(audio[:valid], sr=sr).astype(np.float32))
        n_written += 1
    return {"n_written": n_written, "n_skipped_existing": n_skipped, "out_dir": str(out_dir)}


def extract_g10(stems: list[str], cache_root: str | Path) -> np.ndarray:
    """Returns X [N, G10_DIM] fp32 aligned to `stems`. Requires
    cache/handcrafted/signature/ populated by extract_signature() (or the
    notebook cell's parallel fan-out of signature_features())."""
    cache_root = Path(cache_root)
    out_dir = cache_root / "handcrafted" / "signature"
    if not out_dir.exists():
        raise FileNotFoundError(
            f"no signature cache at {out_dir} - run "
            "features.signature.extract_signature() (or the A5b_k3_signature "
            "notebook cell's extraction step) first")
    out = np.zeros((len(stems), G10_DIM), dtype=np.float32)
    for i, stem in enumerate(stems):
        out[i] = np.load(out_dir / f"{stem}.npy")
    return out


__all__ = ["logsig_depth2", "acoustic_path", "signature_features",
           "extract_signature", "extract_g10", "G10_NAMES", "G10_DIM",
           "N_PATH_CHANNELS"]
