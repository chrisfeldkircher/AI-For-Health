"""
G4 — energy / pause / breath scalars from waveform RMS + manner labels.

~11 low-dim scalars per utterance (plan.md § 5.7). Computes RMS on demand
from the .wav (librosa, hop=320, frame_length=2048 — same grid as the manner
labels) and combines with the cached pYIN+RMS manner labels for regime
stratification.

Per-stem cache at cache/handcrafted/g4/{stem}.npy (~50 bytes each, 1 MB total
for 19 101 files). Recomputes only missing stems on subsequent runs.

Features (in order):
   0  rms_lin_mean
   1  rms_lin_std
   2  rms_db_mean
   3  rms_db_std
   4  low_energy_ratio          # frames < (peak_dB - 20)
   5  energy_slope_db_per_sec
   6  rms_db_voiced_minus_silence
   7  rms_db_unvoiced_minus_silence
   8  rms_db_voiced_minus_unvoiced
   9  long_pause_per_sec        # silence runs >= 200 ms
  10  median_silence_seg_sec
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


G4_NAMES: tuple[str, ...] = (
    "rms_lin_mean",
    "rms_lin_std",
    "rms_db_mean",
    "rms_db_std",
    "low_energy_ratio",
    "energy_slope_db_per_sec",
    "rms_db_voiced_minus_silence",
    "rms_db_unvoiced_minus_silence",
    "rms_db_voiced_minus_unvoiced",
    "long_pause_per_sec",
    "median_silence_seg_sec",
)
G4_DIM = len(G4_NAMES)


def _runs(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=np.int64)
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def _regime_db_mean(rms_db: np.ndarray, mask: np.ndarray) -> float:
    """Mean of rms_db over `mask`; falls back to global min when empty so
    contrast features stay defined (and signal "this regime is absent")."""
    if mask.any():
        return float(rms_db[mask].mean())
    return float(rms_db.min())


def energy_scalars(
    audio: np.ndarray,
    labels: np.ndarray,
    *, sr: int = 16000,
    hop_length: int = 320,
    frame_length: int = 2048,
    frame_rate: float = 50.0,
    low_energy_db: float = 20.0,
    long_pause_min_frames: int = 10,  # 10 frames @ 50 Hz = 200 ms
) -> np.ndarray:
    import librosa

    rms = librosa.feature.rms(
        y=audio.astype(np.float32, copy=False),
        frame_length=frame_length, hop_length=hop_length, center=True,
    )[0]

    T = min(rms.shape[0], int(labels.shape[0]))
    rms = rms[:T]
    labels = labels[:T]

    rms_db = 20.0 * np.log10(rms + 1e-8)
    silence  = labels == 0
    voiced   = labels == 1
    unvoiced = labels == 2

    duration_s = max(T / frame_rate, 1e-3)

    if T > 1:
        slope = float(np.polyfit(
            np.arange(T, dtype=np.float32) / frame_rate,
            rms_db.astype(np.float32), 1,
        )[0])
    else:
        slope = 0.0

    low_thresh = rms_db.max() - low_energy_db
    low_energy_ratio = float((rms_db < low_thresh).mean()) if T else 0.0

    db_v = _regime_db_mean(rms_db, voiced)
    db_u = _regime_db_mean(rms_db, unvoiced)
    db_s = _regime_db_mean(rms_db, silence)

    sr_runs = _runs(silence)
    long_pause_per_sec = float((sr_runs >= long_pause_min_frames).sum()) / duration_s
    median_silence_seg_sec = (
        float(np.median(sr_runs) / frame_rate) if sr_runs.size else 0.0
    )

    return np.array([
        float(rms.mean()),
        float(rms.std()),
        float(rms_db.mean()),
        float(rms_db.std()),
        low_energy_ratio,
        slope,
        db_v - db_s,
        db_u - db_s,
        db_v - db_u,
        long_pause_per_sec,
        median_silence_seg_sec,
    ], dtype=np.float32)


def _load_audio_mono(wav_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Lightweight loader: scipy + manual resample fallback. URTIC is 16 kHz mono."""
    from scipy.io import wavfile
    sr, audio = wavfile.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def extract_g4(
    stems: list[str],
    cache_root: str | Path,
    wav_dir: str | Path,
    *, sr: int = 16000,
    skip_existing: bool = True,
    progress: bool = True,
) -> np.ndarray:
    """Returns X [N, G4_DIM] fp32, aligned to `stems`. Caches per-stem npy."""
    cache_root = Path(cache_root)
    labels_dir = cache_root / "manner_labels"
    g4_dir     = cache_root / "handcrafted" / "g4"
    g4_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(wav_dir)

    iterator = stems
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(stems, desc="g4-energy")
        except ImportError:
            pass

    out = np.zeros((len(stems), G4_DIM), dtype=np.float32)
    for i, stem in enumerate(iterator):
        cache_path = g4_dir / f"{stem}.npy"
        if skip_existing and cache_path.exists():
            out[i] = np.load(cache_path)
            continue

        labels = torch.load(
            labels_dir / f"{stem}.pt",
            weights_only=True, map_location="cpu",
        ).numpy().astype(np.int8, copy=False)
        audio = _load_audio_mono(wav_dir / f"{stem}.wav", target_sr=sr)
        feats = energy_scalars(audio, labels, sr=sr)
        np.save(cache_path, feats)
        out[i] = feats
    return out
