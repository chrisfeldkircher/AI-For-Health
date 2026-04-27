"""
G5 — modulation spectrogram features (Huckvale-style MOD family).

For each utterance: compute the time series of dB energy in each mel band,
then FFT over time per band → modulation spectrum. Cold speech perturbs
amplitude modulation at three relevant rates:
  - <2 Hz   : prosodic phrasing / breath group rhythm
  - 3-8 Hz  : syllable rate (slowed / less crisp under congestion)
  - 10-20 Hz: fine glottal / articulatory perturbation

Aggregation kept narrow on purpose so the linear-only A5a probe cannot
launder speaker identity through capacity:
   4 acoustic super-bands (low / mid-low / mid-high / high)
 × 8 modulation bands (log-spaced 1-20 Hz)
 × {mean, std}
 = 64-d per utterance.

Cache layout: cache/handcrafted/modulation/{stem}.npy (one 64-d fp32 vector).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


N_ACOUSTIC_BANDS = 4
N_MOD_BANDS = 8
MOD_FMIN_HZ = 1.0
MOD_FMAX_HZ = 20.0


def _mod_band_edges() -> np.ndarray:
    return np.logspace(
        np.log10(MOD_FMIN_HZ), np.log10(MOD_FMAX_HZ), N_MOD_BANDS + 1
    )


def modulation_features(
    audio: np.ndarray,
    sr: int = 16000,
    *, n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    fmin_audio: float = 50.0,
    fmax_audio: float = 8000.0,
) -> np.ndarray:
    """64-d modulation feature vector for one waveform.

    Steps:
      1. Power mel spectrogram (n_mels=40, hop=160 → 100 Hz frame rate).
      2. dB conversion.
      3. Per band: subtract per-band mean (remove DC), Hann window in time,
         take rFFT magnitude → modulation spectrum [n_mels, F_mod].
      4. Bin into 4 acoustic super-bands × 8 log-spaced modulation bands
         (1-20 Hz) and reduce each cell to (mean, std). Flatten to 64-d.
    """
    import librosa

    audio = audio.astype(np.float32, copy=False)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        fmin=fmin_audio, fmax=fmax_audio, power=2.0,
    )
    log_mel = librosa.power_to_db(mel + 1e-10).astype(np.float32)  # [n_mels, T]
    T = int(log_mel.shape[1])

    out = np.zeros((N_ACOUSTIC_BANDS, N_MOD_BANDS, 2), dtype=np.float32)
    if T < 8:
        return out.reshape(-1)

    log_mel = log_mel - log_mel.mean(axis=1, keepdims=True)
    window = np.hanning(T).astype(np.float32)
    spec = np.fft.rfft(log_mel * window[None, :], axis=1)
    mag = np.abs(spec).astype(np.float32)                          # [n_mels, F]

    fs_frame = sr / hop_length                                     # 100 Hz
    freqs = np.fft.rfftfreq(T, d=1.0 / fs_frame)                   # [F]
    edges = _mod_band_edges()

    mels_per_band = n_mels // N_ACOUSTIC_BANDS
    for ab in range(N_ACOUSTIC_BANDS):
        m_lo = ab * mels_per_band
        m_hi = (ab + 1) * mels_per_band if ab < N_ACOUSTIC_BANDS - 1 else n_mels
        sub = mag[m_lo:m_hi]                                        # [k, F]
        for mb in range(N_MOD_BANDS):
            f_lo, f_hi = edges[mb], edges[mb + 1]
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if not mask.any():
                continue
            v = sub[:, mask]
            out[ab, mb, 0] = float(v.mean())
            out[ab, mb, 1] = float(v.std())

    return out.reshape(-1)


@torch.no_grad()
def extract_modulation(
    dataset: Dataset,
    cache_root: str | Path,
    *, sr: int = 16000,
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    fmin_audio: float = 50.0,
    fmax_audio: float = 8000.0,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict:
    """Walk `dataset` (AudioDataset-like), write per-stem 64-d modulation
    vector to cache_root/handcrafted/modulation/{stem}.npy."""
    from .extract import _pad_collate

    out_dir = Path(cache_root) / "handcrafted" / "modulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=_pad_collate,
    )
    if progress:
        try:
            from tqdm.auto import tqdm
            loader = tqdm(loader, desc="modulation")
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

        feat = modulation_features(
            x, sr=sr,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            fmin_audio=fmin_audio, fmax_audio=fmax_audio,
        )
        np.save(target, feat.astype(np.float32))
        n_written += 1

    return {
        "n_written": n_written,
        "n_skipped_existing": n_skipped,
        "out_dir": str(out_dir),
    }
