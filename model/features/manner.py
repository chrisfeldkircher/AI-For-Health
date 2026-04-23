"""
Acoustic manner-of-articulation labelling via pYIN voicing + RMS energy.

Three categories, computed directly from the raw waveform:
  0 silence  — RMS energy below per-utterance peak by a silence threshold (dB)
  1 voiced   — speech (above silence) AND pYIN flags as voiced
  2 unvoiced — speech AND pYIN flags as unvoiced (fricative turbulence, bursts)

Frame rate matches WavLM (50 Hz at hop 320, sr 16 kHz). Output length is
truncated/zero-padded to the WavLM frame count for the same utterance so
labels index cleanly into `frames/L{N}/{stem}.pt`.

Why this instead of a phoneme CTC labeller on URTIC:
  - pYIN voicing detection has decades of validation in the speech literature;
    a reviewer can name the paper. Smearing heuristics on underconfident CTC
    output cannot be validated against any ground truth we have for URTIC.
  - Cold signal lives in *voiced* regions (nasal formants, glottal pulse) and
    *unvoiced* regions (fricative spectrum broadening with mucus). The 3-way
    split captures the same articulation axis that phoneme categories would,
    at a coarser but defensible granularity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MANNER_CATEGORIES = ("silence", "voiced", "unvoiced")
CAT_SILENCE, CAT_VOICED, CAT_UNVOICED = 0, 1, 2


def compute_manner(
    audio: np.ndarray,
    valid_samples: int,
    wavlm_frame_count: int,
    sr: int = 16000,
    hop_length: int = 320,
    frame_length: int = 2048,
    fmin: float = 65.0,
    fmax: float = 400.0,
    silence_rel_db: float = 30.0,
) -> np.ndarray:
    """
    audio               : [T_audio] float32 at sr (may be zero-padded)
    valid_samples       : number of unpadded samples in audio
    wavlm_frame_count   : target output length (WavLM frames for this utterance)
    silence_rel_db      : silence floor relative to per-utterance peak RMS
    returns             : [wavlm_frame_count] int8 labels in {0, 1, 2}
    """
    import librosa

    x = audio[:valid_samples].astype(np.float32, copy=False)

    # pYIN is the slow component; fmin/fmax bracket human F0 (male 65 Hz to
    # female 400 Hz). center=True gives frames time-centred on hop multiples,
    # aligning (up to 1–2 frame slop) with WavLM's 20-ms grid.
    _f0, voiced_flag, _voiced_prob = librosa.pyin(
        x, sr=sr, fmin=fmin, fmax=fmax,
        frame_length=frame_length, hop_length=hop_length, center=True,
    )
    rms = librosa.feature.rms(
        y=x, frame_length=frame_length, hop_length=hop_length, center=True,
    )[0]

    # Silence gate: RMS in dB vs per-utterance peak.
    rms_db = 20.0 * np.log10(rms + 1e-8)
    silence_thresh = rms_db.max() - silence_rel_db
    speech_mask = rms_db >= silence_thresh
    voiced_flag = np.where(np.isnan(voiced_flag), False, voiced_flag)

    labels = np.full(rms.shape, CAT_SILENCE, dtype=np.int8)
    labels[speech_mask & voiced_flag]  = CAT_VOICED
    labels[speech_mask & ~voiced_flag] = CAT_UNVOICED

    # Align to WavLM frame count. librosa center=True gives 1 + valid_samples // hop
    # frames (~= valid_samples / hop + 1); WavLM CNN gives slightly fewer. Truncate
    # if longer, pad with silence if shorter — mismatch is always small (1–3 frames).
    T = wavlm_frame_count
    if labels.shape[0] >= T:
        return labels[:T]
    out = np.full(T, CAT_SILENCE, dtype=np.int8)
    out[: labels.shape[0]] = labels
    return out


@torch.no_grad()
def extract_manner_labels(
    dataset: Dataset,
    cache_root: str,
    backbone_id: str = "microsoft_wavlm-large",
    frames_cache_root: Optional[str] = None,
    sr: int = 16000,
    hop_length: int = 320,
    frame_length: int = 2048,
    fmin: float = 65.0,
    fmax: float = 400.0,
    silence_rel_db: float = 30.0,
    num_workers: int = 0,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict:
    """
    Walks `dataset` (AudioDataset-like: yields {"file_name", "audio"}), computes
    per-frame manner labels aligned to the WavLM frame cache, writes:

      {cache_root}/manner_labels/{stem}.pt      [T] int8
      {cache_root}/manner_labels/categories.json

    Frame counts are read from `{frames_cache_root}/{backbone_id}/frames/L1/{stem}.pt`
    (defaults to `cache_root` for the normal case). Pass `frames_cache_root` separately
    when writing labels to a tmp/validation dir while reading frames from the real cache.
    """
    import json
    from .extract import _pad_collate

    out_dir = Path(cache_root) / "manner_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "categories.json").write_text(
        json.dumps({"names": list(MANNER_CATEGORIES)}, indent=2),
        encoding="utf-8",
    )

    frames_root = frames_cache_root if frames_cache_root is not None else cache_root
    frames_dir = Path(frames_root) / backbone_id / "frames" / "L1"
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"frame cache missing at {frames_dir} — run extract_frames() first"
        )

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=_pad_collate,
    )
    if progress:
        try:
            from tqdm.auto import tqdm
            loader = tqdm(loader, desc="manner[pYIN+RMS]")
        except ImportError:
            pass

    n_written = 0
    for batch in loader:
        fn = batch["file_name"][0]
        stem = fn[:-4] if fn.endswith(".wav") else fn
        target = out_dir / f"{stem}.pt"
        if skip_existing and target.exists():
            continue

        frame_path = frames_dir / f"{stem}.pt"
        if not frame_path.exists():
            continue
        wavlm_T = torch.load(frame_path, map_location="cpu", weights_only=True).shape[0]

        audio = batch["audio"][0].numpy()
        valid_samples = int(batch["attention_mask"][0].sum().item())

        labels = compute_manner(
            audio=audio, valid_samples=valid_samples, wavlm_frame_count=wavlm_T,
            sr=sr, hop_length=hop_length, frame_length=frame_length,
            fmin=fmin, fmax=fmax, silence_rel_db=silence_rel_db,
        )
        torch.save(torch.from_numpy(labels), target)
        n_written += 1

    return {
        "n_written": n_written,
        "categories": list(MANNER_CATEGORIES),
        "out_dir": str(out_dir),
    }
