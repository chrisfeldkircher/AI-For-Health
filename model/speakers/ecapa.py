"""
ECAPA-TDNN speaker-embedding extraction for pseudo-speaker clustering.

Substrate for:
  - A2 speaker-probe diagnostic (is speaker identity recoverable from z?)
  - A5.5 augmentation (same-speaker exclusion in SpliceSpec generation)
  - A6 contrastive pretraining (speaker-masked loss formulation)
  - A7 MDD adversary (pseudo-speaker targets for the adversarial head)

The URTIC 4students release has no speaker IDs in the TSV, so we cluster
ECAPA embeddings on train and assign devel/test by nearest centroid. This
module only extracts and caches — clustering lives in cluster.py.

Cache layout:
    cache/ecapa-voxceleb/{file_stem}.pt   # [192] fp16 per file
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset


TARGET_SR = 16000


def _load_and_normalise(path: Path) -> tuple[np.ndarray, int]:
    sr, audio = wavfile.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, sr


def _maybe_resample(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)


class _EcapaAudioDataset(Dataset):
    """
    Loads ComParE chunks, normalises, resamples to 16 kHz, and pads to a
    fixed length so the collate can stack into a batch. Returns the original
    relative length so encode_batch can ignore the pad via wav_lens.
    """
    def __init__(self, wav_paths: list[Path], max_seconds: float = 30.0):
        self.paths = wav_paths
        self.max_samples = int(max_seconds * TARGET_SR)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        p = self.paths[idx]
        audio, sr = _load_and_normalise(p)
        audio = _maybe_resample(audio, sr, TARGET_SR)
        raw_len = len(audio)
        if raw_len >= self.max_samples:
            audio = audio[: self.max_samples]
            rel_len = 1.0
        else:
            audio = np.pad(audio, (0, self.max_samples - raw_len))
            rel_len = raw_len / self.max_samples
        return {
            "audio":   torch.from_numpy(audio),
            "rel_len": float(rel_len),
            "stem":    p.stem,
        }


def _ecapa_collate(batch: list[dict]) -> dict:
    return {
        "audio":    torch.stack([b["audio"] for b in batch]),
        "rel_lens": torch.tensor([b["rel_len"] for b in batch], dtype=torch.float32),
        "stems":    [b["stem"] for b in batch],
    }


def load_ecapa_encoder(
    device: str = "cuda",
    savedir: str = "./cache/speechbrain/spkrec-ecapa-voxceleb",
):
    """
    Load speechbrain/spkrec-ecapa-voxceleb. Handles both SpeechBrain import
    layouts (inference.speaker in >=0.5.16, pretrained in older versions).
    """
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": device},
    )


@torch.no_grad()
def extract_ecapa(
    wav_paths: list[Path],
    out_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    max_seconds: float = 30.0,
    skip_existing: bool = True,
    num_workers: int = 0,
) -> int:
    """
    Extract ECAPA-TDNN speaker embeddings for each wav and cache as [192] fp16.

    Returns the number of files newly processed (skipped ones excluded).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_process = [p for p in wav_paths
                  if not (skip_existing and (out_dir / f"{p.stem}.pt").exists())]
    n_skip = len(wav_paths) - len(to_process)
    print(f"[ecapa] to_process={len(to_process)}  skipped_existing={n_skip}  out={out_dir}")
    if not to_process:
        return 0

    encoder = load_ecapa_encoder(device=device)
    ds = _EcapaAudioDataset(to_process, max_seconds=max_seconds)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=_ecapa_collate, num_workers=num_workers,
    )

    try:
        from tqdm.auto import tqdm
        it = tqdm(loader, desc="ecapa", leave=False)
    except ImportError:
        it = loader

    n = 0
    for batch in it:
        wav = batch["audio"].to(device)
        wav_lens = batch["rel_lens"].to(device)
        emb = encoder.encode_batch(wav, wav_lens)       # [B, 1, 192]
        emb = emb.squeeze(1).to(torch.float16).cpu()
        for i, stem in enumerate(batch["stems"]):
            torch.save(emb[i].clone(), out_dir / f"{stem}.pt")
        n += len(batch["stems"])
    return n


def load_ecapa_matrix(
    wav_paths: list[Path], cache_dir: Path
) -> tuple[np.ndarray, list[str]]:
    """
    Load cached ECAPA embeddings as an [N, 192] float32 matrix + ordered
    stem list. Order is preserved from `wav_paths` so callers can align to
    labels / file lists without re-sorting.
    """
    cache_dir = Path(cache_dir)
    stems: list[str] = []
    rows: list[np.ndarray] = []
    for p in wav_paths:
        t = torch.load(cache_dir / f"{p.stem}.pt", weights_only=True, map_location="cpu")
        rows.append(t.to(torch.float32).numpy())
        stems.append(p.stem)
    return np.stack(rows), stems
