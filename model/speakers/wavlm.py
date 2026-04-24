"""
WavLM-based speaker-embedding extraction as an alternative substrate to
ECAPA-TDNN for pseudo-speaker diagnostics.

We use `microsoft/wavlm-base-plus-sv` — the WavLM-base variant fine-tuned
for speaker verification with an x-vector head — so the comparison against
ECAPA-VoxCeleb is apples-to-apples (both are SV-tuned speaker embeddings,
just different backbones). Output is 512-d.

Cache layout:
    cache/wavlm-base-plus-sv/{file_stem}.pt   # [512] fp16 per file
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ecapa import TARGET_SR, _load_and_normalise, _maybe_resample


class _WavLMAudioDataset(Dataset):
    def __init__(self, wav_paths: list[Path], max_seconds: float = 30.0):
        self.paths = wav_paths
        self.max_samples = int(max_seconds * TARGET_SR)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        p = self.paths[idx]
        audio, sr = _load_and_normalise(p)
        audio = _maybe_resample(audio, sr, TARGET_SR)
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        return {"audio": audio.astype(np.float32), "stem": p.stem}


class _WavLMCollate:
    """Module-level collate so DataLoader workers can pickle it."""
    def __init__(self, feat_extractor, target_sr: int = TARGET_SR):
        self.feat_extractor = feat_extractor
        self.target_sr = target_sr

    def __call__(self, batch: list[dict]) -> dict:
        audios = [b["audio"] for b in batch]
        enc = self.feat_extractor(
            audios, sampling_rate=self.target_sr, padding=True,
            return_tensors="pt", return_attention_mask=True,
        )
        return {
            "input_values":   enc["input_values"],
            "attention_mask": enc["attention_mask"],
            "stems":          [b["stem"] for b in batch],
        }


def load_wavlm_encoder(
    model_id: str = "microsoft/wavlm-base-plus-sv",
    device: str = "cpu",
):
    from transformers import AutoFeatureExtractor, WavLMForXVector
    feat = AutoFeatureExtractor.from_pretrained(model_id)
    model = WavLMForXVector.from_pretrained(model_id).eval().to(device)
    return feat, model


@torch.no_grad()
def extract_wavlm(
    wav_paths: list[Path],
    out_dir: Path,
    model_id: str = "microsoft/wavlm-base-plus-sv",
    device: str = "cpu",
    batch_size: int = 8,
    max_seconds: float = 30.0,
    skip_existing: bool = True,
    num_workers: int = 0,
) -> int:
    """
    Extract WavLM speaker embeddings for each wav and cache as [512] fp16.
    Returns the number of files newly processed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_process = [p for p in wav_paths
                  if not (skip_existing and (out_dir / f"{p.stem}.pt").exists())]
    n_skip = len(wav_paths) - len(to_process)
    print(f"[wavlm] to_process={len(to_process)}  skipped_existing={n_skip}  out={out_dir}")
    if not to_process:
        return 0

    feat_extractor, model = load_wavlm_encoder(model_id=model_id, device=device)

    ds = _WavLMAudioDataset(to_process, max_seconds=max_seconds)
    collate = _WavLMCollate(feat_extractor, target_sr=TARGET_SR)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
    )

    try:
        from tqdm.auto import tqdm
        it = tqdm(loader, desc="wavlm", leave=False)
    except ImportError:
        it = loader

    n = 0
    for batch in it:
        iv = batch["input_values"].to(device)
        am = batch["attention_mask"].to(device)
        out = model(input_values=iv, attention_mask=am)
        emb = out.embeddings.to(torch.float16).cpu()   # [B, 512]
        for i, stem in enumerate(batch["stems"]):
            torch.save(emb[i].clone(), out_dir / f"{stem}.pt")
        n += len(batch["stems"])
    return n


def load_wavlm_matrix(
    wav_paths: list[Path], cache_dir: Path
) -> tuple[np.ndarray, list[str]]:
    """Load cached WavLM embeddings as [N, 512] float32 + stem list."""
    cache_dir = Path(cache_dir)
    stems: list[str] = []
    rows: list[np.ndarray] = []
    for p in wav_paths:
        t = torch.load(cache_dir / f"{p.stem}.pt", weights_only=True, map_location="cpu")
        rows.append(t.to(torch.float32).numpy())
        stems.append(p.stem)
    return np.stack(rows), stems
