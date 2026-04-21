"""
Round-trip smoke test:
  1. Build WavLM-base-plus
  2. Pull 10 clips from AudioDataset
  3. Extract pooled stats, write cache + manifest
  4. Reload cache, run through LayerWeightedPooledHead
  5. Assert shapes/dtypes
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
sys.path.insert(0, str(MODEL_ROOT))

from data.data import AudioDataset
from features.backbone import build_backbone
from features.cache import CacheManifest, load_pooled
from features.extract import extract_pooled
from features.head import LayerWeightedPooledHead


def run(
    data_dir: str = "../dataset/ComParE2017_Cold_4students",
    cache_root: str = "../cache",
    backbone_name: str = "wavlm-base-plus",
    n_chunks: int = 10,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device} backbone={backbone_name}")

    ds = AudioDataset(
        data_dir=data_dir,
        split="train",
        use_mel=False,
        use_opensmile=False,
        pad_or_truncate_secs=5.0,
    )
    subset = Subset(ds, list(range(min(n_chunks, len(ds)))))
    print(f"[smoke] dataset loaded: {len(subset)} chunks")

    backbone = build_backbone(backbone_name, device=device)
    print(
        f"[smoke] backbone ready: n_layers={backbone.n_layers} "
        f"hidden_dim={backbone.hidden_dim} hash={backbone.checkpoint_hash}"
    )

    manifest = extract_pooled(
        backbone=backbone,
        dataset=subset,
        cache_root=cache_root,
        batch_size=2,
        skip_existing=False,
        progress=True,
    )
    print(f"[smoke] extract done, wrote {manifest.n_chunks} chunks")

    manifest_path = Path(cache_root) / backbone.backbone_id / "manifest.json"
    reloaded = CacheManifest.load(manifest_path)
    assert reloaded.is_compatible(manifest), "manifest round-trip failed"
    print(f"[smoke] manifest round-trip ok: {reloaded.created_at}")

    sample_stem = ds.file_list[0][:-4]
    pooled_path = Path(cache_root) / backbone.backbone_id / "pooled" / f"{sample_stem}.pt"
    cached = load_pooled(pooled_path)
    print(f"[smoke] cached tensor: shape={tuple(cached.shape)} dtype={cached.dtype}")
    assert cached.shape == (backbone.n_layers, 4 * backbone.hidden_dim)
    assert cached.dtype == torch.float16

    head = LayerWeightedPooledHead(
        n_layers=backbone.n_layers,
        stat_dim=4 * backbone.hidden_dim,
        proj_dim=128,
        n_classes=2,
    )
    batch = cached.unsqueeze(0)
    logits, emb = head(batch)
    print(f"[smoke] head forward ok: logits={tuple(logits.shape)} emb={tuple(emb.shape)}")
    assert logits.shape == (1, 2)
    assert emb.shape == (1, 128)

    groups = head.param_groups(base_lr=1e-3)
    assert abs(groups[0]["lr"] - 1e-4) < 1e-12, "layer_weights lr must be 0.1 × head lr"
    print("[smoke] param_groups ok (layer_weights lr=1e-4, head lr=1e-3)")

    print("[smoke] PASS")


if __name__ == "__main__":
    run()
