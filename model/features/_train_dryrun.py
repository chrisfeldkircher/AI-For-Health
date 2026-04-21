"""Dry-run of the trainer plumbing on the 10-chunk smoke cache."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
sys.path.insert(0, str(MODEL_ROOT))

from data.cached_dataset import PooledCacheDataset, stratified_split, load_labels
from features.head import LayerWeightedPooledHead
from features.train import train_head


def main():
    data_dir = "../dataset/ComParE2017_Cold_4students"
    cache_root = "../cache"
    backbone_id = "microsoft_wavlm-base-plus"

    full_ds = PooledCacheDataset(data_dir=data_dir, cache_root=cache_root,
                                 backbone_id=backbone_id, split="train")
    print(f"[dryrun] cached train chunks: {len(full_ds)}  counts={full_ds.class_counts()}")

    labels_map = load_labels(data_dir)
    train_files, val_files = stratified_split(full_ds.files, labels_map, val_frac=0.3, seed=7)
    train_ds = PooledCacheDataset(data_dir, cache_root, backbone_id, file_list=train_files)
    val_ds = PooledCacheDataset(data_dir, cache_root, backbone_id, file_list=val_files)
    print(f"[dryrun] split: train={len(train_ds)} val={len(val_ds)}")

    sample = full_ds[0]["pooled"]
    n_layers, stat_dim = sample.shape
    head = LayerWeightedPooledHead(n_layers=n_layers, stat_dim=stat_dim,
                                   proj_dim=64, n_classes=2, dropout=0.0)

    train_head(
        head=head,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=val_ds,  # reuse val as stand-in so the "test" code path is exercised
        epochs=3,
        batch_size=4,
        base_lr=1e-3,
        class_weights=full_ds.class_weights(),
        early_stop_patience=99,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ckpt_path=None,
    )
    print("\n[dryrun] PASS")


if __name__ == "__main__":
    main()
