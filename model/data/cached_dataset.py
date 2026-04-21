from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

LABEL_MAP = {"C": 1, "NC": 0}


def load_labels(data_dir: str) -> dict[str, int]:
    path = Path(data_dir) / "lab" / "ComParE2017_Cold.tsv"
    out: dict[str, int] = {}
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        next(reader)
        for row in reader:
            if len(row) >= 2:
                out[row[0]] = LABEL_MAP.get(row[1].strip(), -1)
    return out


class PooledCacheDataset(Dataset):
    """
    Loads pre-extracted pooled-stats tensors for the layer-weighted head.

    Each sample is a dict:
        pooled:    [n_layers, stat_dim] fp16
        label:     long scalar (1=Cold, 0=Non-Cold, -1=unlabelled)
        file_name: str
    """

    def __init__(
        self,
        data_dir: str,
        cache_root: str,
        backbone_id: str,
        split: str = "train",
        file_list: Optional[list[str]] = None,
    ):
        self.data_dir = data_dir
        self.pooled_dir = Path(cache_root) / backbone_id / "pooled"
        if not self.pooled_dir.exists():
            raise FileNotFoundError(f"no pooled cache at {self.pooled_dir}")

        self.labels = load_labels(data_dir)
        cached_stems = {p.stem for p in self.pooled_dir.glob("*.pt")}

        if file_list is not None:
            self.files = list(file_list)
        else:
            self.files = sorted(
                f for f in self.labels
                if f.startswith(f"{split}_") and f[:-4] in cached_stems
            )

        missing = [f for f in self.files if f[:-4] not in cached_stems]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files missing from pooled cache "
                f"(first: {missing[:3]}). Run extraction first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        fn = self.files[idx]
        stem = fn[:-4]
        pooled = torch.load(
            self.pooled_dir / f"{stem}.pt",
            weights_only=True,
            map_location="cpu",
        )
        return {
            "pooled": pooled,
            "label": torch.tensor(self.labels.get(fn, -1), dtype=torch.long),
            "file_name": fn,
        }

    def get_labels(self) -> list[int]:
        return [self.labels[f] for f in self.files]

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for f in self.files:
            lab = self.labels[f]
            counts[lab] = counts.get(lab, 0) + 1
        return counts

    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts()
        total = sum(counts.values())
        n_classes = max(counts.keys()) + 1
        w = torch.ones(n_classes, dtype=torch.float32)
        for c, n in counts.items():
            if c >= 0:
                w[c] = total / (n_classes * max(n, 1))
        return w


def stratified_split(
    files: list[str],
    labels: dict[str, int],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Per-class random split. Note: URTIC isn't annotated with speaker IDs in
    the 4students release, so this cannot guarantee speaker-disjoint splits.
    Use only for early-stopping val; use the official devel split as the
    speaker-disjoint honest held-out estimate of test performance."""
    rng = random.Random(seed)
    by_class: dict[int, list[str]] = {}
    for f in files:
        by_class.setdefault(labels[f], []).append(f)

    train_out: list[str] = []
    val_out: list[str] = []
    for cls, fs in by_class.items():
        fs_sorted = sorted(fs)
        rng.shuffle(fs_sorted)
        n_val = max(1, int(round(len(fs_sorted) * val_frac)))
        val_out.extend(fs_sorted[:n_val])
        train_out.extend(fs_sorted[n_val:])

    train_out.sort()
    val_out.sort()
    return train_out, val_out
