from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json


@dataclass(frozen=True)
class SpliceSpec:
    """One cross-speaker splicing realisation.

    The source chunk's waveform is sliced at `splice_position` (fractional,
    0..1) and the tail is replaced by the partner's tail (rescaled to match
    duration). Consumed by `model/features/extract.py` when building the
    augmented pooled-stats cache.
    """
    source: str
    partner: str
    splice_position: float
    seed: int


def _partner_pool(
    source: str,
    files: list[str],
    speakers: Optional[dict[str, str]],
    symmetric_across_classes: bool,
    labels: dict[str, int],
) -> list[str]:
    pool = [f for f in files if f != source]
    if speakers is not None and source in speakers:
        src_spk = speakers[source]
        pool = [f for f in pool if speakers.get(f) != src_spk]
    if not symmetric_across_classes:
        src_label = labels.get(source, -1)
        pool = [f for f in pool if labels.get(f, -1) == src_label]
    return pool


def generate_splice_specs(
    files: list[str],
    labels: dict[str, int],
    speakers: Optional[dict[str, str]] = None,
    k: int = 10,
    seed: int = 42,
    symmetric_across_classes: bool = True,
    splice_position_range: tuple[float, float] = (0.3, 0.7),
) -> dict[str, list[SpliceSpec]]:
    """
    Build K splicing realisations per source file.

    symmetric_across_classes=True draws partners from the whole corpus
    regardless of class, so "was spliced" is independent of the label.
    """
    rng = random.Random(seed)
    specs: dict[str, list[SpliceSpec]] = {}
    low, high = splice_position_range

    for source in files:
        pool = _partner_pool(source, files, speakers, symmetric_across_classes, labels)
        if not pool:
            specs[source] = []
            continue
        source_specs = []
        for j in range(k):
            partner = rng.choice(pool)
            pos = rng.uniform(low, high)
            sub_seed = rng.randint(0, 2**31 - 1)
            source_specs.append(SpliceSpec(source, partner, pos, sub_seed))
        specs[source] = source_specs
    return specs


def save_specs(specs: dict[str, list[SpliceSpec]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {src: [asdict(s) for s in lst] for src, lst in specs.items()}
    with open(path, "w") as f:
        json.dump(serial, f)


def load_specs(path) -> dict[str, list[SpliceSpec]]:
    with open(Path(path), "r") as f:
        raw = json.load(f)
    return {src: [SpliceSpec(**d) for d in lst] for src, lst in raw.items()}
