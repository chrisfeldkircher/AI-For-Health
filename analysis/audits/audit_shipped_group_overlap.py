"""Audit whether historical pseudo-speaker splits remain grouped under side-local ECAPA labels.

The historical TSV was fitted on Train and then assigned Development recordings to
Train centroids.  This script reproduces the exact StratifiedGroupKFold split used
by the experiments, then measures overlap using an independently fitted, side-local
KMeans partition.  Cold labels are used only to reproduce stratification; they are
never used to construct either speaker proxy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import pandas as pd


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
DATA_DIR = ROOT / "dataset" / "ComParE2017_Cold_4students"
SHIPPED_TSV = ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv"
SIDE_LABELS = ROOT / "results" / "speaker_proxy_method_labels.npz"
OUT = ROOT / "results" / "shipped_group_overlap_audit.json"

sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import load_labels, stratified_grouped_split  # noqa: E402


def load_group_map(tsv: Path) -> dict[str, int]:
    frame = pd.read_csv(tsv, sep="\t", usecols=["file_stem", "cluster"])
    return dict(zip(frame["file_stem"].astype(str), frame["cluster"].astype(int)))


def overlap_stats(
    fit_files: list[str], holdout_files: list[str], groups: dict[str, int]
) -> dict[str, int | float]:
    def stem(name: str) -> str:
        return name[:-4] if name.endswith(".wav") else name

    fit_groups = {groups[stem(name)] for name in fit_files}
    holdout_groups = {groups[stem(name)] for name in holdout_files}
    shared = fit_groups & holdout_groups
    all_files = fit_files + holdout_files
    affected = sum(groups[stem(name)] in shared for name in all_files)
    return {
        "fit_unique_groups": len(fit_groups),
        "holdout_unique_groups": len(holdout_groups),
        "overlapping_groups": len(shared),
        "affected_recordings": affected,
        "affected_recording_fraction": affected / len(all_files),
    }


def class_rate(files: list[str], labels: dict[str, int]) -> float:
    return float(np.mean([labels[name] for name in files]))


def one_split(
    *,
    files: list[str],
    labels: dict[str, int],
    split_groups: dict[str, int],
    audit_groups: dict[str, int],
    val_frac: float,
    seed: int,
) -> dict:
    fit_files, holdout_files = stratified_grouped_split(
        files, labels, split_groups, val_frac=val_frac, seed=seed
    )
    return {
        "seed": seed,
        "fit_recordings": len(fit_files),
        "holdout_recordings": len(holdout_files),
        "fit_cold_rate": class_rate(fit_files, labels),
        "holdout_cold_rate": class_rate(holdout_files, labels),
        "overlap_under_split_labels": overlap_stats(
            fit_files, holdout_files, split_groups
        ),
        "overlap_under_side_local_labels": overlap_stats(
            fit_files, holdout_files, audit_groups
        ),
    }


def summarize(records: list[dict]) -> dict:
    counts = [r["overlap_under_side_local_labels"]["overlapping_groups"] for r in records]
    fractions = [
        r["overlap_under_side_local_labels"]["affected_recording_fraction"]
        for r in records
    ]
    return {
        "n_splits": len(records),
        "overlapping_groups_min": min(counts),
        "overlapping_groups_max": max(counts),
        "overlapping_groups_mean": mean(counts),
        "affected_recording_fraction_min": min(fractions),
        "affected_recording_fraction_max": max(fractions),
        "affected_recording_fraction_mean": mean(fractions),
        "affected_recording_fraction_population_sd": pstdev(fractions),
    }


def main() -> None:
    labels = load_labels(str(DATA_DIR))
    shipped = load_group_map(SHIPPED_TSV)
    saved = np.load(SIDE_LABELS, allow_pickle=False)

    side_local: dict[str, dict[str, int]] = {}
    for side in ("train", "devel"):
        stems = saved[f"{side}__stems"].astype(str)
        groups = saved[f"{side}__kmeans"].astype(int)
        side_local[side] = dict(zip(stems, groups))

    side_files = {
        side: sorted(f"{stem}.wav" for stem in side_local[side])
        for side in ("train", "devel")
    }
    for side, files in side_files.items():
        missing_labels = [name for name in files if labels.get(name, -1) not in (0, 1)]
        missing_shipped = [name for name in files if name[:-4] not in shipped]
        if missing_labels or missing_shipped:
            raise RuntimeError(
                f"{side}: missing labels={len(missing_labels)}, "
                f"missing shipped groups={len(missing_shipped)}"
            )

    train = one_split(
        files=side_files["train"],
        labels=labels,
        split_groups=shipped,
        audit_groups=side_local["train"],
        val_frac=0.10,
        seed=42,
    )

    devel_seeds = [42, 1, 2, 3, 5, 11, 17, 23, 31, 53, 99]
    devel_records = [
        one_split(
            files=side_files["devel"],
            labels=labels,
            split_groups=shipped,
            audit_groups=side_local["devel"],
            val_frac=0.50,
            seed=seed,
        )
        for seed in devel_seeds
    ]

    # Positive control: splitting Development by the side-local labels must close
    # overlap under those same labels.  It is not a performance rerun.
    corrected_control = one_split(
        files=side_files["devel"],
        labels=labels,
        split_groups=side_local["devel"],
        audit_groups=side_local["devel"],
        val_frac=0.50,
        seed=42,
    )

    report = {
        "question": (
            "Did the historical Train-centroid k=210 labels actually make the "
            "within-side experimental splits speaker-proxy disjoint?"
        ),
        "inputs": {
            "historical_split_groups": str(SHIPPED_TSV.relative_to(ROOT)),
            "independent_audit_groups": str(SIDE_LABELS.relative_to(ROOT)),
            "audit_partition": "side-local raw-L2 ECAPA KMeans, k=210, seed=42",
            "splitter": "model/data/cached_dataset.py::stratified_grouped_split",
            "cold_label_use": "stratification only; not used to fit speaker proxies",
        },
        "train_canonical_seed42_val_frac_0.10": train,
        "devel_historical_split_audited_by_side_local_groups": {
            "canonical_seed42": devel_records[0],
            "all_seeds": devel_records,
            "summary": summarize(devel_records),
        },
        "devel_side_local_split_positive_control_seed42": corrected_control,
        "interpretation": {
            "train": (
                "The shipped labels were fitted on Train, so the canonical Train split "
                "remains disjoint under the independently recovered Train-local partition."
            ),
            "devel": (
                "Zero overlap under the historical IDs is true by construction but is not "
                "evidence of speaker-proxy disjointness: those IDs fragment the recovered "
                "Development-local structure across both halves."
            ),
            "scope": (
                "This invalidates speaker-disjoint wording for results selected/evaluated on "
                "the historical within-Development split. It does not invalidate whole-side "
                "Train-to-Development or Development-to-Train evaluations, which do not use "
                "pseudo-speaker sub-splitting."
            ),
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[wrote] {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
