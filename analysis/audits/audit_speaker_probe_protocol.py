"""Audit whether the current speaker-probe protocol measures speaker leakage.

The production probe is trained on URTIC train chunks and evaluated on the
official devel split.  URTIC train and devel contain different speakers, while
``k210_seed42.tsv`` gives devel chunks the ID of their nearest *train* KMeans
centroid.  Consequently, cross-pool top-1 is not a conventional speaker-ID
probe: the probe never saw the devel identities it is asked to recognize.

This script extracts the frozen A2.5 representation and compares two otherwise
identical nearest-centroid probes:

1. held-chunk / same-identity: fit on 80% of each train pseudo-speaker and test
   on the remaining 20%; every evaluation identity was seen by the probe;
2. cross-official-pool: fit on the same train subset and evaluate on devel,
   reproducing the semantic mismatch in the existing honesty gate.

It writes a small JSON artifact so the result can be cited without rerunning
the GPU extraction.  This is a diagnostic only; it does not modify training or
submission artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, normalize


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
MODEL_ROOT = ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from data.cached_dataset import PooledCacheDataset  # noqa: E402
from features import LayerWeightedPooledHead  # noqa: E402
from speakers.cluster import load_pseudo_speakers  # noqa: E402
from speakers.probe import extract_z  # noqa: E402


def _align(names: list[str], pseudo: dict[str, int]) -> np.ndarray:
    return np.asarray([pseudo[Path(name).stem] for name in names], dtype=np.int64)


def _per_identity_holdout(
    labels: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return fit/eval indices with every identity represented on both sides."""
    rng = np.random.default_rng(seed)
    fit, evaluation = [], []
    for identity in np.unique(labels):
        idx = np.flatnonzero(labels == identity)
        rng.shuffle(idx)
        n_eval = max(1, int(round(len(idx) * fraction)))
        n_eval = min(n_eval, len(idx) - 1)
        if n_eval < 1:
            raise ValueError(f"identity {identity} has fewer than two chunks")
        evaluation.extend(idx[:n_eval])
        fit.extend(idx[n_eval:])
    return np.asarray(sorted(fit)), np.asarray(sorted(evaluation))


def _centroid_probe(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float | int]:
    """Standardized cosine nearest-centroid identity probe."""
    scaler = StandardScaler().fit(x_fit)
    x_fit_n = normalize(scaler.transform(x_fit), axis=1)
    x_eval_n = normalize(scaler.transform(x_eval), axis=1)
    identities = np.unique(y_fit)
    centroids = np.vstack([x_fit_n[y_fit == identity].mean(0) for identity in identities])
    centroids = normalize(centroids, axis=1)
    similarities = x_eval_n @ centroids.T
    order = np.argsort(-similarities, axis=1)
    pred = identities[order[:, 0]]
    top5 = identities[order[:, : min(5, len(identities))]]

    p_fit = Counter(y_fit.tolist())
    p_eval = Counter(y_eval.tolist())
    n_fit, n_eval = len(y_fit), len(y_eval)
    distribution_chance = sum(
        (p_fit[k] / n_fit) * (p_eval.get(k, 0) / n_eval) for k in p_fit
    )
    return {
        "n_fit": int(n_fit),
        "n_eval": int(n_eval),
        "n_identities_fit": int(len(identities)),
        "n_identities_eval": int(len(np.unique(y_eval))),
        "top1": float(np.mean(pred == y_eval)),
        "top5": float(np.mean(np.any(top5 == y_eval[:, None], axis=1))),
        "nmi": float(normalized_mutual_info_score(y_eval, pred)),
        "uniform_chance_1_over_k": float(1.0 / len(identities)),
        "distribution_matched_chance": float(distribution_chance),
        "majority_baseline": float(max(p_eval.values()) / n_eval),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=0.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "audit_speaker_probe_protocol.json",
    )
    args = parser.parse_args()

    data_dir = ROOT / "dataset" / "ComParE2017_Cold_4students"
    cache_root = ROOT / "cache"
    backbone = "microsoft_wavlm-large"
    pseudo_path = cache_root / "pseudo_speakers" / "k210_seed42.tsv"
    checkpoint = (
        cache_root
        / backbone
        / f"head_A2grouped_honestprior_seed{args.seed}.pt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = PooledCacheDataset(str(data_dir), str(cache_root), backbone, split="train")
    devel_ds = PooledCacheDataset(str(data_dir), str(cache_root), backbone, split="devel")
    sample = train_ds[0]["pooled"]
    head = LayerWeightedPooledHead(
        n_layers=sample.shape[0], stat_dim=sample.shape[1], proj_dim=128,
        n_classes=2, dropout=0.5,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    head.load_state_dict(state["state_dict"])
    head.eval()

    print(f"[extract] A2.5 z on {device}: train={len(train_ds)} devel={len(devel_ds)}")
    z_train, names_train = extract_z(head, train_ds, device=device, batch_size=512)
    z_devel, names_devel = extract_z(head, devel_ds, device=device, batch_size=512)
    pseudo = load_pseudo_speakers(pseudo_path)
    y_train = _align(names_train, pseudo)
    y_devel = _align(names_devel, pseudo)
    fit_idx, eval_idx = _per_identity_holdout(
        y_train, fraction=args.holdout, seed=args.seed
    )

    x_fit = z_train.numpy()[fit_idx]
    y_fit = y_train[fit_idx]
    within = _centroid_probe(x_fit, y_fit, z_train.numpy()[eval_idx], y_train[eval_idx])
    cross = _centroid_probe(x_fit, y_fit, z_devel.numpy(), y_devel)

    ratio = within["top1"] / max(cross["top1"], 1e-12)
    report = {
        "rung_id": "audit_speaker_probe_protocol",
        "hypothesis": (
            "Evaluating a speaker-ID probe on a different official speaker pool, "
            "labeled by nearest train centroid, understates recoverable identity."
        ),
        "representation": f"A2.5 z, seed={args.seed}, dim={z_train.shape[1]}",
        "pseudo_labels": str(pseudo_path.relative_to(ROOT)),
        "probe": "StandardScaler + cosine nearest centroid",
        "held_chunk_same_identity": within,
        "cross_official_pool_nearest_train_centroid_labels": cross,
        "same_identity_to_cross_pool_top1_ratio": float(ratio),
        "protocol_mismatch_confirmed": bool(within["top1"] > cross["top1"] + 0.05),
        "interpretation": (
            "Use held-chunk same-identity classification or pairwise verification "
            "to measure identity leakage. Use group-disjoint folds separately to "
            "measure cold-classifier generalization. One split cannot serve both roles."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[same identity] top1={within['top1']:.4f} top5={within['top5']:.4f} "
          f"chance={within['distribution_matched_chance']:.4f}")
    print(f"[cross pool]    top1={cross['top1']:.4f} top5={cross['top5']:.4f} "
          f"chance={cross['distribution_matched_chance']:.4f}")
    print(f"[ratio] {ratio:.2f}x; mismatch_confirmed={report['protocol_mismatch_confirmed']}")
    print(f"[wrote] {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
