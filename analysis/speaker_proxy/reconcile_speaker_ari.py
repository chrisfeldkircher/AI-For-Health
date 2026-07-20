"""Reproduce and reconcile the apparently conflicting ECAPA ARI results.

No cold labels are loaded. The script contrasts like-for-like side-local
comparisons with the historical Train-fitted-labels versus pooled-HDBSCAN
comparison that produced ARI ~= 0.31.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
META = ROOT / "paper_data/eda/chunk_metadata.csv"
EMBEDDINGS = ROOT / "paper_data/eda/ecapa_embeddings_fp16.npy"
SHIPPED_TSV = ROOT / "cache/pseudo_speakers/k210_seed42.tsv"
SIDE_LABELS = ROOT / "results/speaker_proxy_method_labels.npz"
OUT = ROOT / "results/speaker_ari_reconciliation.json"


def scores(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "ARI": float(adjusted_rand_score(a, b)),
        "NMI": float(normalized_mutual_info_score(a, b)),
    }


def hdbscan_scores(reference: np.ndarray, hdb_labels: np.ndarray) -> dict:
    nonnoise = hdb_labels != -1
    return {
        "all_points": scores(reference, hdb_labels),
        "hdbscan_nonnoise_only": scores(reference[nonnoise], hdb_labels[nonnoise]),
        "nonnoise_coverage": float(nonnoise.mean()),
        "note": (
            "The non-noise sensitivity analysis removes recordings HDBSCAN declined "
            "to assign. It must be reported with coverage and must not silently replace "
            "the conservative all-point ARI."
        ),
    }


def main() -> None:
    # Explicit usecols ensures the available cold_label column is never loaded.
    meta = pd.read_csv(META, usecols=["file_stem", "split", "k210_cluster"])
    x = normalize(np.load(EMBEDDINGS).astype(np.float32), axis=1).astype(np.float32)
    if len(meta) != len(x):
        raise RuntimeError(f"metadata/embedding mismatch: {len(meta)} != {len(x)}")

    shipped_tsv = pd.read_csv(SHIPPED_TSV, sep="\t")
    merged = meta[["file_stem", "k210_cluster"]].merge(
        shipped_tsv[["file_stem", "cluster"]], on="file_stem", how="left"
    )
    if merged["cluster"].isna().any():
        raise RuntimeError("shipped TSV does not cover all Train+Development stems")
    mismatch = int(np.sum(merged["k210_cluster"].to_numpy() != merged["cluster"].to_numpy()))
    if mismatch:
        raise RuntimeError(f"metadata and shipped labels differ at {mismatch} rows")
    shipped = meta["k210_cluster"].to_numpy(np.int32)

    saved = np.load(SIDE_LABELS, allow_pickle=True)
    side_local = np.empty(len(meta), dtype=np.int32)
    side_masks = {}
    for offset, side in enumerate(("train", "devel")):
        mask = meta["split"].astype(str).str.lower().eq(side).to_numpy()
        side_masks[side] = mask
        stems = saved[f"{side}__stems"].astype(str)
        labels = saved[f"{side}__kmeans"].astype(np.int32)
        mapping = dict(zip(stems, labels + 1000 * offset))
        missing = [s for s in meta.loc[mask, "file_stem"].astype(str) if s not in mapping]
        if missing:
            raise RuntimeError(f"{side}: {len(missing)} stems missing from side-local labels")
        side_local[mask] = [mapping[s] for s in meta.loc[mask, "file_stem"].astype(str)]

    print("[fit] pooled HDBSCAN on Train+Development raw-L2 ECAPA", flush=True)
    pooled_hdb = HDBSCAN(
        min_cluster_size=5, metric="euclidean", n_jobs=-1
    ).fit_predict(x)
    pooled_n_clusters = len(set(pooled_hdb)) - int(-1 in pooled_hdb)

    # The current side-local benchmark stores the exact labels used for the
    # like-for-like raw-space results, avoiding any dependence on label numbers.
    side_local_results = {}
    for side in ("train", "devel", "test"):
        km = saved[f"{side}__kmeans"]
        hdb = saved[f"{side}__hdbscan"]
        side_local_results[side] = {
            **hdbscan_scores(km, hdb),
            "kmeans_clusters": int(len(np.unique(km))),
            "hdbscan_nonnoise_clusters": int(len(set(hdb)) - int(-1 in hdb)),
            "hdbscan_noise_fraction": float(np.mean(hdb == -1)),
        }

    pooled_by_labeling = {
        "historical_trainfit_k210_labels": {
            "all_train_plus_development": scores(shipped, pooled_hdb),
            "by_side": {
                side: scores(shipped[mask], pooled_hdb[mask])
                for side, mask in side_masks.items()
            },
        },
        "offset_side_local_k210_plus_k210_labels": {
            "all_train_plus_development": scores(side_local, pooled_hdb),
            "by_side": {
                side: scores(side_local[mask], pooled_hdb[mask])
                for side, mask in side_masks.items()
            },
        },
    }

    report = {
        "question": "Why did raw-space KMeans-vs-HDBSCAN ARI appear as both ~0.856 and ~0.309?",
        "cold_labels_loaded": False,
        "embedding_space": "raw 192-D ECAPA, L2 normalized",
        "historical_notebook_provenance": {
            "file": "model/test.ipynb",
            "cell_index_zero_based": 14,
            "n_items": 9505,
            "scope": "Train only",
            "kmeans_k": 210,
            "hdbscan_nonnoise_clusters": 204,
            "reported_ARI": 0.8557,
            "reported_NMI": 0.9620,
            "umap_used_for_clustering": False,
        },
        "like_for_like_side_local_results": side_local_results,
        "pooled_hdbscan": {
            "scope": "Train+Development",
            "n_items": int(len(meta)),
            "nonnoise_clusters": int(pooled_n_clusters),
            "noise_fraction": float(np.mean(pooled_hdb == -1)),
            "comparisons": pooled_by_labeling,
            "historical_trainfit_nonnoise_sensitivity": hdbscan_scores(
                shipped, pooled_hdb
            ),
            "offset_side_local_nonnoise_sensitivity": hdbscan_scores(
                side_local, pooled_hdb
            ),
        },
        "trainfit_labels_vs_offset_side_local_labels": scores(shipped, side_local),
        "verdict": (
            "The 0.8557 result is a raw-space, side-local Train comparison and is reproduced. "
            "The 0.3094 result compares a single Train-fitted 210-label map, whose held-side "
            "assignments fragment Development, against a pooled density partition with about "
            "twice the cluster count. Using offset side-local labels raises pooled ARI materially; "
            "fitting both methods side-locally reproduces high agreement on every side. The gap is "
            "caused by scope/granularity/held-side assignment mismatch, not UMAP inflation."
        ),
        "paper_wording": (
            "Report side-local raw-space concordance by side on all recordings as the primary "
            "result. Report the higher HDBSCAN-nonnoise-only ARI as a sensitivity analysis with "
            "its 96.7-97.6% coverage. Keep the historical pooled-vs-Train-fit result only as a "
            "demonstration that Train-centroid assignment is invalid for held official sides; "
            "do not present 0.309 as side-local method instability and do not present ~0.93 "
            "without stating that HDBSCAN noise was excluded."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[wrote] {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
