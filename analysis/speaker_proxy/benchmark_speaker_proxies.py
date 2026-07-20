"""Label-free, side-local benchmark of URTIC pseudo-speaker partitions.

The released corpus has no speaker IDs, but the official design states 210
speakers per partition.  This benchmark therefore evaluates cluster stability
within Train, Development and Test separately.  It never loads cold labels.

Questions answered:
  * Is ECAPA+KMeans stable across random seeds?
  * Do fixed-k agglomerative and sparse spectral clustering recover the same
    side-local structure?
  * Does HDBSCAN independently recover about 210 clusters per side?
  * On local nearest-neighbour edges, how often do the fixed-k methods agree
    that two chunks belong to the same speaker proxy?

The resulting clusters remain proxies, not ground truth.  External labeled
corpus recovery summaries are copied into the report as calibration evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    HDBSCAN,
    KMeans,
    SpectralClustering,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
ARCHIVE = ROOT / "cache/ecapa-voxceleb/ecapa_embeddings.npz"
OUT_JSON = ROOT / "results/speaker_proxy_method_benchmark.json"
OUT_LABELS = ROOT / "results/speaker_proxy_method_labels.npz"
SIDES = ("train", "devel", "test")
K = 210
KMEANS_SEEDS = (42, 1, 2, 3, 5, 20260720)
KNN_K = 10


def partition_summary(
    x: np.ndarray,
    labels: np.ndarray,
    neighbour_idx: np.ndarray,
    *,
    silhouette_seed: int = 42,
) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    same = labels[neighbour_idx] == labels[:, None]
    rng = np.random.default_rng(silhouette_seed)
    take = rng.choice(len(x), size=min(3000, len(x)), replace=False)
    silhouette = None
    if len(unique) >= 2 and len(unique) < len(take):
        silhouette = float(silhouette_score(x[take], labels[take], metric="euclidean"))
    return {
        "n_clusters_including_noise": int(len(unique)),
        "noise_fraction": float(np.mean(labels == -1)),
        "cluster_size_min": int(counts.min()),
        "cluster_size_median": float(np.median(counts)),
        "cluster_size_max": int(counts.max()),
        "knn10_same_cluster_fraction": float(same.mean()),
        "top1_same_cluster_fraction": float(same[:, 0].mean()),
        "silhouette_sample": silhouette,
    }


def agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "ARI": float(adjusted_rand_score(a, b)),
        "NMI": float(normalized_mutual_info_score(a, b)),
    }


def pairwise_seed_stability(labels: dict[str, np.ndarray]) -> dict:
    rows = []
    for (name_a, a), (name_b, b) in itertools.combinations(labels.items(), 2):
        rows.append({"a": name_a, "b": name_b, **agreement(a, b)})
    aris = np.asarray([row["ARI"] for row in rows])
    nmis = np.asarray([row["NMI"] for row in rows])
    return {
        "pairs": rows,
        "ARI_mean": float(aris.mean()),
        "ARI_min": float(aris.min()),
        "ARI_max": float(aris.max()),
        "NMI_mean": float(nmis.mean()),
        "NMI_min": float(nmis.min()),
        "NMI_max": float(nmis.max()),
    }


def neighbour_consensus(
    method_labels: dict[str, np.ndarray], neighbour_idx: np.ndarray
) -> dict:
    names = list(method_labels)
    votes = np.zeros(neighbour_idx.shape, dtype=np.int8)
    for labels in method_labels.values():
        votes += labels[neighbour_idx] == labels[:, None]
    n_methods = len(names)
    hist = {str(v): float(np.mean(votes == v)) for v in range(n_methods + 1)}
    any_same = votes > 0
    return {
        "methods": names,
        "edge_definition": f"directed ECAPA {KNN_K}-nearest-neighbour edges",
        "vote_fraction": hist,
        "unanimous_same_fraction": float(np.mean(votes == n_methods)),
        "majority_same_fraction": float(np.mean(votes >= (n_methods // 2 + 1))),
        "conditional_unanimity_given_any_same": float(
            np.mean(votes[any_same] == n_methods) if np.any(any_same) else 0.0
        ),
    }


def fit_side(x: np.ndarray, side: str, *, skip_spectral: bool) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.time()
    nearest = NearestNeighbors(n_neighbors=KNN_K + 1, metric="euclidean", n_jobs=-1).fit(x)
    neighbour_idx = nearest.kneighbors(x, return_distance=False)[:, 1:]

    kmeans_labels = {}
    for seed in KMEANS_SEEDS:
        print(f"[{side}] KMeans seed={seed}", flush=True)
        kmeans_labels[f"kmeans_seed{seed}"] = KMeans(
            n_clusters=K, n_init=10, random_state=seed
        ).fit_predict(x)
    reference = kmeans_labels["kmeans_seed42"]

    print(f"[{side}] agglomerative average-cosine k={K}", flush=True)
    agg = AgglomerativeClustering(
        n_clusters=K, metric="cosine", linkage="average"
    ).fit_predict(x)

    method_labels = {"kmeans": reference, "agglomerative": agg}
    if not skip_spectral:
        print(f"[{side}] spectral nearest-neighbour k={K}", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spectral = SpectralClustering(
                n_clusters=K,
                affinity="nearest_neighbors",
                n_neighbors=20,
                assign_labels="cluster_qr",
                random_state=42,
                n_jobs=-1,
            ).fit_predict(x)
        method_labels["spectral"] = spectral

    print(f"[{side}] HDBSCAN raw-L2", flush=True)
    hdb = HDBSCAN(min_cluster_size=5, metric="euclidean", n_jobs=-1).fit_predict(x)

    summaries = {
        name: partition_summary(x, labels, neighbour_idx)
        for name, labels in {**method_labels, "hdbscan": hdb}.items()
    }
    comparison = {
        name: agreement(reference, labels)
        for name, labels in {**method_labels, "hdbscan": hdb}.items()
        if name != "kmeans"
    }
    report = {
        "n_chunks": int(len(x)),
        "embedding_dim": int(x.shape[1]),
        "k_prior": K,
        "kmeans_seed_stability": pairwise_seed_stability(kmeans_labels),
        "method_summaries": summaries,
        "agreement_vs_kmeans_seed42": comparison,
        "fixed_k_neighbour_consensus": neighbour_consensus(method_labels, neighbour_idx),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    return report, {**method_labels, "hdbscan": hdb}


def external_calibration() -> dict:
    output = {}
    for name in ("libri_en_dev", "mls_de", "mls_de_small"):
        path = ROOT / "results" / f"ecapa_recovery_{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        true_k = next(
            (row for key, row in data["k_sweep"].items() if key.endswith("true_k")), None
        )
        output[name] = true_k
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-spectral", action="store_true",
        help="run the faster KMeans/AHC/HDBSCAN benchmark only",
    )
    parser.add_argument(
        "--spectral-only", action="store_true",
        help="add spectral results to an existing --skip-spectral artifact",
    )
    args = parser.parse_args()

    archive = np.load(ARCHIVE, allow_pickle=True)
    stems = archive["stems"].astype(str)
    split = archive["split"].astype(str)
    embeddings = normalize(archive["embeddings"].astype(np.float32), axis=1).astype(np.float32)

    if args.spectral_only:
        if not OUT_JSON.exists() or not OUT_LABELS.exists():
            raise SystemExit("--spectral-only requires an existing benchmark JSON and labels NPZ")
        report = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        saved = np.load(OUT_LABELS, allow_pickle=True)
        arrays = {name: saved[name] for name in saved.files}
        for side in SIDES:
            take = split == side
            x = embeddings[take]
            print(f"[{side}] spectral nearest-neighbour k={K}", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectral = SpectralClustering(
                    n_clusters=K,
                    affinity="nearest_neighbors",
                    n_neighbors=20,
                    assign_labels="cluster_qr",
                    random_state=42,
                    n_jobs=-1,
                ).fit_predict(x)
            nearest = NearestNeighbors(
                n_neighbors=KNN_K + 1, metric="euclidean", n_jobs=-1
            ).fit(x)
            neighbour_idx = nearest.kneighbors(x, return_distance=False)[:, 1:]
            kmeans = arrays[f"{side}__kmeans"]
            agg = arrays[f"{side}__agglomerative"]
            report["sides"][side]["method_summaries"]["spectral"] = partition_summary(
                x, spectral, neighbour_idx
            )
            report["sides"][side]["agreement_vs_kmeans_seed42"]["spectral"] = agreement(
                kmeans, spectral
            )
            report["sides"][side]["fixed_k_neighbour_consensus"] = neighbour_consensus(
                {"kmeans": kmeans, "agglomerative": agg, "spectral": spectral},
                neighbour_idx,
            )
            arrays[f"{side}__spectral"] = spectral.astype(np.int32)
        cross_method_ari = [
            item["ARI"]
            for side in SIDES
            for name, item in report["sides"][side]["agreement_vs_kmeans_seed42"].items()
            if name != "hdbscan"
        ]
        report["headline"]["mean_fixed_k_cross_method_ARI"] = float(np.mean(cross_method_ari))
        report["protocol"]["spectral_skipped"] = False
        OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(OUT_LABELS, **arrays)
        print(json.dumps({side: {
            "spectral_vs_kmeans": report["sides"][side]["agreement_vs_kmeans_seed42"]["spectral"],
            "consensus": report["sides"][side]["fixed_k_neighbour_consensus"],
        } for side in SIDES}, indent=2))
        print(f"[updated] {OUT_JSON.relative_to(ROOT)} and {OUT_LABELS.relative_to(ROOT)}")
        return

    all_reports = {}; arrays = {}
    for side in SIDES:
        take = split == side
        print(f"\n=== {side}: {int(take.sum())} chunks ===", flush=True)
        report, labels = fit_side(embeddings[take], side, skip_spectral=args.skip_spectral)
        all_reports[side] = report
        arrays[f"{side}__stems"] = stems[take]
        for name, values in labels.items():
            arrays[f"{side}__{name}"] = values.astype(np.int32)

    seed_ari = [row["kmeans_seed_stability"]["ARI_mean"] for row in all_reports.values()]
    fixed_method_ari = []
    for row in all_reports.values():
        fixed_method_ari.extend(
            item["ARI"] for name, item in row["agreement_vs_kmeans_seed42"].items()
            if name != "hdbscan"
        )
    report = {
        "question": "How reproducible are ECAPA pseudo-speaker partitions within each official side?",
        "protocol": {
            "cold_labels_loaded": False,
            "clustering_scope": "each official side independently",
            "known_speaker_count_prior_per_side": K,
            "embedding": "SpeechBrain ECAPA-VoxCeleb 192d, L2 normalized",
            "kmeans_seeds": list(KMEANS_SEEDS),
            "spectral_skipped": bool(args.skip_spectral),
        },
        "external_labeled_corpus_calibration": external_calibration(),
        "sides": all_reports,
        "headline": {
            "mean_kmeans_pairwise_seed_ARI_across_sides": float(np.mean(seed_ari)),
            "minimum_side_kmeans_pairwise_seed_ARI": float(min(
                row["kmeans_seed_stability"]["ARI_min"] for row in all_reports.values()
            )),
            "mean_fixed_k_cross_method_ARI": float(np.mean(fixed_method_ari)),
        },
        "limitations": [
            "URTIC true speaker IDs are unavailable, so agreement is not accuracy.",
            "HDBSCAN chooses its own granularity and should not be expected to match fixed k exactly.",
            "All methods use the same ECAPA representation; a second embedding family is still required for cross-view validation.",
            "Cold labels are deliberately excluded and must not be used to choose a speaker partition.",
        ],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(OUT_LABELS, **arrays)
    print(json.dumps({"headline": report["headline"], "sides": {
        side: {
            "seed_ARI": row["kmeans_seed_stability"]["ARI_mean"],
            "vs_kmeans": row["agreement_vs_kmeans_seed42"],
            "hdbscan_clusters": row["method_summaries"]["hdbscan"]["n_clusters_including_noise"],
            "consensus": row["fixed_k_neighbour_consensus"],
        } for side, row in all_reports.items()
    }}, indent=2))
    print(f"[wrote] {OUT_JSON.relative_to(ROOT)} and {OUT_LABELS.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
