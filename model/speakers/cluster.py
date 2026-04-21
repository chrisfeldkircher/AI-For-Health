"""
Pseudo-speaker clustering over cached ECAPA-TDNN embeddings.

URTIC 4students has no speaker IDs, so we cluster ECAPA embeddings on train
and assign devel/test chunks by nearest centroid. These pseudo-speaker IDs
are the substrate for:
  - the speaker probe (diagnostic — is speaker id recoverable from z?)
  - A5.5 augmentation (same-speaker exclusion in SpliceSpec generation)
  - A6 contrastive speaker-masked loss
  - A7 MDD adversarial head (pseudo-speaker targets)

Design:
  - Fit KMeans on train embeddings only (no devel/test leakage).
  - Sweep k in {100, 210, 420}. URTIC has ~630 speakers total, roughly split
    evenly across train/dev/test, so train likely holds ~210 speakers;
    k=100 probes coarser groupings, k=420 over-clusters as a stress test.
  - L2-normalise before clustering so Euclidean distance ≡ cosine distance.
  - Report silhouette (subsampled), intra/inter distance ratio, and the
    cluster-size distribution.
  - Write one versioned TSV per k: cache/pseudo_speakers/k{K}_seed{S}.tsv.
    Swapping k doesn't invalidate other caches.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .ecapa import load_ecapa_matrix


@dataclass
class ClusterReport:
    k: int
    seed: int
    silhouette: float
    intra_inter_ratio: float
    cluster_size_min: int
    cluster_size_median: int
    cluster_size_max: int
    cluster_size_mean: float
    n_empty_on_devel: int
    tsv_path: Path


def _intra_inter_ratio(
    X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> float:
    """
    intra : mean L2 distance from each point to its assigned centroid.
    inter : mean pairwise L2 distance between distinct centroids.
    ratio : intra / inter, lower = tighter / more separable clustering.
    """
    from scipy.spatial.distance import pdist
    assigned = centroids[labels]
    intra = float(np.linalg.norm(X - assigned, axis=1).mean())
    inter = float(pdist(centroids).mean()) if centroids.shape[0] > 1 else 1.0
    return intra / inter if inter > 0 else float("nan")


def fit_and_assign(
    train_paths: list[Path],
    devel_paths: list[Path],
    test_paths: list[Path],
    ecapa_cache: Path,
    out_dir: Path,
    ks: tuple[int, ...] = (100, 210, 420),
    seed: int = 42,
    silhouette_sample: int = 3000,
) -> list[ClusterReport]:
    """
    Run the k-sweep. For each k: fit on train, assign all splits, write TSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[cluster] loading cached ECAPA embeddings ...")
    X_train, stems_train = load_ecapa_matrix(train_paths, ecapa_cache)
    X_devel, stems_devel = load_ecapa_matrix(devel_paths, ecapa_cache)
    X_test,  stems_test  = load_ecapa_matrix(test_paths,  ecapa_cache)
    print(f"  train: {X_train.shape}   devel: {X_devel.shape}   test: {X_test.shape}")

    X_train = normalize(X_train, axis=1)
    X_devel = normalize(X_devel, axis=1)
    X_test  = normalize(X_test,  axis=1)

    rng = np.random.default_rng(seed)
    reports: list[ClusterReport] = []
    for k in ks:
        print(f"\n[cluster] k={k}  (KMeans, n_init=10) ...")
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        train_labels = km.fit_predict(X_train)
        devel_labels = km.predict(X_devel)
        test_labels  = km.predict(X_test)

        sample_n = min(silhouette_sample, X_train.shape[0])
        sample_idx = rng.choice(X_train.shape[0], size=sample_n, replace=False)
        sil = float(silhouette_score(
            X_train[sample_idx], train_labels[sample_idx], metric="euclidean"
        ))
        ratio = _intra_inter_ratio(X_train, train_labels, km.cluster_centers_)

        _, counts = np.unique(train_labels, return_counts=True)
        size_min    = int(counts.min())
        size_median = int(np.median(counts))
        size_max    = int(counts.max())
        size_mean   = float(counts.mean())
        n_empty_devel = int(k - np.unique(devel_labels).size)

        print(f"  silhouette (n={sample_n}) : {sil:.4f}")
        print(f"  intra/inter ratio        : {ratio:.4f}")
        print(f"  cluster sizes (train)    : min={size_min}  median={size_median}  "
              f"max={size_max}  mean={size_mean:.1f}")
        print(f"  clusters unused on devel : {n_empty_devel} / {k}")

        tsv_path = out_dir / f"k{k}_seed{seed}.tsv"
        with tsv_path.open("w", encoding="utf-8") as f:
            f.write("file_stem\tsplit\tcluster\n")
            for s, c in zip(stems_train, train_labels):
                f.write(f"{s}\ttrain\t{int(c)}\n")
            for s, c in zip(stems_devel, devel_labels):
                f.write(f"{s}\tdevel\t{int(c)}\n")
            for s, c in zip(stems_test, test_labels):
                f.write(f"{s}\ttest\t{int(c)}\n")
        print(f"  wrote {tsv_path}")

        reports.append(ClusterReport(
            k=k, seed=seed,
            silhouette=sil, intra_inter_ratio=ratio,
            cluster_size_min=size_min, cluster_size_median=size_median,
            cluster_size_max=size_max, cluster_size_mean=size_mean,
            n_empty_on_devel=n_empty_devel, tsv_path=tsv_path,
        ))
    return reports


def load_pseudo_speakers(tsv_path: Path) -> dict[str, int]:
    """Load cluster assignments from a versioned TSV, keyed by file_stem."""
    out: dict[str, int] = {}
    with Path(tsv_path).open(encoding="utf-8") as f:
        next(f)
        for line in f:
            stem, _split, cluster = line.strip().split("\t")
            out[stem] = int(cluster)
    return out
