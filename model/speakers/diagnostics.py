"""
Label-free diagnostics for pseudo-speaker embeddings.

There are no ground-truth speaker IDs on URTIC 4students, so we evaluate
clusterability of the embedding space via:

  1. UMAP(2) for visual inspection (returned as an array; plotting is the
     caller's responsibility so this module has no matplotlib dependency).
  2. UMAP(d_reduce) → KMeans(k) vs HDBSCAN, with ARI / NMI between the two
     partitions as a stability signal. Agreement means both algorithms
     recover the same structure; disagreement means the space is either
     genuinely continuous or noisy.
  3. kNN label cohesion on the KMeans partition in the reduced space:
     mean fraction of a point's k nearest neighbours that share its cluster.
     A high value (≳0.8) means local neighbourhoods respect the partition,
     which is the property downstream consumers actually need (same-speaker
     exclusion, masked contrastive loss, adversarial head).

Inputs are L2-normalised inside `diagnose_embeddings` so Euclidean distance
≡ cosine distance, matching the existing `cluster.py` convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass
class DiagnosticReport:
    name: str
    space: str                       # "raw-L2" or "umap{d}"
    n_samples: int
    dim_in: int
    dim_cluster: int                 # dim of the space clustering ran in
    kmeans_k: int
    hdbscan_n_clusters: int
    hdbscan_noise_frac: float
    ari_kmeans_vs_hdbscan: float
    nmi_kmeans_vs_hdbscan: float
    knn_cohesion_mean: float
    silhouette_kmeans: float
    silhouette_hdbscan: float        # NaN if HDBSCAN finds <2 clusters
    umap2: np.ndarray                # [N, 2], always computed for viz
    kmeans_labels: np.ndarray        # [N]
    hdbscan_labels: np.ndarray       # [N], -1 = noise


def _knn_cohesion(X: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    For each point, fraction of its k nearest neighbours (excluding itself)
    that share its cluster label. Returned as the mean over all points.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    neighbours = idx[:, 1:]                          # drop self
    same = labels[neighbours] == labels[:, None]     # [N, k]
    return float(same.mean())


def diagnose_embeddings(
    X: np.ndarray,
    name: str,
    kmeans_k: int = 210,
    umap_dim: int | None = 32,
    knn_k: int = 10,
    hdbscan_min_cluster_size: int = 5,
    silhouette_sample: int = 3000,
    seed: int = 42,
) -> DiagnosticReport:
    """
    Run the full diagnostic bundle on an embedding matrix.

    X is assumed to be raw (un-normalised) embeddings; we L2-normalise
    internally so Euclidean ≡ cosine, matching the `cluster.py` convention.

    `umap_dim`:
      - int  → cluster in UMAP(umap_dim) space (optimistic: UMAP is designed
        to enhance local separability, so silhouette/cohesion are inflated).
      - None → cluster in the raw L2-normalised space. More rigorous.
    UMAP(2) is always computed for the scatter plot.
    """
    import umap  # local import: optional dependency

    n = X.shape[0]
    dim_in = X.shape[1]

    Xn = normalize(X, axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)

    reducer2 = umap.UMAP(
        n_components=2, metric="cosine", random_state=seed, n_jobs=1,
    )
    X2 = reducer2.fit_transform(Xn)

    if umap_dim is None:
        Xc = Xn
        space = "raw-L2"
    else:
        reducer_d = umap.UMAP(
            n_components=umap_dim, metric="cosine", random_state=seed, n_jobs=1,
        )
        Xc = reducer_d.fit_transform(Xn)
        space = f"umap{umap_dim}"

    km = KMeans(n_clusters=kmeans_k, n_init=10, random_state=seed)
    km_labels = km.fit_predict(Xc)

    hdb = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric="euclidean",
    )
    hdb_labels = hdb.fit_predict(Xc)
    n_noise = int((hdb_labels == -1).sum())
    n_clusters = int(len({l for l in hdb_labels if l != -1}))
    noise_frac = n_noise / n

    ari = float(adjusted_rand_score(km_labels, hdb_labels))
    nmi = float(normalized_mutual_info_score(km_labels, hdb_labels))

    cohesion = _knn_cohesion(Xc, km_labels, k=knn_k)

    sample_n = min(silhouette_sample, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    sil_km = float(silhouette_score(
        Xc[sample_idx], km_labels[sample_idx], metric="euclidean"
    ))
    if n_clusters >= 2:
        mask = hdb_labels != -1
        if mask.sum() >= 2 and len(set(hdb_labels[mask])) >= 2:
            sub_idx = rng.choice(
                np.where(mask)[0],
                size=min(silhouette_sample, int(mask.sum())),
                replace=False,
            )
            sil_hdb = float(silhouette_score(
                Xc[sub_idx], hdb_labels[sub_idx], metric="euclidean"
            ))
        else:
            sil_hdb = float("nan")
    else:
        sil_hdb = float("nan")

    return DiagnosticReport(
        name=name,
        space=space,
        n_samples=n,
        dim_in=dim_in,
        dim_cluster=Xc.shape[1],
        kmeans_k=kmeans_k,
        hdbscan_n_clusters=n_clusters,
        hdbscan_noise_frac=noise_frac,
        ari_kmeans_vs_hdbscan=ari,
        nmi_kmeans_vs_hdbscan=nmi,
        knn_cohesion_mean=cohesion,
        silhouette_kmeans=sil_km,
        silhouette_hdbscan=sil_hdb,
        umap2=X2,
        kmeans_labels=km_labels,
        hdbscan_labels=hdb_labels,
    )


def print_report(r: DiagnosticReport) -> None:
    print(f"=== {r.name}  (N={r.n_samples}, dim_in={r.dim_in}, cluster space={r.space}/{r.dim_cluster}d) ===")
    print(f"  KMeans k={r.kmeans_k:<4d} silhouette     : {r.silhouette_kmeans:+.4f}")
    print(f"  HDBSCAN clusters    : {r.hdbscan_n_clusters}")
    print(f"  HDBSCAN noise frac  : {r.hdbscan_noise_frac:.3f}")
    print(f"  HDBSCAN silhouette  : {r.silhouette_hdbscan:+.4f}")
    print(f"  ARI  KMeans vs HDB  : {r.ari_kmeans_vs_hdbscan:+.4f}")
    print(f"  NMI  KMeans vs HDB  : {r.nmi_kmeans_vs_hdbscan:+.4f}")
    print(f"  kNN cohesion (k=10) : {r.knn_cohesion_mean:.4f}")
