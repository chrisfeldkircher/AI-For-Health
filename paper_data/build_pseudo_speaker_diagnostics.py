"""Raw-L2 ECAPA pseudo-speaker validation -> results/pseudo_speaker_ecapa_diagnostics.json

Validates the SHIPPED k=210 pseudo-speaker labels (the exact ones used for every
grouped split / leakage audit; provenance verified: chunk_metadata.csv k210 ==
cache/pseudo_speakers/k210_seed42.tsv, 0/19101 mismatch) in the SAME space the
production KMeans used: L2-normalized full-192-D ECAPA (model/speakers/cluster.py
does `normalize(X, axis=1)` then KMeans). We do NOT refit KMeans and we do NOT use
the UMAP-32 space (model/speakers/diagnostics.py's own docstring flags that as
"optimistic: UMAP enhances separability; silhouette/cohesion inflated").

Metrics (all in raw-L2 192-D):
  - Primary: HDBSCAN (independent, parameter-free-ish) vs shipped labels: ARI/NMI,
    plus HDBSCAN n_clusters + noise_frac. Pre-registered degeneracy rule:
    noise_frac > 0.5 OR n_clusters < 50  -> instrument limitation, NOT proxy failure.
  - kNN label cohesion on the shipped labels directly (no second clustering; the
    most honest single number): fraction of each chunk's cosine-NN sharing its
    pseudo-speaker, vs the chance baseline sum(p_c^2).
  - Agglomerative @ k=210 (cosine/average) on a seeded subsample: parameter-matched
    independent method, no degeneracy mode -> ARI/NMI corroboration.
  - Silhouette of the shipped labels (subsample), explicitly RELATIVE
    (euclidean on L2-normalized full-D), not an absolute quality claim.

Deterministic, CPU-only, no model training. Run with the datascience env.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
)

ROOT = Path(__file__).resolve().parent.parent
EDA = ROOT / "paper_data" / "eda"
TSV = ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv"
OUT = ROOT / "results" / "pseudo_speaker_ecapa_diagnostics.json"
SEED = 42
HDBSCAN_MIN_CLUSTER = 5          # matches model/speakers/diagnostics.py default
DEGEN_NOISE_FRAC = 0.5           # pre-registered
DEGEN_MIN_NCLUSTERS = 50         # pre-registered
KNN_K = 10
AGGLO_SUBSAMPLE = 8000
SIL_SUBSAMPLE = 5000

print(f"[load] {EDA/'ecapa_embeddings_fp16.npy'}  +  {EDA/'chunk_metadata.csv'}")
ecapa = np.load(EDA / "ecapa_embeddings_fp16.npy").astype(np.float32)   # (N,192)
meta = pd.read_csv(EDA / "chunk_metadata.csv")
assert len(meta) == ecapa.shape[0] == 19101, (len(meta), ecapa.shape)

# provenance re-assert (cheap, in-artifact record)
tsv = pd.read_csv(TSV, sep="\t")
mrg = meta[["file_stem", "k210_cluster"]].merge(
    tsv[["file_stem", "cluster"]], on="file_stem", how="inner")
n_mismatch = int((mrg.k210_cluster != mrg.cluster).sum())
assert n_mismatch == 0, f"provenance mismatch {n_mismatch}"
y = meta["k210_cluster"].to_numpy()
k = int(meta["k210_cluster"].nunique())

# L2-normalize, exactly as production cluster.py does (normalize(X, axis=1))
X = normalize(ecapa, axis=1).astype(np.float32)
N = X.shape[0]
print(f"[space] raw-L2 192-D  N={N}  shipped k={k}  (provenance: 0/{N} mismatch)")

# cluster-proportion chance baseline for same-label-among-neighbors
_, counts = np.unique(y, return_counts=True)
p = counts / counts.sum()
chance_same = float((p ** 2).sum())

rng = np.random.default_rng(SEED)

# --- Primary: HDBSCAN on raw-L2 (euclidean on L2-normalized == cosine geometry) ---
print("[hdbscan] fitting on raw-L2 192-D (this is the high-D stress test) ...")
hdb = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER, metric="euclidean",
              n_jobs=-1).fit(X)
hl = hdb.labels_
hdb_n_clusters = int(len(set(hl)) - (1 if -1 in hl else 0))
hdb_noise_frac = float((hl == -1).mean())
hdb_ari = float(adjusted_rand_score(y, hl))
hdb_nmi = float(normalized_mutual_info_score(y, hl))
# also ARI/NMI on non-noise points only (HDBSCAN noise excluded)
nz = hl != -1
hdb_ari_nz = float(adjusted_rand_score(y[nz], hl[nz])) if nz.sum() > 1 else None
hdb_nmi_nz = float(normalized_mutual_info_score(y[nz], hl[nz])) if nz.sum() > 1 else None
degenerate = (hdb_noise_frac > DEGEN_NOISE_FRAC) or (hdb_n_clusters < DEGEN_MIN_NCLUSTERS)
print(f"  HDBSCAN n_clusters={hdb_n_clusters}  noise_frac={hdb_noise_frac:.3f}  "
      f"ARI={hdb_ari:.3f}  NMI={hdb_nmi:.3f}  -> "
      f"{'DEGENERATE (instrument limitation, not proxy failure)' if degenerate else 'usable'}")

# --- kNN label cohesion on shipped labels directly (load-bearing honest metric) ---
print(f"[knn-cohesion] cosine, k={KNN_K}, all {N} points ...")
nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="cosine", n_jobs=-1).fit(X)
_, idx = nn.kneighbors(X)
idx = idx[:, 1:]                                  # drop self
same = (y[idx] == y[:, None]).mean(axis=1)
knn_cohesion_mean = float(same.mean())
knn_cohesion_std = float(same.std())
knn_lift_over_chance = knn_cohesion_mean / chance_same
print(f"  mean same-pseudo-speaker among {KNN_K}-NN = {knn_cohesion_mean:.4f} "
      f"(chance {chance_same:.4f}, {knn_lift_over_chance:.1f}x)")

# --- Agglomerative @ k=210 on a seeded subsample (parameter-matched, no degeneracy) ---
sub = rng.choice(N, size=min(AGGLO_SUBSAMPLE, N), replace=False)
print(f"[agglomerative] k={k}, cosine/average, subsample n={len(sub)} ...")
agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
al = agg.fit_predict(X[sub])
agg_ari = float(adjusted_rand_score(y[sub], al))
agg_nmi = float(normalized_mutual_info_score(y[sub], al))
print(f"  Agglomerative ARI={agg_ari:.3f}  NMI={agg_nmi:.3f} (vs shipped, on subsample)")

# --- Silhouette of shipped labels (subsample), RELATIVE only ---
ssub = rng.choice(N, size=min(SIL_SUBSAMPLE, N), replace=False)
sil = float(silhouette_score(X[ssub], y[ssub], metric="euclidean"))
print(f"[silhouette] shipped-label silhouette (raw-L2, n={len(ssub)}) = {sil:.3f}  "
      f"[RELATIVE: euclidean on L2-normalized full-D, not an absolute quality claim]")

# --- branch decision (per the pre-registered plan) ---
strong = (not degenerate and hdb_ari >= 0.5) or (agg_ari >= 0.5) or (knn_lift_over_chance >= 20)
branch = "validated_proxy" if strong else "best_available_proxy"
print(f"\n[branch] pseudo-speaker narrative branch = {branch}")

out = {
    "rung_id": "pseudo_speaker_ecapa_diagnostics",
    "description": (
        "Raw-L2 (L2-normalized full-192-D ECAPA, the production cluster.py space) "
        "validation of the SHIPPED k=210 pseudo-speaker labels used for every "
        "grouped split. NOT refit KMeans; NOT UMAP-32 (which cluster diagnostics.py "
        "flags as optimistic/inflated). HDBSCAN is the independent stress test with a "
        "pre-registered degeneracy rule; kNN label cohesion validates the shipped "
        "labels directly with no second clustering; agglomerative@k is a "
        "parameter-matched corroboration; silhouette is reported RELATIVE only."
    ),
    "provenance": {
        "shipped_label_source": "cache/pseudo_speakers/k210_seed42.tsv",
        "validated_via": "paper_data/eda/chunk_metadata.csv k210_cluster",
        "exact_match": True, "n_mismatch": n_mismatch, "n_chunks": N,
        "note": "labels validated are byte-exact the experiment-used labels",
    },
    "space": "raw-L2 (L2-normalized full 192-D ECAPA; matches model/speakers/cluster.py)",
    "k": k,
    "hdbscan": {
        "min_cluster_size": HDBSCAN_MIN_CLUSTER, "metric": "euclidean_on_L2norm",
        "n_clusters": hdb_n_clusters, "noise_frac": hdb_noise_frac,
        "ari_vs_shipped": hdb_ari, "nmi_vs_shipped": hdb_nmi,
        "ari_vs_shipped_excl_noise": hdb_ari_nz, "nmi_vs_shipped_excl_noise": hdb_nmi_nz,
        "degenerate": bool(degenerate),
        "degeneracy_rule": f"noise_frac>{DEGEN_NOISE_FRAC} OR n_clusters<{DEGEN_MIN_NCLUSTERS}",
        "interpretation": (
            "degenerate => high-D HDBSCAN instrument limitation, NOT proxy failure; "
            "lean on kNN-cohesion + agglomerative" if degenerate else
            "HDBSCAN usable as independent concordance evidence"),
    },
    "knn_label_cohesion": {
        "k": KNN_K, "metric": "cosine",
        "mean_same_pseudo_speaker": knn_cohesion_mean,
        "std": knn_cohesion_std,
        "chance_baseline_sum_p2": chance_same,
        "lift_over_chance": knn_lift_over_chance,
        "note": "load-bearing: validates the SHIPPED labels directly, no 2nd clustering",
    },
    "agglomerative_k_matched": {
        "k": k, "metric": "cosine", "linkage": "average",
        "subsample_n": int(len(sub)),
        "ari_vs_shipped": agg_ari, "nmi_vs_shipped": agg_nmi,
    },
    "silhouette_shipped_labels_RELATIVE": {
        "value": sil, "subsample_n": int(len(ssub)),
        "caveat": "euclidean on L2-normalized full-D; RELATIVE, not absolute quality",
    },
    "narrative_branch": branch,
    "branch_rule": (
        "validated_proxy if (HDBSCAN non-degenerate AND ARI>=0.5) OR agglo ARI>=0.5 "
        "OR kNN lift>=20x; else best_available_proxy (leakage-audit then framed on "
        "ECAPA being an independent speaker-verification substrate + functional "
        "leakage reduction, not near-perfect cluster recovery)"),
}
OUT.write_text(json.dumps(out, indent=2))
print(f"[wrote] {OUT.relative_to(ROOT)}")
