"""
Adjudicate the ARI 0.31 vs 0.877 conflict for the ECAPA pseudo-speaker proxy.

Reviewer: 0.309 is the honest raw-space cross-method number; validation overstated.
New agent: 0.309 was an artifact of comparing the POOLED/fragmented shipped k210
           (train-only fit, devel assigned by nearest train centroid) against a
           cohesive clustering; real SIDE-LOCAL cross-method ARI is ~0.877.

Test, all in raw L2-normalized ECAPA space, no cold labels:
  A. one side-local KMeans seed pair (42 vs 7; the six-seed benchmark is the
     comprehensive stability result)
  B. side-local cross-method ARI KMeans(210) vs HDBSCAN, both on all points and
     as a sensitivity analysis excluding HDBSCAN noise
  C. HDBSCAN non-noise cluster count per side (~204-205 versus the chosen
     k=210 prior; URTIC does not provide ground-truth side-level speaker IDs)
  D. reproduce the ~0.31: shipped-k210 vs side-local HDBSCAN, TRAIN vs DEVEL
     -- if train is high and devel is low, the 0.31 was the fragmentation artifact
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from speakers.cluster import load_pseudo_speakers

NPZ = ROOT / "cache" / "ecapa-voxceleb" / "ecapa_embeddings.npz"
K = 210

d = np.load(NPZ, allow_pickle=True)
stems = d["stems"].astype(str); split = d["split"].astype(str); emb = d["embeddings"].astype(np.float32)
shipped = load_pseudo_speakers(ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv")


def ari_nonoise(a, b):
    """ARI on points where b (HDBSCAN) is non-noise."""
    m = b >= 0
    return adjusted_rand_score(a[m], b[m]), float(m.mean())


print("=" * 82)
print("SPEAKER-PROXY ARI ADJUDICATION (raw L2 ECAPA space, side-local, no cold labels)")
print("=" * 82)
print(f"{'side':<8} {'seed42v7':>9} {'KM-HDBall':>10} {'KM-HDBnz':>9} {'HDBk':>6} "
      f"{'noise%':>7} {'ship-KM':>8} {'ship-HDBall':>11} {'ship-HDBnz':>10}")
print("-" * 82)
rows = {}
for sp in ("train", "devel", "test"):
    m = split == sp
    X = normalize(emb[m], axis=1)
    ss = stems[m]
    shp = np.array([shipped[s] for s in ss])
    km_a = KMeans(K, n_init=10, random_state=42).fit_predict(X)
    km_b = KMeans(K, n_init=10, random_state=7).fit_predict(X)
    hdb = HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(X)
    seed_ari = adjusted_rand_score(km_a, km_b)
    km_hdb_all = adjusted_rand_score(km_a, hdb)
    km_hdb, cov = ari_nonoise(km_a, hdb)
    hdb_k = int(len(set(hdb[hdb >= 0])))
    noise = float((hdb < 0).mean())
    ship_km = adjusted_rand_score(shp, km_a)
    ship_hdb_all = adjusted_rand_score(shp, hdb)
    ship_hdb, _ = ari_nonoise(shp, hdb)
    rows[sp] = dict(seed_ari=seed_ari, km_hdb_all=km_hdb_all, km_hdb=km_hdb,
                    hdb_k=hdb_k, noise=noise, ship_km=ship_km,
                    ship_hdb_all=ship_hdb_all, ship_hdb=ship_hdb)
    print(f"{sp:<8} {seed_ari:>9.3f} {km_hdb_all:>10.3f} {km_hdb:>9.3f} "
          f"{hdb_k:>6d} {noise*100:>6.1f}% {ship_km:>8.3f} "
          f"{ship_hdb_all:>11.3f} {ship_hdb:>10.3f}")
print("-" * 82)

# D': the actual pooled-vs-sidelocal mismatch that likely produced 0.31
mtd = np.isin(split, ["train", "devel"])
Xpool = normalize(emb[mtd], axis=1)
sp_pool = stems[mtd]
shp_pool = np.array([shipped[s] for s in sp_pool])
hdb_pool = HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(Xpool)
ship_hdb_pool, cov_pool = ari_nonoise(shp_pool, hdb_pool)
ship_hdb_pool_all = adjusted_rand_score(shp_pool, hdb_pool)
print(f"POOLED shipped-k210 vs HDBSCAN: all-point ARI={ship_hdb_pool_all:.3f}; "
      f"non-noise-only ARI={ship_hdb_pool:.3f} at coverage={cov_pool:.3f}")

print("\nVERDICT")
tr, dv = rows["train"], rows["devel"]
print(f"  side-local all-point ARI: train {tr['km_hdb_all']:.3f}, "
      f"devel {dv['km_hdb_all']:.3f}")
print(f"  side-local non-noise sensitivity: train {tr['km_hdb']:.3f}, "
      f"devel {dv['km_hdb']:.3f} (must report coverage)")
print(f"  shipped-k210 vs side-local: TRAIN {tr['ship_km']:.3f} (cohesive) vs "
      f"DEVEL {dv['ship_km']:.3f} (fragmented if low)")
print(f"  pooled shipped vs HDBSCAN: all={ship_hdb_pool_all:.3f}, "
      f"non-noise={ship_hdb_pool:.3f}")
print("  If side-local cross-method >> pooled-mismatch => the 0.31 was the")
print("  fragmentation/pooling artifact, not weak ECAPA speaker structure.")
