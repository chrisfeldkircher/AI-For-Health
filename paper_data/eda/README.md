# paper_data/eda/

Exploratory-data-analysis bundle for the URTIC dataset. All files are sized
so the whole directory fits in < 20 MB — the multi-GB raw audio and the
4096-d × 25-layer WavLM-Large pooled cache stay on the experiment machine.

## What's here

| file | size | content |
| --- | --- | --- |
| `chunk_metadata.csv` | 580 KB | 19101 rows × {file_stem, split, cold_label, k100_cluster, k210_cluster, k420_cluster}. **This row order is shared by every `.npy` below — row i in the metadata = row i in every embedding matrix.** |
| `ecapa_embeddings_fp16.npy` | 7.3 MB | (19101, 192) ECAPA-TDNN speaker embeddings. The pseudo-speakers in `chunk_metadata.csv` are KMeans (k∈{100, 210, 420}) on these. |
| `wavlm_a25_pca128_fp16.npy` | 4.9 MB | (19101, 128) PCA-128 of the WavLM-A2.5 substrate. Original substrate = `softmax(50 · sub@1) · pooled` over 25 WavLM-Large layers (4096-d). |
| `wavlm_a25_pca_components_fp16.npy` | 1.0 MB | (128, 4096) PCA basis — multiply by PCA-128 coords to recover an approximation to the 4096-d substrate if needed. |
| `wavlm_a25_pca_variance.csv` | 3.6 KB | per-component explained-variance ratio + cumulative; check at what k you've captured 90 / 95 / 99% of variance |
| `wavlm_a25_layer_weights.csv` | 0.7 KB | per-layer (0–25) sub@1 vector + honesty-prior softmax weights used to build the substrate |
| `pseudo_speaker_cluster_stats.csv` | ~30 KB | per-cluster (k=210): size, cold rate, train share, intra-cluster cosine, nearest-other-cluster cosine, gap |
| `umap_coords_ecapa_2d.csv` | ~700 KB | 2D UMAP coords (n_neighbors=30, min_dist=0.10, cosine) on raw ECAPA-192; joined with metadata (split, cold_label, k210_cluster) for easy colouring |
| `umap_coords_wavlm_a25_2d.csv` | ~700 KB | 2D UMAP coords on the PCA-128 of the WavLM-A2.5 substrate; joined with metadata |

## Loading examples (Python, on Mac)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Metadata + 2D UMAPs are CSV — open in pandas
meta  = pd.read_csv("paper_data/eda/chunk_metadata.csv")
u_w   = pd.read_csv("paper_data/eda/umap_coords_wavlm_a25_2d.csv")
u_e   = pd.read_csv("paper_data/eda/umap_coords_ecapa_2d.csv")
stats = pd.read_csv("paper_data/eda/pseudo_speaker_cluster_stats.csv")

# 1. UMAP of WavLM-A2.5 substrate, coloured by cold label
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(u_w.umap_x, u_w.umap_y, c=u_w.cold_label, s=2, alpha=0.4, cmap="coolwarm")
ax.set_title("WavLM-A2.5 substrate (PCA-128 → UMAP-2D), coloured by cold label")
plt.tight_layout(); plt.show()

# 2. UMAP of ECAPA, coloured by pseudo-speaker (k=210)
import matplotlib.cm as cm
fig, ax = plt.subplots(figsize=(7, 7))
colors = cm.tab20(u_e.k210_cluster % 20)
ax.scatter(u_e.umap_x, u_e.umap_y, c=colors, s=2, alpha=0.6)
ax.set_title("ECAPA-TDNN (UMAP-2D), coloured by k=210 pseudo-speaker")
plt.tight_layout(); plt.show()

# 3. Pseudo-speaker cluster quality: intra vs nearest-other-centroid cosine
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(stats.intra_cluster_mean_cosine, stats.nearest_other_cluster_cosine,
           c=stats.cold_rate, cmap="viridis", s=14)
ax.plot([0, 1], [0, 1], "k--", lw=0.5)
ax.set_xlabel("intra-cluster mean cosine (high = tight)")
ax.set_ylabel("nearest-other-cluster cosine (low = well-separated)")
plt.colorbar(label="cold rate"); plt.tight_layout(); plt.show()

# 4. Cluster size + cold-rate distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(stats.n_chunks, bins=30); axes[0].set_xlabel("chunks per cluster"); axes[0].set_title(f"k=210 cluster sizes (median {int(stats.n_chunks.median())})")
axes[1].hist(stats.cold_rate, bins=20); axes[1].set_xlabel("cluster cold rate"); axes[1].set_title("cluster-level cold imbalance")
plt.tight_layout(); plt.show()

# 5. PCA scree (variance captured vs n components)
v = pd.read_csv("paper_data/eda/wavlm_a25_pca_variance.csv")
plt.plot(v.component, v.cumulative_variance_ratio, marker=".")
for thr in (0.5, 0.9, 0.95):
    plt.axhline(thr, ls="--", alpha=0.4)
plt.xlabel("# PCA components"); plt.ylabel("cum. variance ratio")
plt.title("PCA scree on WavLM-A2.5 substrate"); plt.tight_layout(); plt.show()
```

## Re-running UMAP with different hyperparameters

UMAP coords here are at `n_neighbors=30, min_dist=0.10, metric="cosine"`. If
you want to try other settings on the Mac:

```python
import numpy as np
import umap
meta  = pd.read_csv("paper_data/eda/chunk_metadata.csv")
emb   = np.load("paper_data/eda/wavlm_a25_pca128_fp16.npy")  # (N, 128) fp16
u     = umap.UMAP(n_neighbors=15, min_dist=0.30, metric="cosine", random_state=42)
xy    = u.fit_transform(emb.astype(np.float32))  # ~30s
```

For the raw ECAPA: `emb = np.load("paper_data/eda/ecapa_embeddings_fp16.npy")` (192-d).

## Row-order invariants

Every embedding `.npy` is row-aligned to `chunk_metadata.csv`. Specifically:

```python
meta = pd.read_csv("paper_data/eda/chunk_metadata.csv")
ecapa = np.load("paper_data/eda/ecapa_embeddings_fp16.npy")
assert len(meta) == ecapa.shape[0] == 19101
# meta.iloc[i].file_stem corresponds to ecapa[i, :]
```

The metadata is sorted by `(split, file_stem)` so row 0 = first devel chunk
alphabetically, etc.

## How the WavLM-A2.5 substrate is built

For each chunk, the experiment cache stores `(25, 4096) fp16` pooled stats
(mean, std, skew, kurt concatenated) per WavLM-Large hidden layer. The A2.5
substrate compresses this to a single 4096-d vector by:

```
layer_w = softmax(T_INV * sub_at_1)        # T_INV=50; sub_at_1 from results/A5d_layer_honesty.csv
substrate = sum_l layer_w[l] * pooled[l]   # (4096,)
```

This is the **initial** (honesty-prior, deterministic) layer-weight pooling —
NOT the post-trained-head weighted sum, which is seed-dependent. For an EDA
that needs to be reproducible from immutable per-layer-audit numbers, this is
the right choice. After PCA-128 we keep ~85% of the variance (see
`wavlm_a25_pca_variance.csv`).

## Pseudo-speaker labels — what they are

`k{100, 210, 420}_cluster` are KMeans cluster assignments on the
**ECAPA-TDNN-Voxceleb speaker embeddings** (192-d, ℓ2-normalised, cosine-style).
The clustering is fit once on all of `train + devel` (no test) at
`SPLIT_SEED=42`. URTIC does not release true speaker IDs; the project's
strict-protocol speaker-disjoint sub-split uses `k=210` as the
grouping variable.

Why these labels make sense — to verify visually:
- **Cluster size**: median ~90 chunks/cluster at k=210, range typically 30–200.
- **Intra-cluster cosine**: should be high (~0.70+) since each cluster is a
  speaker-coherent subset of ECAPA space.
- **Nearest-other-cluster cosine**: should be lower than intra, by margin ~0.10+.
- **Train/devel share per cluster**: should be ~50/50 if no leakage (and yes
  it is — speakers do appear in both splits, which is why the project applies
  StratifiedGroupKFold sub-splitting downstream).
