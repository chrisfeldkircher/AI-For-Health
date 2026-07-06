# ECAPA-TDNN speaker embeddings and speaker labels

Shared so the whole team uses one set of speaker embeddings and one speaker
grouping. This avoids everyone running ECAPA separately and ending up with
different speaker-ID counts.

## Speaker labels (use these for grouped cross-validation)

`../pseudo_speakers/k210_seed42.tsv` is the canonical grouping. Columns:

```
file_stem   split   cluster
train_0001  train   31
...
```

- kmeans, k = 210, on the ECAPA embeddings below, seed 42.
- Covers all three splits: train (9505), devel (9596), test (9551).
- Use the `cluster` column as the grouping variable in StratifiedGroupKFold so
  cross-validation is speaker-honest and comparable across branches.
- `k100_seed42.tsv` and `k420_seed42.tsv` are the same at coarser/finer k.

## Embeddings

`ecapa_embeddings.npz` (about 10 MB) packs all 28652 per-clip embeddings.

```python
import numpy as np
d = np.load("cache/ecapa-voxceleb/ecapa_embeddings.npz", allow_pickle=True)
stems = d["stems"]          # (28652,) e.g. "train_0001"
split = d["split"]          # (28652,) "train" | "devel" | "test"
emb   = d["embeddings"]     # (28652, 192) float16
by_stem = {s: emb[i] for i, s in enumerate(stems)}
```

- Model: ECAPA-TDNN, speechbrain `spkrec-ecapa-voxceleb`, 192-d, float16.
- The per-clip `.pt` files in this folder are the source and are gitignored;
  the packed npz is the shared artifact.
- Re-pack from source with `python pack_ecapa_embeddings.py` at the repo root.

## Reproducing the labels from the embeddings

kmeans(k=210, seed=42) on the L2-normalized embeddings reproduces
`k210_seed42.tsv`. If you re-cluster yourself you may get a different labeling,
so please use the committed tsv rather than your own run.
