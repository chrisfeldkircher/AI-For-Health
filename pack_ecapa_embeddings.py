"""Pack the per-utterance ECAPA-TDNN speaker embeddings into one shareable file.

The embeddings live as 28652 tiny per-clip .pt tensors in
cache/ecapa-voxceleb/ (train + devel + test, 192-d float16). That is 119 MB of
disk blocks over 28k files and is gitignored. This script packs them into a
single cache/ecapa-voxceleb/ecapa_embeddings.npz (about 11 MB) so teammates can
pull one file and get the exact same embeddings.

Arrays in the npz:
  stems       (N,)      str, e.g. "train_0001", sorted
  split       (N,)      str, "train" | "devel" | "test"
  embeddings  (N, 192)  float16, row i is the embedding for stems[i]
  model       ()        str, provenance

Load with:
  import numpy as np
  d = np.load("cache/ecapa-voxceleb/ecapa_embeddings.npz", allow_pickle=True)
  stems, split, emb = d["stems"], d["split"], d["embeddings"]
  by_stem = {s: emb[i] for i, s in enumerate(stems)}

The canonical speaker grouping (kmeans, k=210) is already committed as
cache/pseudo_speakers/k210_seed42.tsv (columns: file_stem, split, cluster).
Use that as the grouping variable so results are comparable across branches.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EMB_DIR = ROOT / "cache" / "ecapa-voxceleb"
OUT = EMB_DIR / "ecapa_embeddings.npz"

pt_files = sorted(EMB_DIR.glob("*.pt"))
assert pt_files, f"no .pt embeddings under {EMB_DIR}"
print(f"[pack] found {len(pt_files)} embedding files")

stems, splits, rows = [], [], []
for p in pt_files:
    stem = p.stem
    t = torch.load(p, weights_only=False, map_location="cpu")
    v = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    v = v.reshape(-1).astype(np.float16)
    assert v.shape[0] == 192, f"{stem}: expected 192-d, got {v.shape}"
    stems.append(stem)
    splits.append(stem.split("_")[0])
    rows.append(v)

emb = np.vstack(rows).astype(np.float16)
stems_arr = np.array(stems)
split_arr = np.array(splits)

# sanity: split counts
uniq, counts = np.unique(split_arr, return_counts=True)
print("[pack] split counts:", dict(zip(uniq.tolist(), counts.tolist())))
print(f"[pack] embeddings matrix: {emb.shape} {emb.dtype}")

np.savez_compressed(
    OUT,
    stems=stems_arr,
    split=split_arr,
    embeddings=emb,
    model=np.array("ECAPA-TDNN (speechbrain spkrec-ecapa-voxceleb), 192-d, float16"),
)
mb = OUT.stat().st_size / (1024 * 1024)
print(f"[pack] wrote {OUT.relative_to(ROOT)}  ({mb:.1f} MB, {len(stems)} rows)")
