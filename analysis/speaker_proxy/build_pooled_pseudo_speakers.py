"""Build the POOLED pseudo-speaker labeling that makes within-split groupings
faithful on every split, fixing the V7/V8 fragmentation of the shipped
train-only-fit labels (see results/speaker_pipeline_verification.json).

Design:
  - train+devel POOLED KMeans, k=420 (~210 speakers per split, URTIC), n_init=10,
    seed=42, L2-normalized ECAPA. Shared label space across train and devel, so
    probe / adversary uses that span train->devel stay coherent.
  - test clustered SEPARATELY, k=210, same recipe, cluster IDs offset by +1000.
    Test speakers belong to neither pool; nothing trains on test targets, the
    labels exist for diagnostics and honest within-test grouping only.
  - Output: cache/pseudo_speakers/pooled_k420_seed42.tsv (same format as the
    shipped tsvs). The shipped k210_seed42.tsv is NOT touched: every locked
    result was produced under it and stays comparable under it.

Also verifies the new labeling (same V7/V8 metrics as the pipeline audit) and
writes results/pooled_pseudo_speakers_diagnostics.json.

Run from repo root:  python build_pooled_pseudo_speakers.py   (~3 min CPU)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))

import sklearn
from data.cached_dataset import stratified_grouped_split, load_labels

NPZ = ROOT / "cache" / "ecapa-voxceleb" / "ecapa_embeddings.npz"
OUT_TSV = ROOT / "cache" / "pseudo_speakers" / "pooled_k420_seed42.tsv"
OUT_JSON = ROOT / "results" / "pooled_pseudo_speakers_diagnostics.json"
K_POOLED = 420      # ~210 train + ~210 devel speakers (URTIC)
K_TEST = 210        # ~210 test speakers
TEST_OFFSET = 1000  # keeps the test label space visibly separate
SEED = 42

d = np.load(NPZ, allow_pickle=True)
stems = d["stems"].astype(str)
split = d["split"].astype(str)
emb = d["embeddings"].astype(np.float32)
idx = {s: i for i, s in enumerate(stems)}

# ---- fit pooled train+devel ---------------------------------------------------
pool_mask = np.isin(split, ["train", "devel"])
X_pool = normalize(emb[pool_mask], axis=1)
pool_stems = stems[pool_mask]
print(f"[fit] pooled train+devel: {X_pool.shape[0]} clips, k={K_POOLED}, n_init=10, seed={SEED}")
km_pool = KMeans(n_clusters=K_POOLED, n_init=10, random_state=SEED).fit(X_pool)
pseudo = {s: int(km_pool.labels_[i]) for i, s in enumerate(pool_stems)}

# ---- fit test separately -------------------------------------------------------
test_mask = split == "test"
X_test = normalize(emb[test_mask], axis=1)
test_stems = stems[test_mask]
print(f"[fit] test separate: {X_test.shape[0]} clips, k={K_TEST}, offset +{TEST_OFFSET}")
km_test = KMeans(n_clusters=K_TEST, n_init=10, random_state=SEED).fit(X_test)
for i, s in enumerate(test_stems):
    pseudo[s] = TEST_OFFSET + int(km_test.labels_[i])

# ---- write tsv (same format/order convention as shipped: train, devel, test) ---
OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_TSV.open("w", encoding="utf-8", newline="\n") as f:
    f.write("file_stem\tsplit\tcluster\n")
    for sp in ("train", "devel", "test"):
        for s in stems[split == sp]:
            f.write(f"{s}\t{sp}\t{pseudo[s]}\n")
print(f"[write] {OUT_TSV.relative_to(ROOT)}  ({sum(split!='')} rows)")

# ---- verify: V7 cohesion + V8 split leakage under the NEW labels ---------------
def v7(sp: str) -> float:
    m = split == sp
    X = normalize(emb[m], axis=1)
    lab = np.array([pseudo[s] for s in stems[m]])
    n = X.shape[0]
    same = 0
    B = 2048
    for i0 in range(0, n, B):
        sim = X[i0:i0 + B] @ X.T
        for r in range(sim.shape[0]):
            sim[r, i0 + r] = -2.0
        same += int((lab[sim.argmax(axis=1)] == lab[i0:i0 + B]).sum())
    return float(same / n)

labels_map = load_labels(str(ROOT / "dataset" / "ComParE2017_Cold_4students"))

def v8(sp: str, frac: float):
    files = sorted(f for f in labels_map if f.startswith(sp + "_"))
    a, b = stratified_grouped_split(files, labels_map, pseudo, val_frac=frac, seed=SEED)
    a_st = set(f[:-4] for f in a)
    order = [f[:-4] for f in files]
    X = normalize(np.vstack([emb[idx[s]] for s in order]), axis=1)
    side = np.array([0 if s in a_st else 1 for s in order], dtype=np.int8)
    n = X.shape[0]
    nn_side = np.empty(n, dtype=np.int8)
    B = 2048
    for i0 in range(0, n, B):
        sim = X[i0:i0 + B] @ X.T
        for r in range(sim.shape[0]):
            sim[r, i0 + r] = -2.0
        nn_side[i0:i0 + B] = side[sim.argmax(axis=1)]
    ga = {pseudo[s] for s in a_st}
    gb = {pseudo[f[:-4]] for f in b}
    ra = float(np.mean([labels_map[f] for f in a]))
    rb = float(np.mean([labels_map[f] for f in b]))
    return {"nn_on_other_side": float((nn_side != side).mean()),
            "cluster_overlap": len(ga & gb),
            "cold_rate_a": ra, "cold_rate_b": rb,
            "n_a": len(a), "n_b": len(b)}

print("\n[verify] V7 top-1 NN same-cluster (shipped -> pooled):")
res_v7 = {}
SHIPPED_V7 = {"train": 0.9739, "devel": 0.4919, "test": 0.4731}
for sp in ("train", "devel", "test"):
    res_v7[sp] = v7(sp)
    print(f"  {sp}: {SHIPPED_V7[sp]:.3f} -> {res_v7[sp]:.3f}")

print("\n[verify] V8 grouped-split NN-on-other-side (shipped -> pooled):")
res_v8 = {}
SHIPPED_V8 = {"train": 0.0014, "devel": 0.2746}
for sp, frac in (("train", 0.10), ("devel", 0.50)):
    res_v8[sp] = v8(sp, frac)
    print(f"  {sp}: {SHIPPED_V8[sp]:.4f} -> {res_v8[sp]['nn_on_other_side']:.4f}  "
          f"(overlap {res_v8[sp]['cluster_overlap']}, "
          f"cold rates {res_v8[sp]['cold_rate_a']:.4f}/{res_v8[sp]['cold_rate_b']:.4f})")

ok = (res_v7["devel"] > 0.9 and res_v7["test"] > 0.9
      and res_v8["devel"]["nn_on_other_side"] < 0.05
      and res_v8["train"]["nn_on_other_side"] < 0.05
      and res_v8["train"]["cluster_overlap"] == 0
      and res_v8["devel"]["cluster_overlap"] == 0)
print(f"\n[verdict] pooled labeling {'SOUND' if ok else 'CHECK FAILED'}")

out = {
    "rung_id": "pooled_pseudo_speakers_diagnostics",
    "description": (
        "Faithful within-split pseudo-speaker labeling: KMeans fit on train+devel "
        "POOLED (k=420, ~210 speakers/split) with shared label space; test "
        "clustered separately (k=210, IDs offset +1000). Fixes the V7/V8 "
        "fragmentation of the train-only-fit shipped labels. NOTE: the reported "
        "UAR is NOT biased by the old fragmentation (results/"
        "audit_leakage_optimism.json); this labeling exists so that "
        "'speaker-disjoint' is literally true on all splits and so that "
        "pseudo-speaker TARGETS (speaker probe on devel, A6/A7 heads) are "
        "measured against faithful labels. Locked results stay under "
        "k210_seed42.tsv; do not mix groupings within one experiment."),
    "sklearn_version": sklearn.__version__,
    "k_pooled": K_POOLED, "k_test": K_TEST, "test_offset": TEST_OFFSET,
    "seed": SEED, "tsv": str(OUT_TSV.relative_to(ROOT)),
    "v7_top1_nn_same_cluster": {"shipped": SHIPPED_V7, "pooled": res_v7},
    "v8_split_leakage": {"shipped": SHIPPED_V8,
                         "pooled": {sp: res_v8[sp]["nn_on_other_side"] for sp in res_v8},
                         "detail": res_v8},
    "sound": bool(ok),
}
OUT_JSON.write_text(json.dumps(out, indent=2))
print(f"[wrote] {OUT_JSON.relative_to(ROOT)}")
