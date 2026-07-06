"""Verification audit of the speaker-identity pipeline the whole team relies on.

Covers the gaps the 2024 diagnostics (results/pseudo_speaker_ecapa_diagnostics.json,
train+devel only) did not:

  V1  packed npz integrity      ecapa_embeddings.npz rows == per-clip .pt files
  V2  embedding sanity          NaN/Inf, zero rows, duplicates, norm range (all splits)
  V3  kmeans reproducibility    refit KMeans(k=210, n_init=10, seed=42) on train
                                -> compare to shipped k210_seed42.tsv train labels
  V4  devel/test assignment     recompute centroids from shipped train labels,
                                nearest-centroid assign devel+test -> compare shipped
  V5  test-split label cohesion 10-NN cosine same-cluster rate on TEST (the split
                                the prior diagnostics never saw), devel as anchor
  V6  split machinery           stratified_grouped_split: cluster-disjointness,
                                cold stratification, determinism (run twice)

Output: results/speaker_pipeline_verification.json + printed PASS/FAIL table.
Run from repo root:  python verify_speaker_pipeline.py   (CPU, ~2-4 min)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))

import sklearn
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from data.cached_dataset import stratified_grouped_split, load_labels
from speakers.cluster import load_pseudo_speakers

NPZ = ROOT / "cache" / "ecapa-voxceleb" / "ecapa_embeddings.npz"
PT_DIR = ROOT / "cache" / "ecapa-voxceleb"
TSV = ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv"
DATA_DIR = str(ROOT / "dataset" / "ComParE2017_Cold_4students")
OUT = ROOT / "results" / "speaker_pipeline_verification.json"
K = 210
SEED = 42

report: dict = {"rung_id": "speaker_pipeline_verification",
                "sklearn_version": sklearn.__version__,
                "checks": {}}
failures: list[str] = []


def check(name: str, ok: bool, detail: dict) -> None:
    report["checks"][name] = {"pass": bool(ok), **detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


print("=== V0: load artifacts ===")
d = np.load(NPZ, allow_pickle=True)
stems = d["stems"].astype(str)
split = d["split"].astype(str)
emb16 = d["embeddings"]
order = {s: i for i, s in enumerate(stems)}
rows = list(csv.DictReader(TSV.open(encoding="utf-8"), delimiter="\t"))
tsv_map = {r["file_stem"]: (r["split"], int(r["cluster"])) for r in rows}
print(f"  npz: {emb16.shape} {emb16.dtype};  tsv rows: {len(rows)}")

# ---------------------------------------------------------------- V1
print("\n=== V1: packed npz vs per-clip .pt (300 random stems) ===")
rng = np.random.default_rng(0)
sample = rng.choice(stems, size=300, replace=False)
n_bad = 0
for s in sample:
    t = torch.load(PT_DIR / f"{s}.pt", weights_only=False, map_location="cpu")
    v = (t.numpy() if hasattr(t, "numpy") else np.asarray(t)).reshape(-1).astype(np.float16)
    if not np.array_equal(v, emb16[order[s]]):
        n_bad += 1
check("V1_npz_matches_pt", n_bad == 0, {"sampled": 300, "mismatches": n_bad})

# ---------------------------------------------------------------- V2
print("\n=== V2: embedding sanity (all 28652) ===")
emb32 = emb16.astype(np.float32)
n_nan = int(np.isnan(emb32).sum())
n_inf = int(np.isinf(emb32).sum())
norms = np.linalg.norm(emb32, axis=1)
n_zero = int((norms < 1e-6).sum())
# duplicate rows via hashing rounded embeddings
hashes = {}
n_dup = 0
dup_examples = []
for i in range(emb32.shape[0]):
    h = emb16[i].tobytes()
    if h in hashes:
        n_dup += 1
        if len(dup_examples) < 3:
            dup_examples.append((stems[hashes[h]], stems[i]))
    else:
        hashes[h] = i
check("V2_embedding_sanity", (n_nan == 0 and n_inf == 0 and n_zero == 0 and n_dup == 0),
      {"nan": n_nan, "inf": n_inf, "zero_norm_rows": n_zero,
       "duplicate_rows": n_dup, "dup_examples": dup_examples,
       "norm_min": float(norms.min()), "norm_med": float(np.median(norms)),
       "norm_max": float(norms.max())})

# ---------------------------------------------------------------- V3
print("\n=== V3: KMeans reproducibility on train (k=210, n_init=10, seed=42) ===")
tr_mask = split == "train"
X_tr = normalize(emb32[tr_mask], axis=1)
tr_stems = stems[tr_mask]
shipped_tr = np.array([tsv_map[s][1] for s in tr_stems], dtype=np.int64)
km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
refit_tr = km.fit_predict(X_tr)
exact = float((refit_tr == shipped_tr).mean())
ari = float(adjusted_rand_score(shipped_tr, refit_tr))
# exact label equality requires identical sklearn internals; ARI is the robust bar
check("V3_kmeans_reproducibility", ari > 0.95 or exact > 0.99,
      {"exact_label_match_frac": exact, "ari_vs_shipped": ari,
       "note": "ARI robust to label permutation and sklearn version drift"})

# ---------------------------------------------------------------- V4
print("\n=== V4: devel/test nearest-centroid assignment vs shipped ===")
cent = np.vstack([X_tr[shipped_tr == c].mean(axis=0) for c in range(K)])
res_v4 = {}
ok_v4 = True
for sp in ("devel", "test"):
    m = split == sp
    Xs = normalize(emb32[m], axis=1)
    ss = stems[m]
    shipped = np.array([tsv_map[s][1] for s in ss], dtype=np.int64)
    d2 = ((Xs ** 2).sum(1, keepdims=True) - 2 * Xs @ cent.T
          + (cent ** 2).sum(1)[None, :])
    pred = d2.argmin(axis=1)
    agree = float((pred == shipped).mean())
    res_v4[sp] = agree
    # centroids from shipped labels are close to but not identical to KMeans'
    # final centers; near-boundary points can flip. 0.95 is the soundness bar.
    ok_v4 &= agree > 0.95
check("V4_assignment_consistency", ok_v4,
      {"agree_devel": res_v4["devel"], "agree_test": res_v4["test"],
       "note": "nearest-centroid re-derivation from shipped train labels"})

# ---------------------------------------------------------------- V5
print("\n=== V5: 10-NN cosine label cohesion, TEST split (never validated before) ===")
res_v5 = {}
for sp in ("devel", "test"):
    m = split == sp
    Xs = normalize(emb32[m], axis=1)
    ss = stems[m]
    lab = np.array([tsv_map[s][1] for s in ss], dtype=np.int64)
    n = Xs.shape[0]
    same_frac = np.zeros(n, dtype=np.float32)
    B = 1024
    for i0 in range(0, n, B):
        sim = Xs[i0:i0 + B] @ Xs.T                      # cosine (unit norm)
        for r in range(sim.shape[0]):
            sim[r, i0 + r] = -2.0                        # drop self
        nn = np.argpartition(-sim, 10, axis=1)[:, :10]
        same_frac[i0:i0 + B] = (lab[nn] == lab[i0:i0 + B, None]).mean(axis=1)
    counts = np.bincount(lab, minlength=K).astype(np.float64)
    chance = float(((counts / n) ** 2).sum())
    res_v5[sp] = {"mean_same_cluster": float(same_frac.mean()),
                  "chance": chance,
                  "lift": float(same_frac.mean() / chance)}
    print(f"  {sp}: same-cluster 10-NN = {same_frac.mean():.4f} "
          f"(chance {chance:.4f}, lift {same_frac.mean()/chance:.0f}x)")
# bar mirrors the prior diagnostics' branch rule (lift >= 20x)
check("V5_test_label_cohesion", res_v5["test"]["lift"] >= 20.0 and res_v5["devel"]["lift"] >= 20.0,
      res_v5)

# ---------------------------------------------------------------- V6
print("\n=== V6: stratified_grouped_split integrity ===")
labels_map = load_labels(DATA_DIR)
pseudo = load_pseudo_speakers(TSV)
res_v6 = {}
ok_v6 = True
for sp, frac in (("train", 0.10), ("devel", 0.50)):
    files = sorted(f for f in labels_map if f.startswith(sp + "_"))
    a1, b1 = stratified_grouped_split(files, labels_map, pseudo, val_frac=frac, seed=SEED)
    a2, b2 = stratified_grouped_split(files, labels_map, pseudo, val_frac=frac, seed=SEED)
    det = (a1 == a2) and (b1 == b2)
    ga = {pseudo[f[:-4]] for f in a1}
    gb = {pseudo[f[:-4]] for f in b1}
    disjoint = len(ga & gb) == 0
    ra = float(np.mean([labels_map[f] for f in a1]))
    rb = float(np.mean([labels_map[f] for f in b1]))
    res_v6[sp] = {"deterministic": det, "cluster_disjoint": disjoint,
                  "n_side_a": len(a1), "n_side_b": len(b1),
                  "cold_rate_a": ra, "cold_rate_b": rb,
                  "overlap_clusters": len(ga & gb)}
    ok_v6 &= det and disjoint and abs(ra - rb) < 0.05
    print(f"  {sp}: deterministic={det} disjoint={disjoint} "
          f"cold_rates=({ra:.4f}, {rb:.4f}) sizes=({len(a1)}, {len(b1)})")
check("V6_split_machinery", ok_v6, res_v6)

# ---------------------------------------------------------------- V7
print("\n=== V7: top-1 NN cluster cohesion per split (fragmentation probe) ===")
# For each clip, is its single nearest neighbor (cosine, same split; ECAPA NN is
# almost surely the same true speaker) in the same cluster? High on train (KMeans
# fit there), low on devel/test = train-only fit fragments disjoint speakers.
res_v7 = {}
for sp in ("train", "devel", "test"):
    m = split == sp
    Xs = normalize(emb32[m], axis=1)
    lab = np.array([tsv_map[s][1] for s in stems[m]], dtype=np.int64)
    n = Xs.shape[0]
    same = 0
    B = 2048
    for i0 in range(0, n, B):
        sim = Xs[i0:i0 + B] @ Xs.T
        for r in range(sim.shape[0]):
            sim[r, i0 + r] = -2.0
        nn1 = sim.argmax(axis=1)
        same += int((lab[nn1] == lab[i0:i0 + B]).sum())
    res_v7[sp] = float(same / n)
    print(f"  {sp}: top-1 NN same-cluster = {same/n:.4f}")
# This is a DIAGNOSTIC, not a pass/fail: train is expected high, devel/test low
# by construction. Flag the asymmetry as a soundness concern to surface, not hide.
v7_concern = res_v7["devel"] < 0.7 or res_v7["test"] < 0.7
report["checks"]["V7_fragmentation"] = {
    "diagnostic": True, "concern": bool(v7_concern),
    "top1_nn_same_cluster": res_v7,
    "interpretation": (
        "train-only KMeans fit makes devel/test pseudo-speakers fragmented "
        "(disjoint speakers assigned to train centroids). Low devel/test cohesion "
        "means 'cluster-disjoint' does not imply 'speaker-disjoint' on those splits."),
}
print(f"  [{'CONCERN' if v7_concern else 'ok'}] devel/test fragmentation")

# ---------------------------------------------------------------- V8
print("\n=== V8: same-speaker-proxy leakage across the actual grouped splits ===")
# Reproduce each grouped split; fraction of clips whose top-1 NN (proxy for same
# true speaker) lands on the OTHER side. ~0 = speaker-honest; toward the min-side
# fraction = the split barely separates true speakers.
res_v8 = {}
for sp, frac in (("train", 0.10), ("devel", 0.50)):
    files = sorted(f for f in labels_map if f.startswith(sp + "_"))
    a, b = stratified_grouped_split(files, labels_map, pseudo, val_frac=frac, seed=SEED)
    a_st = set(f[:-4] for f in a)
    order_sp = [f[:-4] for f in files]
    Xs = normalize(np.vstack([emb32[order[s]] for s in order_sp]), axis=1)
    side = np.array([0 if s in a_st else 1 for s in order_sp], dtype=np.int8)
    n = Xs.shape[0]
    nn_side = np.empty(n, dtype=np.int8)
    B = 2048
    for i0 in range(0, n, B):
        sim = Xs[i0:i0 + B] @ Xs.T
        for r in range(sim.shape[0]):
            sim[r, i0 + r] = -2.0
        nn_side[i0:i0 + B] = side[sim.argmax(axis=1)]
    cross = float((nn_side != side).mean())
    res_v8[sp] = {"nn_on_other_side": cross,
                  "random_floor": float(min(side.mean(), 1 - side.mean()))}
    print(f"  {sp}: NN-on-other-side = {cross:.4f} "
          f"(floor {res_v8[sp]['random_floor']:.3f})")
v8_concern = res_v8["devel"]["nn_on_other_side"] > 0.05
report["checks"]["V8_split_speaker_leakage"] = {
    "diagnostic": True, "concern": bool(v8_concern),
    "per_split": res_v8,
    "interpretation": (
        "devel-side leakage is high because devel pseudo-speakers are fragmented "
        "(V7). The reported devel_test UAR and shadow-mean are computed on this "
        "split, so they are optimistically biased relative to a truly "
        "speaker-disjoint test. Train side is clean."),
}
print(f"  [{'CONCERN' if v8_concern else 'ok'}] devel split same-speaker leakage")

# ---------------------------------------------------------------- verdict
print("\n=== VERDICT ===")
report["all_pass"] = len(failures) == 0
report["failures"] = failures
concerns = [k for k, v in report["checks"].items() if v.get("concern")]
report["diagnostic_concerns"] = concerns
print(f"  integrity checks (V1-V6): {'ALL PASS' if report['all_pass'] else 'FAILURES: ' + ', '.join(failures)}")
print(f"  soundness concerns (V7-V8): {', '.join(concerns) if concerns else 'none'}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2))
print(f"\n[wrote] {OUT.relative_to(ROOT)}")
