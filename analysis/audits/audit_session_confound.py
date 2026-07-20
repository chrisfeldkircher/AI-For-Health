"""
Is the held-chunk 0.94 identity recovery a stable subject signature, or an
artifact of splitting adjacent within-recording chunks?

URTIC is CROSS-SECTIONAL: one session per subject (schuller17 §2.2, session
15 min-2 h), so speaker and session/channel are confounded IN PRINCIPLE -- no
clean cross-session probe exists. But subjects did MULTIPLE tasks (read stories,
voice commands, numbers 1-40, spontaneous speech) within that one session. If
chunk file-indices preserve recording order, an index-BLOCK split (train early
chunks, test late chunks of the same subject) tests whether identity persists
across time/task within the session, vs a random split that can pair adjacent
near-duplicate chunks.

  contiguity : are a cluster's chunks a contiguous index run (subject-ordered)
               or scattered (shuffled)? decides whether the block split is meaningful.
  random     : per-identity random 80/20 (the headline probe) -> reproduces 0.94
  block      : per-identity first-70%/last-30% by file index -> cross-time-within-session

If block ~= random and clusters are contiguous: identity is a persistent subject
signature (voice+channel), not an adjacency artifact. If block << random: the
headline was inflated by within-recording chunk similarity.

Run: <datascience python> audit_session_confound.py
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
META = ROOT / "paper_data" / "eda" / "chunk_metadata.csv"
G4_DIR = ROOT / "cache" / "handcrafted" / "g4"
CQT_DIR = ROOT / "cache" / "handcrafted" / "cqt"
K210 = ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv"
SEED = 42


def load_grouping(tsv):
    out = {}
    with open(tsv, encoding="utf-8") as f:
        next(f)
        for line in f:
            stem, _sp, clu = line.rstrip("\n").split("\t")
            out[stem] = int(clu)
    return out


def load_feat_dir(d, stems):
    return np.vstack([np.load(d / f"{s}.npy").astype(np.float32) for s in stems])


def lr_top1(Xf, yf, Xe, ye):
    sc = StandardScaler().fit(Xf)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    clf.fit(sc.transform(Xf), yf)
    pred = clf.predict(sc.transform(Xe))
    return float((pred == ye).mean()), float(max(Counter(ye.tolist()).values()) / len(ye))


meta = list(csv.DictReader(META.open(encoding="utf-8")))
stems = [r["file_stem"] for r in meta if r["split"] == "train"]
idx_of = {s: int(re.sub(r"\D", "", s)) for s in stems}   # numeric file index
g = load_grouping(K210)
y = np.array([g[s] for s in stems])
order_idx = np.array([idx_of[s] for s in stems])

# ---- cluster index-contiguity -------------------------------------------------
spans, sizes = [], []
for c in np.unique(y):
    ii = np.sort(order_idx[y == c])
    spans.append(ii[-1] - ii[0] + 1)
    sizes.append(len(ii))
spans, sizes = np.array(spans), np.array(sizes)
contig = spans.astype(float) / np.maximum(sizes, 1)   # ~1 => contiguous run; >>1 => scattered
print("=" * 84)
print("CLUSTER INDEX-CONTIGUITY (span / size ; ~1 = subject-ordered contiguous block)")
print("=" * 84)
print(f"  clusters={len(sizes)}  median size={int(np.median(sizes))}")
print(f"  span/size ratio: median={np.median(contig):.2f}  "
      f"p10={np.percentile(contig,10):.2f}  p90={np.percentile(contig,90):.2f}")
frac_contig = float((contig < 2.0).mean())
print(f"  fraction of clusters that are near-contiguous (<2x): {frac_contig:.2f}")
meaningful = frac_contig > 0.5
print(f"  => index-block split is {'MEANINGFUL (chunks are recording-ordered)' if meaningful else 'NOT meaningful (indices are shuffled)'}")

# ---- per-identity random vs block splits --------------------------------------
rng = np.random.default_rng(SEED)
rand_fit, rand_ev, blk_fit, blk_ev = [], [], [], []
for c in np.unique(y):
    ci = np.flatnonzero(y == c)
    if len(ci) < 3:
        continue
    # random 80/20
    perm = ci[rng.permutation(len(ci))]
    ne = max(1, min(int(round(len(ci) * 0.20)), len(ci) - 1))
    rand_ev += list(perm[:ne]); rand_fit += list(perm[ne:])
    # block: first 70% / last 30% by file index
    bi = ci[np.argsort(order_idx[ci])]
    cut = max(1, int(round(len(bi) * 0.70)))
    cut = min(cut, len(bi) - 1)
    blk_fit += list(bi[:cut]); blk_ev += list(bi[cut:])
rand_fit, rand_ev = np.array(rand_fit), np.array(rand_ev)
blk_fit, blk_ev = np.array(blk_fit), np.array(blk_ev)

FEATS = {
    "G4 (11-d)": load_feat_dir(G4_DIR, stems),
    "G9 CQT (168-d)": load_feat_dir(CQT_DIR, stems),
}

print("\n" + "=" * 84)
print("IDENTITY RECOVERY: random 80/20  vs  index-block (train early / test late)")
print("=" * 84)
print(f"{'feature':<18} {'random top1':>14} {'block top1':>14} {'block/random':>14}")
print("-" * 84)
for name, X in FEATS.items():
    r1, _ = lr_top1(X[rand_fit], y[rand_fit], X[rand_ev], y[rand_ev])
    b1, _ = lr_top1(X[blk_fit], y[blk_fit], X[blk_ev], y[blk_ev])
    print(f"{name:<18} {r1:>14.4f} {b1:>14.4f} {b1/max(r1,1e-9):>13.2f}x")
print("-" * 84)
print("block ~= random (and clusters contiguous) => persistent subject signature,")
print("   not an adjacency artifact. block << random => headline was adjacency-inflated.")
print("\nNOTE: neither split separates SPEAKER from SESSION -- URTIC is cross-sectional")
print("(one session per subject), so that separation is impossible in principle here.")
