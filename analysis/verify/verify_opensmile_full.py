"""
Close the openSMILE gap honestly: eGeMAPS-88 was tested (0.504) but the full
ComParE-2016 6373-d set (the official challenge feature set) never went through
the corrected protocol. Run it through the FROZEN apparatus - no new Dev-driven
selection - to answer three questions:
  1. standalone honest outer-CV UAR (does 6373-d beat 7-d G4 = 0.592?)
  2. held-chunk identity top1 (is it just more speaker identity?)
  3. does it ADD over G4+CQT on the disjoint official Dev (fixed z-avg fusion)?
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import load_labels
from speakers.cluster import load_pseudo_speakers

SEED = 20260720
HC = ROOT / "cache" / "handcrafted"


def load(sub, stems, sl=None):
    x = np.stack([np.load(HC / sub / f"{s}.npy") for s in stems]).astype(np.float32)
    return x[:, sl] if sl is not None else x


def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=SEED))


def outer_cv(X, y, groups):
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    fu = []
    for tr, te in sgkf.split(X, y, groups):
        m = mk().fit(X[tr], y[tr])
        fu.append(balanced_accuracy_score(y[te], (m.decision_function(X[te]) >= 0).astype(int)))
    return np.mean(fu), np.std(fu, ddof=1)


def heldchunk_identity(X, ids):
    rng = np.random.default_rng(SEED)
    fit, ev = [], []
    for i in np.unique(ids):
        idx = np.flatnonzero(ids == i); rng.shuffle(idx)
        n = max(1, min(int(round(0.2 * len(idx))), len(idx) - 1))
        ev += list(idx[:n]); fit += list(idx[n:])
    fit, ev = np.array(fit), np.array(ev)
    sc = StandardScaler().fit(X[fit])
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=SEED)
    clf.fit(sc.transform(X[fit]), ids[fit])
    top1 = float((clf.predict(sc.transform(X[ev])) == ids[ev]).mean())
    maj = float(max(Counter(ids[ev].tolist()).values()) / len(ev))
    return top1, maj


def zbranch(Xf, yf, Xe):
    m = mk().fit(Xf, yf)
    sf, se = m.decision_function(Xf), m.decision_function(Xe)
    return (se - sf.mean()) / (sf.std() + 1e-9)


labels = load_labels(str(ROOT / "dataset/ComParE2017_Cold_4students"))
k210 = load_pseudo_speakers(ROOT / "cache/pseudo_speakers/k210_seed42.tsv")
tr = [f[:-4] for f in sorted(f for f in labels if f.startswith("train_"))]
dv = [f[:-4] for f in sorted(f for f in labels if f.startswith("devel_"))]
ytr = np.array([labels[s + ".wav"] for s in tr]); ydv = np.array([labels[s + ".wav"] for s in dv])
gtr = np.array([k210[s] for s in tr])

print("loading ComParE-2016 (6373-d) + eGeMAPS (88-d) ...")
CP_tr = load("compare2016", tr); CP_dv = load("compare2016", dv)
EG_tr = load("egemaps", tr)
G4_tr = load("g4", tr, slice(4, None)); G4_dv = load("g4", dv, slice(4, None))
CQ_tr = load("cqt", tr); CQ_dv = load("cqt", dv)
print(f"  ComParE {CP_tr.shape}, eGeMAPS {EG_tr.shape}")

print("\n" + "=" * 74)
print("1. STANDALONE honest outer-CV (5 grouped folds, fixed threshold 0)")
print("=" * 74)
for name, X in [("ComParE-2016 (6373-d)", CP_tr), ("eGeMAPS (88-d) recheck", EG_tr)]:
    m, s = outer_cv(X, ytr, gtr)
    print(f"  {name:<26} UAR={m:.4f} +/- {s:.4f}   (G4 bar 0.592)")

print("\n" + "=" * 74)
print("2. HELD-CHUNK identity recovery (linear probe; majority ~0.011)")
print("=" * 74)
for name, X in [("ComParE-2016", CP_tr), ("eGeMAPS", EG_tr)]:
    t1, mj = heldchunk_identity(X, gtr)
    print(f"  {name:<16} identity top1={t1:.3f}  (majority {mj:.3f})   [G4=0.13, CQT=0.71-0.94]")

print("\n" + "=" * 74)
print("3. Does ComParE ADD over G4+CQT on disjoint official Dev? (AUC, fixed z-avg)")
print("=" * 74)
zg, zc, zp = zbranch(G4_tr, ytr, G4_dv), zbranch(CQ_tr, ytr, CQ_dv), zbranch(CP_tr, ytr, CP_dv)
fus2 = 0.5 * (zg + zc)
fus3 = (zg + zc + zp) / 3.0
print(f"  G4+CQT (2-way)         AUC={roc_auc_score(ydv, fus2):.4f}")
print(f"  G4+CQT+ComParE (3-way) AUC={roc_auc_score(ydv, fus3):.4f}")
print(f"  ComParE alone          AUC={roc_auc_score(ydv, zp):.4f}")
print(f"  delta (3way - 2way): {roc_auc_score(ydv, fus3) - roc_auc_score(ydv, fus2):+.4f}")
