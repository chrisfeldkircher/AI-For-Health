"""
The decision-relevant question is NOT fusion vs G4 (the weaker feature) but
fusion vs the BEST single feature (CQT). The bidirectional eval-independent test
shows fusion loses to CQT-alone in the Dev->Train direction, so 'always fuse' is
not obviously right. This computes, on the official speaker-disjoint partitions
(no clustering in fit/split), the paired bootstrap CI of fusion - CQT_alone in
BOTH directions.

Verdict logic:
  fusion - CQT CI excludes 0 (both dirs) => fusion strictly justified over CQT.
  straddles 0 / flips sign by direction => fusion ~= CQT; justify fusion as a
    variance-reducing hedge (G4 is the low-identity anchor) rather than a win.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import load_labels
from speakers.cluster import load_pseudo_speakers

SEED = 20260720
G4D = ROOT / "cache" / "handcrafted" / "g4"
CQTD = ROOT / "cache" / "handcrafted" / "cqt"


def load(d, stems, sl=None):
    x = np.stack([np.load(d / f"{s}.npy") for s in stems]).astype(np.float32)
    return x[:, sl] if sl is not None else x


def branch_logit(Xs, ys, Xt):
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced",
                                            solver="liblinear", max_iter=3000,
                                            random_state=SEED)).fit(Xs, ys)
    zs = pipe.decision_function(Xs); zt = pipe.decision_function(Xt)
    return (zt - zs.mean()) / (zs.std() + 1e-9)


def uar0(y, z):
    return balanced_accuracy_score(y, (z >= 0).astype(int))


labels = load_labels(str(ROOT / "dataset" / "ComParE2017_Cold_4students"))
pooled = load_pseudo_speakers(ROOT / "cache" / "pseudo_speakers" / "pooled_k420_seed42.tsv")


def side(pfx):
    files = sorted(f for f in labels if f.startswith(pfx + "_"))
    stems = [f[:-4] for f in files]
    y = np.array([labels[f] for f in files])
    g4 = load(G4D, stems, slice(4, None)); cq = load(CQTD, stems)
    grp = np.array([pooled[s] for s in stems])
    return dict(y=y, g4=g4, cq=cq, grp=grp)


TR, DV = side("train"), side("devel")
rng = np.random.default_rng(SEED)

print("=" * 84)
print("EVAL-INDEPENDENT bidirectional: fusion vs the BEST single feature (CQT), with CIs")
print("=" * 84)
for name, S, T in [("Train->Dev", TR, DV), ("Dev->Train", DV, TR)]:
    zg = branch_logit(S["g4"], S["y"], T["g4"])
    zc = branch_logit(S["cq"], S["y"], T["cq"])
    zf = 0.5 * (zg + zc)
    yt, grp = T["y"], T["grp"]
    ug, uc, uf = uar0(yt, zg), uar0(yt, zc), uar0(yt, zf)
    ag, ac, af = roc_auc_score(yt, zg), roc_auc_score(yt, zc), roc_auc_score(yt, zf)
    print(f"\n-- {name} --")
    print(f"  UAR@0   G4={ug:.4f}  CQT={uc:.4f}  FUSION={uf:.4f}   "
          f"(fusion-CQT {uf-uc:+.4f})")
    print(f"  AUC     G4={ag:.4f}  CQT={ac:.4f}  FUSION={af:.4f}   "
          f"(fusion-CQT {af-ac:+.4f})")
    clusters = np.unique(grp); idx_by = {c: np.flatnonzero(grp == c) for c in clusters}
    dfg, dfc = [], []
    for _ in range(2000):
        samp = rng.choice(clusters, size=len(clusters), replace=True)
        ii = np.concatenate([idx_by[c] for c in samp])
        yb = yt[ii]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        dfg.append(uar0(yb, zf[ii]) - uar0(yb, zg[ii]))
        dfc.append(uar0(yb, zf[ii]) - uar0(yb, zc[ii]))
    dfg, dfc = np.array(dfg), np.array(dfc)
    print(f"  boot UAR@0 fusion-G4 : {dfg.mean():+.4f} CI[{np.percentile(dfg,2.5):+.4f},"
          f"{np.percentile(dfg,97.5):+.4f}] P(>0)={float((dfg>0).mean()):.3f}")
    print(f"  boot UAR@0 fusion-CQT: {dfc.mean():+.4f} CI[{np.percentile(dfc,2.5):+.4f},"
          f"{np.percentile(dfc,97.5):+.4f}] P(>0)={float((dfc>0).mean()):.3f}")
