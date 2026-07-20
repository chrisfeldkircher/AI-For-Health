"""
Does CQT add REAL transferable cold signal over G4, or is the +0.044 an artifact
of combined-data grouped CV riding CQT's identity content?

Decisive arbiter that needs NO pseudo-speaker grouping: the challenge Train and
Development partitions are officially SPEAKER-DISJOINT (schuller17 §2.2). So fit
each branch on official Train, evaluate the fixed z-scored equal-average fusion
on official Development = a real cross-population test. Grouping is used ONLY to
resample subjects for the paired bootstrap CI, not for the split.

Reports, on official Dev:
  ROC-AUC          : threshold-free separability (fair G4 vs G4+CQT comparison)
  UAR @ thresh 0   : the recommended deployment rule (z-avg, threshold 0)
  UAR @ Dev-opt    : upper bound (threshold tuned on Dev -- optimistic, for context)
  paired subject-bootstrap CI on (G4+CQT - G4) for AUC and UAR@0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
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


def fit_branch(Xtr, ytr, Xdv):
    """Balanced LR; return z-scored (train-stats) decision logits on train and dev."""
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced",
                                            solver="liblinear", max_iter=3000,
                                            random_state=SEED)).fit(Xtr, ytr)
    ztr = pipe.decision_function(Xtr)
    zdv = pipe.decision_function(Xdv)
    mu, sd = ztr.mean(), ztr.std() + 1e-9
    return (zdv - mu) / sd


def uar_at(y, score, tau):
    return balanced_accuracy_score(y, (score >= tau).astype(int))


def best_uar(y, score):
    fpr, tpr, th = roc_curve(y, score, pos_label=1)
    return float(np.max(0.5 * (tpr + 1 - fpr)))


labels = load_labels(str(ROOT / "dataset" / "ComParE2017_Cold_4students"))
tr_files = sorted(f for f in labels if f.startswith("train_"))
dv_files = sorted(f for f in labels if f.startswith("devel_"))
tr = [f[:-4] for f in tr_files]; dv = [f[:-4] for f in dv_files]
ytr = np.array([labels[f] for f in tr_files]); ydv = np.array([labels[f] for f in dv_files])

G4tr = load(G4D, tr, slice(4, None)); G4dv = load(G4D, dv, slice(4, None))
CQtr = load(CQTD, tr); CQdv = load(CQTD, dv)

z_g4 = fit_branch(G4tr, ytr, G4dv)
z_cq = fit_branch(CQtr, ytr, CQdv)
z_fus = 0.5 * (z_g4 + z_cq)

print("=" * 78)
print("OFFICIAL Train -> Development (speaker-DISJOINT partitions, no grouping used)")
print("=" * 78)
print(f"  train {len(tr)} chunks / devel {len(dv)} chunks   cold devel {int(ydv.sum())}")
print(f"\n{'model':<16} {'ROC-AUC':>9} {'UAR@0':>9} {'UAR@Dev-opt':>12}")
print("-" * 78)
for name, z in [("G4 alone", z_g4), ("CQT alone", z_cq), ("G4+CQT z-avg", z_fus)]:
    print(f"{name:<16} {roc_auc_score(ydv, z):>9.4f} {uar_at(ydv, z, 0.0):>9.4f} {best_uar(ydv, z):>12.4f}")
print("-" * 78)

# ---- paired subject-bootstrap: (G4+CQT - G4) on disjoint Dev ------------------
pooled = load_pseudo_speakers(ROOT / "cache" / "pseudo_speakers" / "pooled_k420_seed42.tsv")
dv_grp = np.array([pooled[s] for s in dv])
clusters = np.unique(dv_grp)
idx_by_c = {c: np.flatnonzero(dv_grp == c) for c in clusters}
rng = np.random.default_rng(SEED)
d_auc, d_uar0 = [], []
for _ in range(2000):
    samp = rng.choice(clusters, size=len(clusters), replace=True)
    idx = np.concatenate([idx_by_c[c] for c in samp])
    yb = ydv[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue
    d_auc.append(roc_auc_score(yb, z_fus[idx]) - roc_auc_score(yb, z_g4[idx]))
    d_uar0.append(uar_at(yb, z_fus[idx], 0.0) - uar_at(yb, z_g4[idx], 0.0))
d_auc, d_uar0 = np.array(d_auc), np.array(d_uar0)


def ci(a):
    return np.percentile(a, 2.5), np.mean(a), np.percentile(a, 97.5)


print("\nPAIRED subject-bootstrap (2000x, resample pooled_k420 devel clusters):")
lo, m, hi = ci(d_auc)
print(f"  delta ROC-AUC (G4+CQT - G4): {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      f"P(>0)={float((d_auc>0).mean()):.3f}")
lo, m, hi = ci(d_uar0)
print(f"  delta UAR@0   (G4+CQT - G4): {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      f"P(>0)={float((d_uar0>0).mean()):.3f}")
print("\nARBITER: CI excludes 0 on DISJOINT official Dev => CQT adds real transferable")
print("signal (reversal justified). CI includes 0 => the +0.044 was CV/grouping optimism.")
