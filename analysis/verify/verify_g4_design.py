"""
Independent confirmation of the corrected-rerun G4 headline + interrogation of
the "keep 7 gain-invariant dims, drop dims 0-3" design choice.

Same protocol as rerun_corrected_outer_cv.py (5 outer StratifiedGroupKFold on
official Train, k210 train pseudo-speakers, balanced liblinear LR) but with the
recommended FIXED threshold 0 (decision_function >= 0), and compared across G4
variants + under both the train-only and pooled groupings for robustness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import load_labels
from speakers.cluster import load_pseudo_speakers

SEED = 20260720
G4 = ROOT / "cache" / "handcrafted" / "g4"


def load_g4(stems, sl=None):
    x = np.stack([np.load(G4 / f"{s}.npy") for s in stems]).astype(np.float32)
    return x[:, sl] if sl is not None else x


def outer_cv(X, y, groups, fixed0=True):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_uar, rc, rnc = [], [], []
    for tr, te in sgkf.split(np.zeros(len(y)), y, groups):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced",
                                                 solver="liblinear", max_iter=3000,
                                                 random_state=SEED)).fit(X[tr], y[tr])
        score = pipe.decision_function(X[te])
        pred = (score >= 0.0).astype(int)   # fixed threshold 0
        fold_uar.append(balanced_accuracy_score(y[te], pred))
        rc.append(recall_score(y[te], pred, pos_label=1, zero_division=0))
        rnc.append(recall_score(y[te], pred, pos_label=0, zero_division=0))
    return np.array(fold_uar), np.mean(rc), np.mean(rnc)


labels = load_labels(str(ROOT / "dataset" / "ComParE2017_Cold_4students"))
files = sorted(f for f in labels if f.startswith("train_"))
stems = [f[:-4] for f in files]
y = np.array([labels[f] for f in files])

groupings = {
    "k210_train (harness)": {s: c for s, c in load_pseudo_speakers(ROOT/"cache"/"pseudo_speakers"/"k210_seed42.tsv").items()},
    "pooled_k420": {s: c for s, c in load_pseudo_speakers(ROOT/"cache"/"pseudo_speakers"/"pooled_k420_seed42.tsv").items()},
}
variants = {
    "G4 full (11-d)": load_g4(stems),
    "G4 gain-inv (dims 4:, 7-d)": load_g4(stems, slice(4, None)),
    "G4 dropped dims (0:4, 4-d)": load_g4(stems, slice(0, 4)),
}

print("=" * 82)
print("G4 DESIGN CHECK  (5-fold speaker-grouped outer CV, balanced LR, fixed threshold 0)")
print("=" * 82)
for gname, g in groupings.items():
    grp = np.array([g[s] for s in stems])
    print(f"\n-- grouping: {gname}  ({len(np.unique(grp))} groups) --")
    print(f"{'variant':<30} {'mean UAR':>9} {'fold std':>9} {'recallC/NC':>14}")
    for vname, X in variants.items():
        fu, rc, rnc = outer_cv(X, y, grp)
        print(f"{vname:<30} {fu.mean():>9.4f} {fu.std(ddof=1):>9.4f}   {rc:.3f}/{rnc:.3f}")
