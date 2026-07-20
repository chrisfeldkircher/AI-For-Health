"""
Independent check of Agent 3's claim: a grouped-OOF threshold fit on the FITTING
side transfers to the held-out side and beats threshold-0 for the fixed G4+CQT
fusion. Official speaker-disjoint sides; threshold never sees the eval side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import load_labels
from speakers.cluster import load_pseudo_speakers

SEED = 20260720
G4D, CQTD = ROOT/"cache"/"handcrafted"/"g4", ROOT/"cache"/"handcrafted"/"cqt"


def load(d, stems, sl=None):
    x = np.stack([np.load(d/f"{s}.npy") for s in stems]).astype(np.float32)
    return x[:, sl] if sl is not None else x


def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=SEED))


def fused_scores(Xg_fit, Xc_fit, y_fit, Xg_ev, Xc_ev):
    mg, mc = mk().fit(Xg_fit, y_fit), mk().fit(Xc_fit, y_fit)
    def z(m, Xf, Xe):
        sf = m.decision_function(Xf); se = m.decision_function(Xe)
        return (se - sf.mean())/(sf.std()+1e-9)
    return 0.5*(z(mg, Xg_fit, Xg_ev) + z(mc, Xc_fit, Xc_ev))


def uar_at(y, s, t): return balanced_accuracy_score(y, (s >= t).astype(int))


def best_tau(y, s):
    fpr, tpr, th = roc_curve(y, s, pos_label=1)
    return float(th[np.nanargmax(0.5*(tpr+1-fpr))])


labels = load_labels(str(ROOT/"dataset/ComParE2017_Cold_4students"))
pooled = load_pseudo_speakers(ROOT/"cache/pseudo_speakers/pooled_k420_seed42.tsv")

def side(pfx):
    fs = sorted(f for f in labels if f.startswith(pfx+"_"))
    st = [f[:-4] for f in fs]
    return dict(st=st, y=np.array([labels[f] for f in fs]),
                g4=load(G4D, st, slice(4, None)), cq=load(CQTD, st),
                grp=np.array([pooled[s] for s in st]))
TR, DV = side("train"), side("devel")

print("="*80)
print("OOF-threshold transfer: threshold fit on FIT side (grouped 5-fold OOF), applied to EVAL side")
print("="*80)
res = {}
for name, S, T in [("Train->Dev", TR, DV), ("Dev->Train", DV, TR)]:
    # grouped-OOF fused scores on the fit side -> pick UAR-optimal tau
    oof = np.full(len(S["y"]), np.nan)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for itr, iva in sgkf.split(S["g4"], S["y"], S["grp"]):
        oof[iva] = fused_scores(S["g4"][itr], S["cq"][itr], S["y"][itr], S["g4"][iva], S["cq"][iva])
    tau = best_tau(S["y"], oof)
    # refit on full fit side, score eval side
    se = fused_scores(S["g4"], S["cq"], S["y"], T["g4"], T["cq"])
    u0, uc = uar_at(T["y"], se, 0.0), uar_at(T["y"], se, tau)
    res[name] = (u0, uc, tau)
    print(f"  {name}: threshold0 UAR={u0:.4f}   OOF-tau({tau:+.3f}) UAR={uc:.4f}   delta={uc-u0:+.4f}")
m0 = np.mean([res[k][0] for k in res]); mc = np.mean([res[k][1] for k in res])
print("-"*80)
print(f"  bidirectional mean:  threshold0={m0:.4f}   OOF-calibrated={mc:.4f}   delta={mc-m0:+.4f}")
