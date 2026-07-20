"""
Does the corrected (held-chunk) speaker gate FLIP the handcrafted admissions?

audit_speaker_probe_protocol.py showed the cross-pool nearest-centroid probe
understates identity leakage 4.14x -- but ONLY on the 128-d A2.5 backbone z,
with a cosine-centroid probe. The honesty GATE that actually admits feature
groups uses a LINEAR LR probe (model/honesty/probe.py) on the feature groups
themselves (G4 7/11-d, G9 CQT 168-d). The understatement magnitude scales with
how much speaker info a feature carries, so 4x cannot be assumed for the
low-dim handcrafted groups you would actually submit.

This re-measures the gate for each feature under BOTH protocols, with the SAME
linear probe the gate uses, on the SAME shipped k210 grouping the gate used:
  cross_pool  : fit LR on train, eval on DEVEL (nearest-train-centroid labels)  = gate as-used
  held_chunk  : per-identity 80/20 split of TRAIN speakers, eval held-out train = correct instrument
Reports top1 vs the honest MAJORITY baseline for each, and whether a feature
that PASSES the 0.05 gate cross-pool would FAIL it held-chunk.

Run: <datascience python> audit_gate_remeasure.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
META = ROOT / "paper_data" / "eda" / "chunk_metadata.csv"
WAVLM_PCA = ROOT / "paper_data" / "eda" / "wavlm_a25_pca128_fp16.npy"
G4_DIR = ROOT / "cache" / "handcrafted" / "g4"
CQT_DIR = ROOT / "cache" / "handcrafted" / "cqt"
K210 = ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv"
GATE = 0.05
SEED = 42


def load_grouping(tsv: Path) -> dict[str, int]:
    out = {}
    with tsv.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            stem, _sp, clu = line.rstrip("\n").split("\t")
            out[stem] = int(clu)
    return out


def load_feat_dir(d: Path, stems: list[str]) -> np.ndarray:
    return np.vstack([np.load(d / f"{s}.npy").astype(np.float32) for s in stems])


def per_identity_holdout(labels: np.ndarray, frac=0.20, seed=SEED):
    rng = np.random.default_rng(seed)
    fit, ev = [], []
    for ident in np.unique(labels):
        idx = np.flatnonzero(labels == ident)
        rng.shuffle(idx)
        n_ev = max(1, min(int(round(len(idx) * frac)), len(idx) - 1))
        ev.extend(idx[:n_ev]); fit.extend(idx[n_ev:])
    return np.array(sorted(fit)), np.array(sorted(ev))


def lr_probe_top1(Xf, yf, Xe, ye):
    """Linear multinomial LR probe, matching model/honesty/probe.py speaker_probe."""
    sc = StandardScaler().fit(Xf)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    clf.fit(sc.transform(Xf), yf)
    pred = clf.predict(sc.transform(Xe))
    top1 = float((pred == ye).mean())
    maj = float(max(Counter(ye.tolist()).values()) / len(ye))
    return top1, maj


# ---- load ------------------------------------------------------------------
meta = list(csv.DictReader(META.open(encoding="utf-8")))
stems = [r["file_stem"] for r in meta]
sp = {r["file_stem"]: r["split"] for r in meta}
g = load_grouping(K210)
tr = [s for s in stems if sp[s] == "train"]
dv = [s for s in stems if sp[s] == "devel"]
ytr_all = np.array([g[s] for s in tr]); ydv = np.array([g[s] for s in dv])
print(f"train {len(tr)} chunks / {len(set(ytr_all))} pseudo-speakers ; devel {len(dv)} chunks")

wav = np.load(WAVLM_PCA).astype(np.float32)
wmap = {stems[i]: wav[i] for i in range(len(stems))}
FEATS = {
    "WavLM A2.5 (PCA-128)": (np.vstack([wmap[s] for s in tr]), np.vstack([wmap[s] for s in dv])),
    "G4 handcrafted (11-d)": (load_feat_dir(G4_DIR, tr), load_feat_dir(G4_DIR, dv)),
    "G9 CQT handcrafted (168-d)": (load_feat_dir(CQT_DIR, tr), load_feat_dir(CQT_DIR, dv)),
}

# held-chunk split of TRAIN (shared across features for comparability)
fit_idx, ev_idx = per_identity_holdout(ytr_all)

print("\n" + "=" * 92)
print("SPEAKER GATE RE-MEASUREMENT  (linear LR probe, k210 grouping, gate threshold 0.05)")
print("=" * 92)
print(f"{'feature':<28} {'CROSS-POOL (as-used)':>24} {'HELD-CHUNK (correct)':>24} {'flip?':>8}")
print(f"{'':<28} {'top1 / maj / verdict':>24} {'top1 / maj / verdict':>24}")
print("-" * 92)
rows = {}
for name, (Xtr, Xdv) in FEATS.items():
    # cross-pool = gate as used: fit on all train, eval on devel
    cp_top1, cp_maj = lr_probe_top1(Xtr, ytr_all, Xdv, ydv)
    # held-chunk = correct: fit on train-fit subset, eval on held-out train
    hc_top1, hc_maj = lr_probe_top1(Xtr[fit_idx], ytr_all[fit_idx], Xtr[ev_idx], ytr_all[ev_idx])
    cp_pass = "PASS" if cp_top1 <= GATE else "FAIL"
    hc_pass = "PASS" if hc_top1 <= GATE else "FAIL"
    flip = "FLIP" if (cp_pass == "PASS" and hc_pass == "FAIL") else ""
    rows[name] = dict(cp_top1=cp_top1, cp_maj=cp_maj, hc_top1=hc_top1, hc_maj=hc_maj,
                      cp_pass=cp_pass, hc_pass=hc_pass, flip=bool(flip))
    print(f"{name:<28} {cp_top1:>7.4f} /{cp_maj:>6.4f} / {cp_pass:<4} "
          f"{hc_top1:>9.4f} /{hc_maj:>6.4f} / {hc_pass:<4} {flip:>8}")
print("-" * 92)
print("cross-pool top1 vs its OWN majority baseline: if top1 <= maj, the probe is measuring NOISE.")
print("held-chunk top1 vs its majority baseline (~1%): real recoverable identity in the feature.")

print("\nVERDICT")
for name, r in rows.items():
    ratio = r["hc_top1"] / max(r["cp_top1"], 1e-9)
    below = "BELOW its own majority (noise)" if r["cp_top1"] <= r["cp_maj"] else "above majority"
    print(f"  {name}:")
    print(f"    cross-pool {r['cp_top1']:.4f} is {below}; held-chunk {r['hc_top1']:.4f} "
          f"({ratio:.1f}x the cross-pool number)")
    if r["flip"]:
        print(f"    ==> ADMISSION FLIPS: passes the gate cross-pool, FAILS it held-chunk")
    elif r["hc_pass"] == "PASS":
        print(f"    ==> survives the corrected gate (still PASS held-chunk)")
