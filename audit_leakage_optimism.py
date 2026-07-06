"""Does the devel-split speaker leakage actually inflate our reported UAR, and
by how much? Direct test on the committed A2.5 heads, no retraining.

Within the current devel_test split, label each clip "leaked" if its top-1
cosine ECAPA neighbor (proxy for same true speaker) lies in devel_val, else
"clean". If the model leans on speaker identity, it should score HIGHER on the
leaked subset (it effectively saw that speaker on the val side). The clean
subset is the honest estimate of devel_test performance.

Uses: 5 committed A2.5 heads (ensemble mean logit), the current k210 grouped
splits, tau swept on train_threshold. Output: results/audit_leakage_optimism.json
Run from repo root. ~2-4 min on GPU.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))

from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from features import LayerWeightedPooledHead
from features.train import _pooled_collate, predict_probs
from honesty import sweep_tau, evaluate_at_tau
from speakers.cluster import load_pseudo_speakers

DATA_DIR   = str(ROOT / "dataset" / "ComParE2017_Cold_4students")
CACHE_ROOT = str(ROOT / "cache")
BACKBONE   = "microsoft_wavlm-large"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS      = [42, 123, 7, 999, 31337]
TSV        = ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv"
NPZ        = ROOT / "cache" / "ecapa-voxceleb" / "ecapa_embeddings.npz"
OUT        = ROOT / "results" / "audit_leakage_optimism.json"

labels_map = load_labels(DATA_DIR)
pseudo     = load_pseudo_speakers(TSV)
full_train = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, split="train")
full_devel = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, split="devel")
tr_fit, tr_thr   = stratified_grouped_split(full_train.files, labels_map, pseudo, val_frac=0.10, seed=42)
dv_val, dv_test  = stratified_grouped_split(full_devel.files, labels_map, pseudo, val_frac=0.50, seed=42)

def _stems(fs): return [f[:-4] if f.endswith(".wav") else f for f in fs]

sample = full_train[0]["pooled"]; NL, SD = sample.shape

def a2_logit(files):
    """Ensemble mean logit over the 5 committed A2.5 heads."""
    acc = None
    for seed in SEEDS:
        head = LayerWeightedPooledHead(n_layers=NL, stat_dim=SD, proj_dim=128,
                                       n_classes=2, dropout=0.5).to(DEVICE)
        st = torch.load(f"{CACHE_ROOT}/{BACKBONE}/head_A2grouped_honestprior_seed{seed}.pt",
                        weights_only=True, map_location=DEVICE)
        head.load_state_dict(st["state_dict"]); head.eval()
        ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, file_list=files)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=_pooled_collate)
        p, _ = predict_probs(head, loader, DEVICE)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        lg = np.log(p / (1 - p))
        acc = lg if acc is None else acc + lg
        del head
    return acc / len(SEEDS)

print("[a2.5] computing ensemble logits on train_threshold + devel ...")
lg_thr  = a2_logit(tr_thr)
lg_val  = a2_logit(dv_val)
lg_test = a2_logit(dv_test)
y_thr  = np.array([labels_map[f] for f in tr_thr],  dtype=np.int64)
y_val  = np.array([labels_map[f] for f in dv_val],  dtype=np.int64)
y_test = np.array([labels_map[f] for f in dv_test], dtype=np.int64)

tau, _ = sweep_tau(lg_thr, y_thr)
overall = evaluate_at_tau(lg_test, y_test, tau)
print(f"[a2.5] tau*={tau:+.3f}  overall devel_test UAR={overall['uar']:.4f}")

# --- label devel_test clips leaked vs clean by top-1 ECAPA NN into devel_val ---
d = np.load(NPZ, allow_pickle=True)
idx = {s: i for i, s in enumerate(d["stems"].astype(str))}
emb = d["embeddings"].astype(np.float32)
val_stems  = _stems(dv_val)
test_stems = _stems(dv_test)
E_val  = normalize(np.vstack([emb[idx[s]] for s in val_stems]),  axis=1)
E_test = normalize(np.vstack([emb[idx[s]] for s in test_stems]), axis=1)
# nearest neighbor of each test clip across ALL devel (val + test), minus self
E_all = np.vstack([E_val, E_test])
side  = np.array([0]*len(val_stems) + [1]*len(test_stems))  # 0=val,1=test
off = len(val_stems)
leaked = np.zeros(len(test_stems), dtype=bool)
B = 2048
for i0 in range(0, len(test_stems), B):
    sim = E_test[i0:i0+B] @ E_all.T
    for r in range(sim.shape[0]):
        sim[r, off + i0 + r] = -2.0            # drop self
    nn = sim.argmax(axis=1)
    leaked[i0:i0+B] = (side[nn] == 0)          # NN is on the val side

def uar_subset(mask):
    if mask.sum() < 20 or len(np.unique(y_test[mask])) < 2:
        return None
    return float(evaluate_at_tau(lg_test[mask], y_test[mask], tau)["uar"])

uar_leaked = uar_subset(leaked)
uar_clean  = uar_subset(~leaked)
print(f"\n[result] devel_test clips: leaked(NN in val)={int(leaked.sum())}  clean={int((~leaked).sum())}")
print(f"[result] UAR on LEAKED subset = {uar_leaked}")
print(f"[result] UAR on CLEAN  subset = {uar_clean}")
if uar_leaked is not None and uar_clean is not None:
    print(f"[result] leakage optimism (leaked - clean) = {uar_leaked - uar_clean:+.4f}")
    print(f"[result] overall {overall['uar']:.4f} sits between; the CLEAN subset "
          f"{uar_clean:.4f} is the honest devel_test estimate for A2.5")

out = {
    "rung_id": "audit_leakage_optimism",
    "description": "Direct test of devel-split speaker-leakage optimism on the "
                   "committed A2.5 5-seed ensemble. devel_test clips split by "
                   "whether their top-1 ECAPA NN is in devel_val (leaked) vs "
                   "devel_test (clean); UAR compared at the train_threshold tau.",
    "tau": float(tau),
    "overall_devel_test_uar": float(overall["uar"]),
    "n_leaked": int(leaked.sum()), "n_clean": int((~leaked).sum()),
    "uar_leaked_subset": uar_leaked, "uar_clean_subset": uar_clean,
    "leakage_optimism_uar": (None if (uar_leaked is None or uar_clean is None)
                             else uar_leaked - uar_clean),
    "leaked_cold_rate": float(y_test[leaked].mean()),
    "clean_cold_rate": float(y_test[~leaked].mean()),
}
OUT.write_text(json.dumps(out, indent=2))
print(f"\n[wrote] {OUT.relative_to(ROOT)}")
