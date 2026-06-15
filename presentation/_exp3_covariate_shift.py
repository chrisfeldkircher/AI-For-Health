"""Experiment 3: is there high-dimensional COVARIATE shift in the 25x4096 WavLM
pooled stats devel -> test? (the reviewer's caveat about exp2)

exp2 only showed the A2 head's 1-D OUTPUT marginal is stable. That does NOT
prove the high-dim INPUT is stable -- and BN-adapt acts on the input. BN-adapt
corrects per-dimension mean/std ONLY, so the precise, BN-adapt-relevant question
is: do the per-dim means/stds of the pooled features shift devel -> test?

  - Per-dim moment shift ~ 0  -> BN-adapt has nothing to correct -> dead (for real).
  - Per-dim moment shift large -> BN-adapt is a LIVE lever; reviewer's caveat has teeth.

Also runs a domain classifier (devel=0 vs test=1) on a PCA reduction as a check
for JOINT (higher-order) shift that per-dim moments would miss.

Streaming per-dim accumulation -> memory-safe (never holds the full 19k x 102k matrix).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "model")
from data.cached_dataset import PooledCacheDataset, load_labels

DATA_DIR = "dataset/ComParE2017_Cold_4students"
CACHE_ROOT = "cache"
BACKBONE_ID = "microsoft_wavlm-large"

labels_map = load_labels(DATA_DIR)
devel_files = sorted(f for f in labels_map if f.startswith("devel_"))
test_stems = sorted(p.stem for p in (Path(CACHE_ROOT) / BACKBONE_ID / "pooled").glob("test_*.pt"))
test_files = [f"{s}.wav" for s in test_stems]
print(f"[files] devel={len(devel_files)}  test={len(test_files)}")


def stream_moments(files):
    """Streaming per-dim mean/std over the flattened (25*4096) pooled vector."""
    ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE_ID, file_list=files)
    n = 0
    s1 = None
    s2 = None
    for i in range(len(ds)):
        x = ds[i]["pooled"].reshape(-1).to(torch.float64).numpy()
        if s1 is None:
            s1 = np.zeros_like(x)
            s2 = np.zeros_like(x)
        s1 += x
        s2 += x * x
        n += 1
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std, n


print("[stream] computing devel per-dim moments ...")
mu_dev, sd_dev, n_dev = stream_moments(devel_files)
print(f"  devel n={n_dev}  dim={mu_dev.size}")
print("[stream] computing test per-dim moments ...")
mu_te, sd_te, n_te = stream_moments(test_files)
print(f"  test  n={n_te}  dim={mu_te.size}")

# --- per-dim standardized mean shift (the BN-adapt-relevant quantity) ---------
eps = 1e-8
shift = (mu_te - mu_dev) / (sd_dev + eps)        # in devel-sigma units
std_ratio = (sd_te + eps) / (sd_dev + eps)
abs_shift = np.abs(shift)

print("\n[per-dim MEAN shift |mu_te - mu_dev| / sd_dev]  (BN-adapt corrects exactly this):")
for p in (50, 90, 95, 99, 99.9):
    print(f"   {p:>5.1f} pct = {np.percentile(abs_shift, p):.4f} sigma")
print(f"   max          = {abs_shift.max():.4f} sigma")
print(f"   frac > 0.10s = {(abs_shift > 0.10).mean():.4f}")
print(f"   frac > 0.30s = {(abs_shift > 0.30).mean():.4f}")
print(f"   frac > 0.50s = {(abs_shift > 0.50).mean():.4f}")

print("\n[per-dim STD ratio  sd_te / sd_dev]:")
for p in (1, 50, 99):
    print(f"   {p:>4.1f} pct = {np.percentile(std_ratio, p):.4f}")
print(f"   mean = {std_ratio.mean():.4f}")

# --- joint shift: domain classifier on a PCA reduction ------------------------
print("\n[joint] domain classifier (devel=0 vs test=1) on PCA-30, 5-fold CV AUC:")
rng = np.random.RandomState(0)
def sample_matrix(files, k):
    ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE_ID, file_list=files)
    idx = rng.choice(len(ds), size=min(k, len(ds)), replace=False)
    rows = [ds[int(i)]["pooled"].reshape(-1).to(torch.float32).numpy() for i in idx]
    return np.vstack(rows)

K = 3000
Xd = sample_matrix(devel_files, K)
Xt = sample_matrix(test_files, K)
X = np.vstack([Xd, Xt]).astype(np.float64)
y = np.concatenate([np.zeros(len(Xd)), np.ones(len(Xt))])
# standardize on combined, PCA-30, LR with CV AUC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), PCA(n_components=30, random_state=0),
                     LogisticRegression(max_iter=2000, C=1.0))
auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(f"   domain-classifier CV AUC = {auc.mean():.4f} +/- {auc.std():.4f}")
print("   (0.50 = devel/test indistinguishable; >>0.50 = joint covariate shift exists)")

print("\n[interpretation]")
print("   If per-dim mean shift is small (<~0.1 sigma typical) AND std ratio ~1,")
print("   BN-adapt has essentially nothing to correct -> dead for real.")
print("   A high domain-classifier AUC with small per-dim moments would mean the")
print("   shift is higher-order (rotational), which BN-adapt also cannot fix.")
