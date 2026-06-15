"""Experiment 2: is the A2 (WavLM head) logit distribution shifted devel -> test?

Exp 1 exonerated the handcrafted-group fusion (group logits barely shift; re-
standardizing them leaves ranking identical, Spearman 1.0; the -3.74 fused mean
comes from the A2 head). So the question becomes: does the A2 head's own output
distribution shift from devel to test?

  - If A2 devel mean ~= A2 test mean  -> NO location shift; the failure is pure
    ranking/discrimination loss on disjoint test speakers (M8-M19 confound).
    Nothing label-free recovers it.
  - If A2 devel mean >> A2 test mean  -> WavLM-pooled feature shift feeding the
    head; BN-adapt on the pooled scaler (the OTHER half of TTA-Z) could help.

Distributional comparison only (no test labels needed). Mirrors cell 132's
_a2_logit exactly: PooledCacheDataset -> predict_probs -> logit = log(p/(1-p)),
ensemble = mean over the 5 locked seeds.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "model")
from data.cached_dataset import PooledCacheDataset, load_labels
from features import LayerWeightedPooledHead
from features.train import _pooled_collate, predict_probs

DATA_DIR = "dataset/ComParE2017_Cold_4students"
CACHE_ROOT = "cache"
BACKBONE_ID = "microsoft_wavlm-large"
ALL_SEEDS = [42, 123, 7, 999, 31337]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")

labels_map = load_labels(DATA_DIR)
devel_files = sorted(f for f in labels_map if f.startswith("devel_"))
test_files = sorted(
    p.name for p in Path(f"{DATA_DIR}/wav").glob("test_*.wav")
) or [f"{s}.wav" for s in (Path(CACHE_ROOT) / BACKBONE_ID / "pooled").glob("test_*")]
# fallback: derive test file names straight from the pooled cache
test_stems = sorted(p.stem for p in (Path(CACHE_ROOT) / BACKBONE_ID / "pooled").glob("test_*.pt"))
test_files = [f"{s}.wav" for s in test_stems]
print(f"[files] devel={len(devel_files)}  test={len(test_files)}")

# infer head dims from one pooled sample
_probe_ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE_ID, file_list=[devel_files[0]])
_sample = _probe_ds[0]["pooled"]
_NL, _SD = _sample.shape
print(f"[dims] n_layers={_NL}  stat_dim={_SD}")

def load_head(seed):
    head = LayerWeightedPooledHead(n_layers=_NL, stat_dim=_SD, proj_dim=128,
                                   n_classes=2, dropout=0.5).to(DEVICE)
    state = torch.load(f"{CACHE_ROOT}/{BACKBONE_ID}/head_A2grouped_honestprior_seed{seed}.pt",
                       weights_only=True, map_location=DEVICE)
    head.load_state_dict(state["state_dict"])
    head.eval()
    return head

def a2_logit(head, files):
    ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE_ID, file_list=files)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=_pooled_collate)
    p, _ = predict_probs(head, loader, DEVICE)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))

dev_stack = np.zeros((len(devel_files), len(ALL_SEEDS)))
te_stack = np.zeros((len(test_files), len(ALL_SEEDS)))
for j, s in enumerate(ALL_SEEDS):
    h = load_head(s)
    dev_stack[:, j] = a2_logit(h, devel_files)
    te_stack[:, j] = a2_logit(h, test_files)
    del h
    print(f"  seed {s:>5} done")

dev_a2 = dev_stack.mean(axis=1)
te_a2 = te_stack.mean(axis=1)

y_dev = np.array([labels_map[f] for f in devel_files], dtype=np.int64)

def stats(name, x):
    q = np.quantile(x, [0.01, 0.10, 0.50, 0.90, 0.99])
    print(f"  {name:<22} n={len(x):>5} mean={x.mean():>8.3f} std={x.std():>7.3f} "
          f"q01={q[0]:>7.2f} q10={q[1]:>7.2f} q50={q[2]:>7.2f} q90={q[3]:>7.2f} q99={q[4]:>7.2f}")

print("\n[A2 head ensemble logit] devel vs test:")
stats("devel A2", dev_a2)
stats("test  A2", te_a2)
print(f"\n  devel cold prior = {y_dev.mean():.4f}")
print(f"  devel A2 mean among TRUE cold   = {dev_a2[y_dev==1].mean():.3f}")
print(f"  devel A2 mean among TRUE non-cold = {dev_a2[y_dev==0].mean():.3f}")
print(f"  devel A2 separation (cold - noncold) = {dev_a2[y_dev==1].mean() - dev_a2[y_dev==0].mean():.3f}")

# A2-only ranking quality on devel (AUC-like via rank) for reference
from sklearn.metrics import roc_auc_score
print(f"\n  devel A2-only ROC-AUC = {roc_auc_score(y_dev, dev_a2):.4f}")
print(f"  (test A2-only AUC cannot be computed -- no per-row test labels on disk)")

print(f"\n  LOCATION SHIFT devel->test (A2 mean): {te_a2.mean() - dev_a2.mean():+.3f} logit units")
print(f"  SCALE  ratio  devel->test (A2 std):  {te_a2.std()/dev_a2.std():.3f}")
