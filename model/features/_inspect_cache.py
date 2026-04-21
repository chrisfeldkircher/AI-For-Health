"""Inspect distribution of cached pooled stats for pathological values."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

BACKBONE_ID = sys.argv[1] if len(sys.argv) > 1 else "microsoft_wavlm-large"
CACHE_ROOT = Path("../cache")

pooled_dir = CACHE_ROOT / BACKBONE_ID / "pooled"
files = sorted(pooled_dir.glob("*.pt"))[:200]
print(f"inspecting {len(files)} pooled tensors from {pooled_dir}")

n_inf = 0
n_nan = 0
abs_maxes = []
for p in files:
    t = torch.load(p, weights_only=True, map_location="cpu")
    if torch.isinf(t).any():
        n_inf += 1
    if torch.isnan(t).any():
        n_nan += 1
    abs_maxes.append(t.abs().float().max().item())

t0 = torch.load(files[0], weights_only=True, map_location="cpu").float()
L, D = t0.shape
assert D % 4 == 0
d = D // 4
mean_block = t0[:, :d]
std_block = t0[:, d:2*d]
skew_block = t0[:, 2*d:3*d]
kurt_block = t0[:, 3*d:]

print(f"shape per file: {tuple(t0.shape)}  dtype={t0.dtype}")
print(f"n files with any inf: {n_inf} / {len(files)}")
print(f"n files with any nan: {n_nan} / {len(files)}")
print(f"abs-max across samples: min={min(abs_maxes):.2f}  "
      f"median={sorted(abs_maxes)[len(abs_maxes)//2]:.2f}  "
      f"max={max(abs_maxes):.2f}")

print("\nsample 0 (file={}) block-wise stats:".format(files[0].name))
for name, blk in [("mean", mean_block), ("std", std_block),
                  ("skew", skew_block), ("kurt", kurt_block)]:
    vals = blk.flatten()
    print(f"  {name:4s}: min={vals.min().item():+10.3f}  "
          f"max={vals.max().item():+10.3f}  "
          f"mean={vals.mean().item():+8.3f}  "
          f"p99_abs={vals.abs().quantile(0.99).item():8.3f}  "
          f"n_inf={int(torch.isinf(blk).sum())}  "
          f"n_nan={int(torch.isnan(blk).sum())}")
