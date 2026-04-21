"""
Four-suspect diagnostic for the 'C=0, layer_weights frozen, UAR=0.5' collapse.

S1: fp16 overflow on higher-order moments in the cache.
S2: feature-scale explosion (finite but ranging 6+ orders of magnitude).
S3: layer_weights parameter not receiving gradient (graph or registration bug).
S4: class-imbalance collapse from unbalanced first batches.

We reconstruct the ORIGINAL head inline so diagnosis matches the failed run.
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from data.cached_dataset import PooledCacheDataset, stratified_split, load_labels
from features.train import _pooled_collate

DATA_DIR = "../dataset/ComParE2017_Cold_4students"
CACHE_ROOT = "../cache"
BACKBONE_ID = "microsoft_wavlm-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OriginalHead(nn.Module):
    """Verbatim reconstruction of the head that produced the collapse."""
    def __init__(self, n_layers, stat_dim, proj_dim=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.n_layers = n_layers
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.input_norm = nn.LayerNorm(stat_dim)
        self.proj = nn.Sequential(
            nn.Linear(stat_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, pooled):
        x = pooled.to(torch.float32)
        w = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1)
        fused = (x * w).sum(dim=1)
        fused = self.input_norm(fused)
        z = self.proj(fused)
        return self.classifier(z), z

    def param_groups(self, base_lr=1e-3):
        return [
            {"params": [self.layer_weights], "lr": base_lr * 0.1, "name": "layer_weights"},
            {"params": list(self.input_norm.parameters())
                       + list(self.proj.parameters())
                       + list(self.classifier.parameters()),
             "lr": base_lr, "name": "head"},
        ]


def S1_cache_finiteness():
    print("=" * 70)
    print("S1: fp16 overflow / non-finite values in cache")
    print("=" * 70)
    pooled_dir = Path(CACHE_ROOT) / BACKBONE_ID / "pooled"
    paths = sorted(pooled_dir.glob("*.pt"))
    n_bad = 0
    bad_per_moment = {"mean": 0, "std": 0, "skew": 0, "kurt": 0}
    moment_names = ["mean", "std", "skew", "kurt"]

    for p in paths:
        t = torch.load(p, weights_only=True, map_location="cpu")
        if torch.isfinite(t).all():
            continue
        n_bad += 1
        L, D = t.shape
        assert D % 4 == 0
        d = D // 4
        blocks = t.view(L, 4, d)
        for i, name in enumerate(moment_names):
            if not torch.isfinite(blocks[:, i]).all():
                bad_per_moment[name] += 1

    print(f"scanned {len(paths)} files")
    print(f"files with any non-finite value: {n_bad}")
    if n_bad:
        print(f"  breakdown by moment block: {bad_per_moment}")
    verdict = "CONFIRMED" if n_bad else "RULED OUT"
    print(f"S1 verdict: {verdict}\n")
    return n_bad > 0


def S2_feature_scale(train_loader):
    print("=" * 70)
    print("S2: feature-scale explosion across cached feature positions")
    print("=" * 70)
    batch = next(iter(train_loader))
    x = batch["pooled"].float()  # [B, L, D]
    B, L, D = x.shape
    # Flatten L*D to get per-feature stats across the batch
    x_flat = x.reshape(B, L * D)
    per_feat_std = x_flat.std(dim=0)
    per_feat_absmax = x_flat.abs().max(dim=0).values

    print(f"batch shape: [B={B}, L={L}, D={D}]")
    print(f"per-feature std   : min={per_feat_std.min():.4e}  "
          f"median={per_feat_std.median():.4e}  max={per_feat_std.max():.4e}  "
          f"ratio_max/min={(per_feat_std.max()/per_feat_std.clamp(min=1e-20).min()).item():.2e}")
    print(f"per-feature |max| : min={per_feat_absmax.min():.4e}  "
          f"median={per_feat_absmax.median():.4e}  max={per_feat_absmax.max():.4e}")

    # Per-moment-block summary
    d = D // 4
    blocks = x.view(B, L, 4, d)
    names = ["mean", "std", "skew", "kurt"]
    print("per-moment summary (across batch, layers, dim):")
    for i, name in enumerate(names):
        bk = blocks[:, :, i, :]
        print(f"  {name:4s}: std={bk.std():.4e}  |max|={bk.abs().max():.4e}  "
              f"mean_of_std_per_position={bk.std(dim=0).mean():.4e}")

    ratio = (per_feat_std.max() / per_feat_std.clamp(min=1e-20).min()).item()
    confirmed = ratio > 1e3
    verdict = "CONFIRMED (ratio > 1e3)" if confirmed else "RULED OUT"
    print(f"S2 verdict: {verdict}\n")
    return confirmed


def S3_gradient_flow(train_loader, n_layers, stat_dim, class_weights):
    print("=" * 70)
    print("S3: layer_weights gradient flow")
    print("=" * 70)
    head = OriginalHead(n_layers=n_layers, stat_dim=stat_dim,
                        proj_dim=256, n_classes=2, dropout=0.3).to(DEVICE)
    optim = torch.optim.AdamW(head.param_groups(base_lr=1e-3), weight_decay=1e-4)

    # Check registration
    lw_in_params = any(id(p) == id(head.layer_weights) for p in head.parameters())
    lw_in_optim = any(
        id(p) == id(head.layer_weights)
        for g in optim.param_groups for p in g["params"]
    )
    print(f"layer_weights in head.parameters(): {lw_in_params}")
    print(f"layer_weights in optimizer groups : {lw_in_optim}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    batch = next(iter(train_loader))
    pooled = batch["pooled"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    head.train()
    logits, _ = head(pooled)
    loss = loss_fn(logits, labels)
    loss.backward()

    g = head.layer_weights.grad
    print(f"layer_weights.grad is None: {g is None}")
    if g is not None:
        print(f"layer_weights.grad: shape={tuple(g.shape)}  "
              f"abs_max={g.abs().max().item():.4e}  "
              f"abs_mean={g.abs().mean().item():.4e}  "
              f"all_same_val={bool((g == g[0]).all())}")
        print(f"layer_weights.grad sample: {g.detach().cpu().numpy()[:8]}")

    # Check other params also have grads
    head_weight_grad = head.proj[0].weight.grad
    print(f"first Linear weight grad abs_max: {head_weight_grad.abs().max().item():.4e}")

    confirmed = (g is None) or (g.abs().max().item() < 1e-10)
    verdict = "CONFIRMED (grad is None or zero)" if confirmed else "RULED OUT"
    print(f"S3 verdict: {verdict}\n")
    return confirmed


def S4_first_batch_balance(train_loader, n_batches=10):
    print("=" * 70)
    print(f"S4: class balance in first {n_batches} training batches")
    print("=" * 70)
    counts_per_batch = []
    it = iter(train_loader)
    for i in range(n_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        labels = b["label"].numpy()
        c = int((labels == 1).sum())
        nc = int((labels == 0).sum())
        counts_per_batch.append((c, nc))
        print(f"  batch {i:2d}: C={c:3d}  NC={nc:3d}  C_frac={c/(c+nc):.3f}")

    c_min = min(c for c, _ in counts_per_batch)
    c_mean = sum(c for c, _ in counts_per_batch) / len(counts_per_batch)
    print(f"min C per batch: {c_min}  mean C per batch: {c_mean:.1f}")
    # "Balanced sampler non-negotiable" threshold from earlier design: >=8 C per batch.
    confirmed = c_min < 8
    verdict = f"CONCERN (min C per batch = {c_min} < 8)" if confirmed else "RULED OUT"
    print(f"S4 verdict: {verdict}\n")
    return confirmed


def main():
    # S1 first (fastest, no data loaders needed)
    s1 = S1_cache_finiteness()

    # Build dataset / loader for S2, S3, S4
    full_train = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE_ID, split="train")
    labels_map = load_labels(DATA_DIR)
    train_files, _ = stratified_split(full_train.files, labels_map, val_frac=0.15, seed=42)
    train_ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE_ID, file_list=train_files)

    sample = full_train[0]["pooled"]
    n_layers, stat_dim = sample.shape
    class_weights = train_ds.class_weights()
    print(f"train_ds: n={len(train_ds)}  counts={train_ds.class_counts()}")
    print(f"class_weights={class_weights.tolist()}")
    print(f"feature shape per sample: [{n_layers}, {stat_dim}]\n")

    loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                        collate_fn=_pooled_collate, num_workers=0,
                        generator=torch.Generator().manual_seed(42))

    s2 = S2_feature_scale(loader)
    s3 = S3_gradient_flow(loader, n_layers, stat_dim, class_weights)
    s4 = S4_first_batch_balance(loader, n_batches=10)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  S1 (fp16/non-finite):           {'CONFIRMED' if s1 else 'ruled out'}")
    print(f"  S2 (feature-scale explosion):   {'CONFIRMED' if s2 else 'ruled out'}")
    print(f"  S3 (layer_weights no gradient): {'CONFIRMED' if s3 else 'ruled out'}")
    print(f"  S4 (class imbalance per batch): {'CONCERN'   if s4 else 'ruled out'}")


if __name__ == "__main__":
    main()
