"""Nested speaker-group CV for WavLM/HuBERT heads and residual G4 fusion.

Official Development is untouched. In each of five outer Train folds:
  1. an inner speaker split selects the best training epoch and any G4 weight;
  2. a fresh head is trained for that fixed epoch on the full outer-training set;
  3. the outer speaker fold is evaluated once.

The old devel-derived honesty layer prior is deliberately not reused. Layer
weights start uniformly and are learned only from the current outer-training
data. Outputs: results/corrected_outer_cv_foundations.json and OOF NPZ.
"""

from __future__ import annotations

import json
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import PooledCacheDataset, load_labels  # noqa: E402
from features import LayerWeightedPooledHead  # noqa: E402
from features.train import (  # noqa: E402
    _pooled_collate, make_balanced_sampler, train_head,
)
from speakers.cluster import load_pseudo_speakers  # noqa: E402


SEED = 20260720
OUTER_FOLDS = 5
BACKBONES = {
    "WavLM_large": "microsoft_wavlm-large",
    "HuBERT_large": "facebook_hubert-large-ll60k",
}
BETA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
ALPHA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


class CacheSubset(Dataset):
    """Index view that preserves the label API used by the balanced sampler."""

    def __init__(self, parent: Dataset, indices: np.ndarray):
        self.parent = parent
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict:
        return self.parent[int(self.indices[item])]

    def get_labels(self) -> list[int]:
        labels = self.parent.get_labels()
        return [labels[int(i)] for i in self.indices]


class PreloadedCacheDataset(Dataset):
    """One-time RAM preload to avoid reopening 19k tensors on every epoch."""

    def __init__(self, source: PooledCacheDataset):
        first = source[0]["pooled"]
        self.pooled = torch.empty((len(source), *first.shape), dtype=first.dtype)
        self.labels = source.get_labels()
        self.files = list(source.files)
        for i in range(len(source)):
            self.pooled[i].copy_(source[i]["pooled"])
            if (i + 1) % 2000 == 0 or i + 1 == len(source):
                print(f"[preload] {i+1}/{len(source)} tensors", flush=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, item: int) -> dict:
        return {
            "pooled": self.pooled[item],
            "label": torch.tensor(self.labels[item], dtype=torch.long),
            "file_name": self.files[item],
        }

    def get_labels(self) -> list[int]:
        return self.labels


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "uar": float(balanced_accuracy_score(y, pred)),
        "recall_C": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_NC": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "accuracy": float(np.mean(y == pred)),
    }


def best_tau(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, score, pos_label=1)
    values = 0.5 * (tpr + 1 - fpr)
    i = int(np.nanargmax(values))
    tau = float(thresholds[i])
    if not np.isfinite(tau):
        tau = float(np.nextafter(np.max(score), np.inf))
    return tau, float(values[i])


def zfit(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(max(x.std(), 1e-8))


def zapply(x: np.ndarray, pars: tuple[float, float]) -> np.ndarray:
    return (x - pars[0]) / pars[1]


def g4_model(x: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=SEED,
        ),
    ).fit(x, y)


@torch.no_grad()
def head_logit(head: nn.Module, ds: PooledCacheDataset, device: str) -> np.ndarray:
    loader = DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=_pooled_collate
    )
    head.eval()
    values = []
    for batch in loader:
        logits, _ = head(batch["pooled"].to(device))
        values.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def new_head(ds: PooledCacheDataset, device: str) -> LayerWeightedPooledHead:
    n_layers, stat_dim = ds[0]["pooled"].shape
    return LayerWeightedPooledHead(
        n_layers=n_layers, stat_dim=stat_dim, proj_dim=128,
        n_classes=2, dropout=0.5,
    ).to(device)


def train_fixed(
    head: LayerWeightedPooledHead,
    ds: PooledCacheDataset,
    *, epochs: int, seed: int, device: str,
) -> None:
    """Fresh fixed-epoch fit on the full outer-training fold."""
    torch.manual_seed(seed); np.random.seed(seed)
    fit_loader = DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=_pooled_collate
    )
    head.scaler.fit(fit_loader)
    sampler = make_balanced_sampler(ds, seed=seed)
    loader = DataLoader(
        ds, batch_size=64, sampler=sampler, num_workers=0,
        collate_fn=_pooled_collate, drop_last=False,
    )
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(head.param_groups(base_lr=1e-3), weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    for epoch in range(1, max(epochs, 1) + 1):
        head.train(); loss_sum = 0.0; n = 0
        for batch in loader:
            pooled = batch["pooled"].to(device); target = batch["label"].to(device)
            logits, _ = head(pooled); loss = loss_fn(logits, target)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optim.step()
            loss_sum += float(loss.item()) * len(target); n += len(target)
        sched.step()
        print(f"    fixed epoch {epoch}/{epochs} loss={loss_sum/max(n,1):.4f}")


def main() -> None:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = ROOT / "dataset" / "ComParE2017_Cold_4students"
    labels = load_labels(str(data_dir))
    pseudo = load_pseudo_speakers(ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv")
    files = sorted(f for f in labels if f.startswith("train_"))
    stems = [Path(f).stem for f in files]
    y = np.asarray([labels[f] for f in files], dtype=np.int64)
    groups = np.asarray([pseudo[s] for s in stems], dtype=np.int64)
    g4 = np.stack([
        np.load(ROOT / "cache" / "handcrafted" / "g4" / f"{s}.npy")[4:]
        for s in stems
    ]).astype(np.float32)
    index = {f: i for i, f in enumerate(files)}

    outer = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED
    )
    splits = list(outer.split(np.zeros(len(y)), y, groups))
    all_scores = {}
    all_pred = {}
    rows = []

    for backbone_name, backbone in BACKBONES.items():
        all_scores[backbone_name] = np.full(len(y), np.nan)
        all_scores[f"{backbone_name}_plus_G4"] = np.full(len(y), np.nan)
        all_scores[f"G4_anchor_plus_{backbone_name}"] = np.full(len(y), np.nan)
        all_pred[backbone_name] = np.full(len(y), -1, dtype=np.int8)
        all_pred[f"{backbone_name}_plus_G4"] = np.full(len(y), -1, dtype=np.int8)
        all_pred[f"G4_anchor_plus_{backbone_name}"] = np.full(len(y), -1, dtype=np.int8)

        print(f"\n{'='*80}\nBACKBONE {backbone_name} ({backbone}) on {device}\n{'='*80}")
        # Index the on-disk cache once. Constructing PooledCacheDataset repeatedly
        # re-globs all 19k files on Windows and dominates the actual experiment.
        disk_ds = PooledCacheDataset(
            str(data_dir), str(ROOT / "cache"), backbone, file_list=files
        )
        full_ds = PreloadedCacheDataset(disk_ds)
        del disk_ds
        for fold, (otr, ote) in enumerate(splits):
            fold_seed = SEED + fold
            inner = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=SEED + 1000 + fold
            )
            itr_local, iva_local = next(inner.split(
                np.zeros(len(otr)), y[otr], groups[otr]
            ))
            itr, iva = otr[itr_local], otr[iva_local]
            ds_itr = CacheSubset(full_ds, itr)
            ds_iva = CacheSubset(full_ds, iva)
            ds_otr = CacheSubset(full_ds, otr)
            ds_ote = CacheSubset(full_ds, ote)
            print(f"\n--- {backbone_name} OUTER {fold+1}/5 "
                  f"inner_fit={len(itr)} inner_val={len(iva)} outer_test={len(ote)} ---")

            # Stage 1: epoch and residual weight selection inside outer train.
            torch.manual_seed(fold_seed); np.random.seed(fold_seed)
            head_inner = new_head(ds_itr, device)
            trained = train_head(
                head_inner, ds_itr, ds_iva, test_ds=None,
                epochs=15, batch_size=64, base_lr=1e-3, weight_decay=1e-3,
                early_stop_patience=4, class_weights=None, balanced_sampler=True,
                fit_scaler=True, device=device, num_workers=0, seed=fold_seed,
            )
            best_epoch = max(1, int(trained.best_epoch))
            h_fit = head_logit(head_inner, ds_itr, device)
            h_val = head_logit(head_inner, ds_iva, device)
            hp = zfit(h_fit); zh_val = zapply(h_val, hp)
            g4_inner = g4_model(g4[itr], y[itr])
            g_fit = g4_inner.decision_function(g4[itr]).astype(np.float64)
            g_val = g4_inner.decision_function(g4[iva]).astype(np.float64)
            gp = zfit(g_fit); zg_val = zapply(g_val, gp)

            tau_head, inner_head_uar = best_tau(y[iva], zh_val)
            best_fusion = None
            for beta in BETA_GRID:
                fused = zh_val + beta * zg_val
                tau, value = best_tau(y[iva], fused)
                candidate = (value, -beta, beta, tau)
                if best_fusion is None or candidate > best_fusion:
                    best_fusion = candidate
            _, _, beta, tau_fusion = best_fusion
            best_anchor = None
            for alpha in ALPHA_GRID:
                anchored = zg_val + alpha * zh_val
                tau, value = best_tau(y[iva], anchored)
                candidate = (value, -alpha, alpha, tau)
                if best_anchor is None or candidate > best_anchor:
                    best_anchor = candidate
            _, _, alpha, tau_anchor = best_anchor

            # Stage 2: fresh full outer-training fit for selected epoch.
            del head_inner
            if device == "cuda": torch.cuda.empty_cache()
            torch.manual_seed(fold_seed); np.random.seed(fold_seed)
            head_outer = new_head(ds_otr, device)
            train_fixed(head_outer, ds_otr, epochs=best_epoch, seed=fold_seed, device=device)
            h_train = head_logit(head_outer, ds_otr, device)
            h_test = head_logit(head_outer, ds_ote, device)
            hp2 = zfit(h_train); zh_test = zapply(h_test, hp2)
            g4_outer = g4_model(g4[otr], y[otr])
            g_train = g4_outer.decision_function(g4[otr]).astype(np.float64)
            g_test = g4_outer.decision_function(g4[ote]).astype(np.float64)
            gp2 = zfit(g_train); zg_test = zapply(g_test, gp2)

            pred_head = (zh_test >= tau_head).astype(np.int8)
            fused_test = zh_test + beta * zg_test
            pred_fusion = (fused_test >= tau_fusion).astype(np.int8)
            anchor_test = zg_test + alpha * zh_test
            pred_anchor = (anchor_test >= tau_anchor).astype(np.int8)
            all_scores[backbone_name][ote] = zh_test
            all_pred[backbone_name][ote] = pred_head
            all_scores[f"{backbone_name}_plus_G4"][ote] = fused_test
            all_pred[f"{backbone_name}_plus_G4"][ote] = pred_fusion
            all_scores[f"G4_anchor_plus_{backbone_name}"][ote] = anchor_test
            all_pred[f"G4_anchor_plus_{backbone_name}"][ote] = pred_anchor
            result = {
                "backbone": backbone_name, "outer_fold": fold,
                "n_outer_train": int(len(otr)), "n_outer_test": int(len(ote)),
                "n_outer_test_groups": int(len(np.unique(groups[ote]))),
                "best_epoch_inner": best_epoch,
                "head": {"inner_val_uar": inner_head_uar, "tau": tau_head,
                         **metric(y[ote], pred_head)},
                "plus_G4": {"beta": beta, "tau": tau_fusion,
                            "inner_val_uar": float(best_fusion[0]),
                            **metric(y[ote], pred_fusion)},
                "G4_anchor": {"alpha": alpha, "tau": tau_anchor,
                              "inner_val_uar": float(best_anchor[0]),
                              **metric(y[ote], pred_anchor)},
            }
            rows.append(result)
            print(f"  RESULT head={result['head']['uar']:.4f}  "
                  f"+G4={result['plus_G4']['uar']:.4f} beta={beta}  "
                  f"G4-anchor={result['G4_anchor']['uar']:.4f} alpha={alpha} "
                  f"epoch={best_epoch}")
            del head_outer
            if device == "cuda": torch.cuda.empty_cache()
        del full_ds
        gc.collect()

    summary = {}
    for name, pred in all_pred.items():
        if np.any(pred < 0): raise RuntimeError(f"incomplete OOF predictions for {name}")
        if name.startswith("G4_anchor_plus_"):
            key = "G4_anchor"
            backbone_name = name.removeprefix("G4_anchor_plus_")
        elif name.endswith("_plus_G4"):
            key = "plus_G4"
            backbone_name = name.removesuffix("_plus_G4")
        else:
            key = "head"
            backbone_name = name
        fold_uars = [r[key]["uar"] for r in rows if r["backbone"] == backbone_name]
        summary[name] = {
            "outer_oof": metric(y, pred),
            "fold_uar_mean": float(np.mean(fold_uars)),
            "fold_uar_std": float(np.std(fold_uars, ddof=1)),
            "fold_uars": fold_uars,
        }
    ranking = sorted(summary, key=lambda n: summary[n]["outer_oof"]["uar"], reverse=True)
    report = {
        "rung_id": "corrected_outer_cv_foundations",
        "protocol": {
            "selection_pool": "official Train only", "development_used": False,
            "outer_folds": 5, "inner_epoch_split": 5,
            "head": "uniform-init learned layer weighting + 128-d MLP",
            "final_outer_fit": "fresh full outer-train fit at inner-selected epoch",
            "fusion": "inner-selected residual G4 weight, including beta=0",
            "anchored_fusion": "G4 + bounded alpha*backbone, including alpha=0",
        },
        "folds": rows, "summary": summary, "ranking": ranking,
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    out = ROOT / "results" / "corrected_outer_cv_foundations.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    npz = ROOT / "results" / "corrected_outer_cv_foundations_oof.npz"
    np.savez_compressed(
        npz, files=np.asarray(files), y=y, groups=groups,
        **{f"score__{k}": v for k, v in all_scores.items()},
        **{f"pred__{k}": v for k, v in all_pred.items()},
    )
    print("\n=== FOUNDATION CORRECTED RANKING ===")
    for name in ranking:
        s = summary[name]
        print(f"  {name:<25} UAR={s['outer_oof']['uar']:.4f} "
              f"fold={s['fold_uar_mean']:.4f}+/-{s['fold_uar_std']:.4f}")
    print(f"[wrote] {out.relative_to(ROOT)} + {npz.relative_to(ROOT)} "
          f"elapsed={report['elapsed_minutes']:.1f} min")


if __name__ == "__main__":
    main()
