from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .head import LayerWeightedPooledHead


def make_balanced_sampler(train_ds, seed: int = 42) -> WeightedRandomSampler:
    """Guarantees ~50/50 class balance per batch via inverse-frequency weights."""
    labels = train_ds.get_labels()
    counts: dict[int, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    sample_weights = torch.tensor(
        [1.0 / counts[lab] for lab in labels], dtype=torch.double
    )
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights, num_samples=len(labels), replacement=True, generator=g
    )


def _pooled_collate(batch: list[dict]) -> dict:
    return {
        "pooled": torch.stack([b["pooled"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "file_name": [b["file_name"] for b in batch],
    }


def compute_uar(preds: np.ndarray, labels: np.ndarray) -> float:
    classes = np.unique(labels)
    recalls = []
    for c in classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        recalls.append(float((preds[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def per_class_recall(preds: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in np.unique(labels):
        mask = labels == c
        if mask.sum() == 0:
            continue
        out[int(c)] = float((preds[mask] == c).mean())
    return out


@dataclass
class TrainResult:
    best_val_uar: float
    best_epoch: int
    test_uar: float
    test_acc: float
    test_per_class_recall: dict[int, float]
    layer_weights: np.ndarray
    history: list[dict] = field(default_factory=list)


@torch.no_grad()
def evaluate(head: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, dict[int, float], np.ndarray, np.ndarray]:
    head.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        pooled = batch["pooled"].to(device)
        labels = batch["label"].to(device)
        logits, _ = head(pooled)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    uar = compute_uar(preds, labels)
    acc = float((preds == labels).mean())
    return uar, acc, per_class_recall(preds, labels), preds, labels


@torch.no_grad()
def predict_probs(head: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (P(class=1) per sample, ground-truth labels)."""
    head.eval()
    all_p: list[np.ndarray] = []
    all_l: list[np.ndarray] = []
    for batch in loader:
        pooled = batch["pooled"].to(device)
        logits, _ = head(pooled)
        p = torch.softmax(logits, dim=-1)[:, 1]
        all_p.append(p.cpu().numpy())
        lab = batch["label"]
        all_l.append(lab.cpu().numpy() if torch.is_tensor(lab) else np.asarray(lab))
    return np.concatenate(all_p), np.concatenate(all_l)


def sweep_threshold(
    head: nn.Module,
    loader: DataLoader,
    device: str,
    grid: Optional[np.ndarray] = None,
) -> tuple[float, float, list[tuple[float, float]]]:
    """
    Pick the decision threshold tau on P(class=1) that maximises UAR on `loader`.

    `loader` MUST be a held-out slice of *train*, never devel. Using devel to
    select tau and then reporting UAR on the same devel is dev-tuning.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    probs, labels = predict_probs(head, loader, device)
    best_tau, best_uar = 0.5, -1.0
    sweep: list[tuple[float, float]] = []
    for t in grid:
        preds = (probs >= t).astype(np.int64)
        u = compute_uar(preds, labels)
        sweep.append((float(t), float(u)))
        if u > best_uar:
            best_uar, best_tau = u, float(t)
    return best_tau, best_uar, sweep


@torch.no_grad()
def evaluate_at_threshold(
    head: nn.Module, loader: DataLoader, device: str, tau: float
) -> tuple[float, float, dict[int, float]]:
    probs, labels = predict_probs(head, loader, device)
    preds = (probs >= tau).astype(np.int64)
    uar = compute_uar(preds, labels)
    acc = float((preds == labels).mean())
    return uar, acc, per_class_recall(preds, labels)


def train_head(
    head: LayerWeightedPooledHead,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Optional[Dataset] = None,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    base_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 6,
    class_weights: Optional[torch.Tensor] = None,
    balanced_sampler: bool = True,
    fit_scaler: bool = True,
    device: str = "cuda",
    ckpt_path: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> TrainResult:
    """
    Trains a layer-weighted head on pre-extracted pooled features.

    Selects the best epoch by val UAR, then reports a single honest read on
    test_ds (which should be the ComParE devel split — speaker-disjoint from
    train, the closest proxy for the withheld challenge test set).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = head.to(device)

    if balanced_sampler:
        sampler = make_balanced_sampler(train_ds, seed=seed)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, collate_fn=_pooled_collate, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=_pooled_collate, drop_last=False,
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_pooled_collate,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_pooled_collate,
        )

    if fit_scaler and hasattr(head, "scaler"):
        fit_loader = DataLoader(
            train_ds, batch_size=256, shuffle=False,
            num_workers=num_workers, collate_fn=_pooled_collate,
        )
        head.scaler.to(device)
        head.scaler.fit(fit_loader)

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optim = torch.optim.AdamW(head.param_groups(base_lr=base_lr), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    try:
        from tqdm.auto import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(it, **kw): return it

    print(f"[train] device={device}  train={len(train_ds)}  val={len(val_ds)}"
          f"{'  test=' + str(len(test_ds)) if test_ds is not None else ''}")
    print(f"[train] class_weights={class_weights.tolist() if class_weights is not None else None}")
    print(f"[train] balanced_sampler={balanced_sampler}  fit_scaler={fit_scaler}")
    print(f"[train] epochs={epochs}  batch={batch_size}  lr={base_lr}  patience={early_stop_patience}")
    print(f"[train] head params: n_layers={head.n_layers}  stat_dim={head.stat_dim}  proj_dim={head.proj_dim}")
    print(f"[train] majority-class baseline UAR = 0.500 by definition")

    # Sanity check: one untrained forward pass to catch NaN/inf/collapse at epoch 0
    head.train()
    diag_batch = next(iter(train_loader))
    diag_pooled = diag_batch["pooled"].to(device)
    diag_labels = diag_batch["label"].to(device)
    with torch.no_grad():
        diag_logits, _ = head(diag_pooled)
    diag_probs = torch.softmax(diag_logits, dim=-1)
    diag_loss = loss_fn(diag_logits, diag_labels).item()
    print(
        f"[diag] untrained batch: "
        f"logit_range=[{diag_logits.min().item():+.3f}, {diag_logits.max().item():+.3f}]  "
        f"mean_p_C={diag_probs[:, 1].mean().item():.3f}  "
        f"loss={diag_loss:.4f}  "
        f"any_nan={bool(torch.isnan(diag_logits).any())}  "
        f"any_inf={bool(torch.isinf(diag_logits).any())}"
    )
    if torch.isnan(diag_logits).any() or torch.isinf(diag_logits).any():
        raise RuntimeError("Untrained forward produced NaN/inf — fix features/head before training.")
    print()

    best_val_uar = -1.0
    best_epoch = -1
    best_state: Optional[dict] = None
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        head.train()
        running_loss = 0.0
        n_seen = 0
        correct = 0

        pbar = _tqdm(train_loader, desc=f"ep {epoch:02d}/{epochs}", leave=False)
        for batch in pbar:
            pooled = batch["pooled"].to(device)
            labels = batch["label"].to(device)

            logits, _ = head(pooled)
            loss = loss_fn(logits, labels)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optim.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            correct += int((logits.argmax(-1) == labels).sum().item())
            pbar.set_postfix(loss=f"{running_loss / n_seen:.4f}", acc=f"{correct/n_seen:.3f}")

        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)
        train_acc = correct / max(n_seen, 1)

        val_uar, val_acc, val_pcr, _, _ = evaluate(head, val_loader, device)

        with torch.no_grad():
            w = head.layer_softmax().detach().cpu().numpy()
        top_layers = np.argsort(w)[::-1][:3]
        top_str = ", ".join(f"L{int(i)}:{w[i]:.2f}" for i in top_layers)

        improved = val_uar > best_val_uar
        marker = "  *" if improved else ""
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  |  "
            f"val_UAR={val_uar:.4f} val_acc={val_acc:.3f} "
            f"(C={val_pcr.get(1, 0.0):.3f} NC={val_pcr.get(0, 0.0):.3f})  |  "
            f"top_layers=[{top_str}]{marker}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_uar": val_uar,
            "val_acc": val_acc,
            "val_recall_C": val_pcr.get(1, 0.0),
            "val_recall_NC": val_pcr.get(0, 0.0),
            "layer_weights": w.tolist(),
        })

        if improved:
            best_val_uar = val_uar
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": best_state,
                    "val_uar": val_uar,
                    "epoch": epoch,
                    "n_layers": head.n_layers,
                    "stat_dim": head.stat_dim,
                    "proj_dim": head.proj_dim,
                }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n[train] early stop at epoch {epoch} "
                      f"(no val_UAR improvement for {early_stop_patience} epochs)")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    print(f"\n[train] best val_UAR={best_val_uar:.4f} at epoch {best_epoch}")

    test_uar = 0.0
    test_acc = 0.0
    test_pcr: dict[int, float] = {}
    if test_loader is not None:
        test_uar, test_acc, test_pcr, _, _ = evaluate(head, test_loader, device)
        print(
            f"\n[HELD-OUT TEST] devel (speaker-disjoint from train, proxy for hidden test):\n"
            f"    UAR       = {test_uar:.4f}\n"
            f"    accuracy  = {test_acc:.4f}\n"
            f"    recall_C  = {test_pcr.get(1, 0.0):.4f}\n"
            f"    recall_NC = {test_pcr.get(0, 0.0):.4f}"
        )
        gap = best_val_uar - test_uar
        verdict = "optimistic" if gap > 0.02 else "consistent" if abs(gap) <= 0.02 else "pessimistic"
        print(f"    val-to-test gap = {gap:+.4f}  ({verdict})")

    with torch.no_grad():
        final_w = head.layer_softmax().detach().cpu().numpy()

    return TrainResult(
        best_val_uar=best_val_uar,
        best_epoch=best_epoch,
        test_uar=test_uar,
        test_acc=test_acc,
        test_per_class_recall=test_pcr,
        layer_weights=final_w,
        history=history,
    )
