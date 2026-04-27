"""
A5b — constrained late-fusion math (plan.md § 5.3).

Final classifier (per utterance):

    final_logit = logit_A2 + beta * mean_g( zscore_g( logit_g ) )

where the mean is over the K admitted groups (chosen by subtractive_honesty
ranking from A5a, with `label_gain > 0` filtered out). beta and K are swept
on `train_threshold`; the locked (beta*, K*, tau*) is reported once on
`devel_test`.

Key choices:
  - `logit_g` is `clf.decision_function(scaler.transform(X))` — the raw
    log-odds before sigmoid. This is on the same scale as `logits[:,1] -
    logits[:,0]` from a 2-class softmax head, which is how we obtain the A2
    logit. So adding them makes physical sense without a learned scale.
  - z-scoring per group uses mean/std fit on `train_fit` predictions. This
    removes per-group scale differences (G4_energy logits naturally span a
    wider range than G2_prosody logits) so beta has a single interpretation
    across the K groups.
  - Per-group cold probe is fit with the **identical recipe** as
    honesty.probe.cold_probe (StandardScaler + LogisticRegression, C=1.0,
    `class_weight="balanced"`, lbfgs, max_iter=2000, fixed seed). So the
    UAR-on-devel_val we already audited is exactly what this module sees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score


# ---------------------------------------------------------------------------
# Per-group cold probe — same recipe as honesty.probe.cold_probe.
# ---------------------------------------------------------------------------
def fit_cold_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *, C: float = 1.0, max_iter: int = 2000, seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(
        C=C, class_weight="balanced",
        solver="lbfgs", max_iter=max_iter, random_state=seed,
    )
    clf.fit(scaler.transform(X_train), y_train)
    return clf, scaler


def predict_logit(
    clf: LogisticRegression,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """Returns log-odds (binary logit), shape [N]."""
    return clf.decision_function(scaler.transform(X)).astype(np.float64)


# ---------------------------------------------------------------------------
# Per-group z-score (parameters fit on train_fit, applied to all splits).
# ---------------------------------------------------------------------------
@dataclass
class ZScore:
    mu: float
    sigma: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / max(self.sigma, 1e-8)


def fit_zscore(logit_train: np.ndarray) -> ZScore:
    return ZScore(mu=float(logit_train.mean()), sigma=float(logit_train.std() + 1e-8))


# ---------------------------------------------------------------------------
# Fusion + threshold sweep.
# ---------------------------------------------------------------------------
def fuse(
    logit_a2: np.ndarray,
    admitted_z_logits: list[np.ndarray],   # one [N] array per admitted group
    beta: float,
) -> np.ndarray:
    """final_logit = logit_a2 + beta * mean_g(z_g)."""
    if not admitted_z_logits:
        return logit_a2.astype(np.float64)
    stack = np.stack(admitted_z_logits, axis=1)         # [N, K]
    return logit_a2.astype(np.float64) + beta * stack.mean(axis=1)


def uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * (
        recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        + recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    )


def sweep_tau(
    fused_logit: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Pick the threshold on the fused logit (NOT on a sigmoided probability)
    that maximises UAR. The fused logit is a real-valued score; tau lives in
    logit space."""
    if grid is None:
        # Range generous enough to cover the typical fused-logit spread.
        grid = np.linspace(-4.0, 4.0, 321)
    best_tau, best_u = 0.0, -1.0
    for t in grid:
        pred = (fused_logit >= t).astype(np.int64)
        u = uar(y, pred)
        if u > best_u:
            best_u, best_tau = u, float(t)
    return best_tau, best_u


def evaluate_at_tau(
    fused_logit: np.ndarray, y: np.ndarray, tau: float,
) -> dict[str, float]:
    pred = (fused_logit >= tau).astype(np.int64)
    return {
        "uar": uar(y, pred),
        "recall_C": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_NC": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "acc": float((pred == y).mean()),
    }
