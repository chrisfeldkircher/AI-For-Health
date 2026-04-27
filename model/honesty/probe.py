"""
Matched linear probes for the A5a honesty audit (plan.md § 5.1, § 5.6).

Two probes per feature group, same architecture and same input dimensionality:

  - cold_probe   : binary logistic regression on Cold labels with balanced
                   class weights, evaluated by UAR = (recall_C + recall_NC)/2.
  - speaker_probe: multinomial logistic regression on pseudo-speaker IDs
                   (k=210 by default), evaluated by top-1 accuracy and NMI.

Linear-only by design. plan.md § 5.6: "Per-group probe = linear logistic
regression, not an MLP. If a group needs nonlinearity to predict cold, that's
a signal the group should be sub-divided." A nonlinear probe would also
re-introduce the A3 failure mode where an MLP learns to combine speaker-rich
dimensions into a cold prediction.

Standardisation: per-group StandardScaler fit on train_fit, applied to eval.
L2 regularisation at C=1.0 (sklearn default). No hyperparameter search — keep
the probe a fixed measurement instrument so honesty rows stay comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, recall_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ColdProbeResult:
    uar: float
    recall_pos: float
    recall_neg: float


@dataclass
class SpeakerProbeResult:
    top1: float
    nmi: float
    n_classes: int


def cold_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval:  np.ndarray, y_eval:  np.ndarray,
    *, C: float = 1.0, max_iter: int = 2000, seed: int = 42,
) -> ColdProbeResult:
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xe = scaler.transform(X_eval)
    clf = LogisticRegression(
        C=C, class_weight="balanced",
        solver="lbfgs", max_iter=max_iter, random_state=seed,
    )
    clf.fit(Xt, y_train)
    pred = clf.predict(Xe)
    rec_pos = float(recall_score(y_eval, pred, pos_label=1, zero_division=0))
    rec_neg = float(recall_score(y_eval, pred, pos_label=0, zero_division=0))
    return ColdProbeResult(
        uar=0.5 * (rec_pos + rec_neg),
        recall_pos=rec_pos,
        recall_neg=rec_neg,
    )


def speaker_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval:  np.ndarray, y_eval:  np.ndarray,
    *, C: float = 1.0, max_iter: int = 2000, seed: int = 42,
) -> SpeakerProbeResult:
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xe = scaler.transform(X_eval)
    clf = LogisticRegression(
        C=C, solver="lbfgs", max_iter=max_iter, random_state=seed,
    )
    clf.fit(Xt, y_train)
    pred = clf.predict(Xe)
    top1 = float((pred == y_eval).mean())
    nmi  = float(normalized_mutual_info_score(y_eval, pred))
    return SpeakerProbeResult(
        top1=top1, nmi=nmi,
        n_classes=int(np.unique(y_train).size),
    )
