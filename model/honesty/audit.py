"""
A5a honesty audit: combine cold + speaker probes per feature group, write one
row per group to results/A5a_honesty.csv.

Two complementary honesty forms (plan.md § 5.1):

  label_gain_g           = UAR_g   - 0.50
  speaker_gain_g         = top1_g  - 1/k         (chance-floor normalised)
  ratio_honesty_g        = label_gain / (speaker_gain + EPS)
  subtractive_honesty_g  = label_gain - lambda * speaker_gain

The ratio form is parameter-free and intuitive but unstable when speaker_gain
is small. The subtractive form is sharper for the paper's claim and survives
near-zero speaker_gain better. We report subtractive at lambda in {0.5, 1, 2}
as a sensitivity column.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .probe import cold_probe, speaker_probe


EPS = 1e-3


@dataclass
class HonestyRow:
    group: str
    dim: int
    n_train: int
    n_eval: int
    uar: float
    recall_C: float
    recall_NC: float
    speaker_top1: float
    speaker_nmi: float
    n_pseudo_classes: int
    label_gain: float
    speaker_gain: float
    ratio_honesty: float
    subtractive_honesty_lam0p5: float
    subtractive_honesty_lam1: float
    subtractive_honesty_lam2: float


def audit_group(
    name: str,
    X_train: np.ndarray, X_eval: np.ndarray,
    y_cold_train: np.ndarray, y_cold_eval: np.ndarray,
    y_pseudo_train: np.ndarray, y_pseudo_eval: np.ndarray,
    *, C: float = 1.0, max_iter: int = 2000, seed: int = 42,
    verbose: bool = True,
) -> HonestyRow:
    if verbose:
        print(f"[honesty:{name}] X_train={X_train.shape}  X_eval={X_eval.shape}")
    cold = cold_probe(X_train, y_cold_train, X_eval, y_cold_eval,
                      C=C, max_iter=max_iter, seed=seed)
    spk  = speaker_probe(X_train, y_pseudo_train, X_eval, y_pseudo_eval,
                         C=C, max_iter=max_iter, seed=seed)

    label_gain   = cold.uar - 0.50
    speaker_gain = spk.top1 - 1.0 / max(spk.n_classes, 1)
    ratio        = label_gain / (speaker_gain + EPS)

    row = HonestyRow(
        group=name,
        dim=int(X_train.shape[1]),
        n_train=int(X_train.shape[0]),
        n_eval=int(X_eval.shape[0]),
        uar=cold.uar, recall_C=cold.recall_pos, recall_NC=cold.recall_neg,
        speaker_top1=spk.top1, speaker_nmi=spk.nmi,
        n_pseudo_classes=spk.n_classes,
        label_gain=label_gain,
        speaker_gain=speaker_gain,
        ratio_honesty=ratio,
        subtractive_honesty_lam0p5=label_gain - 0.5 * speaker_gain,
        subtractive_honesty_lam1=label_gain - 1.0 * speaker_gain,
        subtractive_honesty_lam2=label_gain - 2.0 * speaker_gain,
    )
    if verbose:
        print(f"  UAR={cold.uar:.4f}  (recall_C={cold.recall_pos:.3f}  "
              f"recall_NC={cold.recall_neg:.3f}  label_gain={label_gain:+.4f})")
        print(f"  spk top1={spk.top1:.4f}  NMI={spk.nmi:.4f}  "
              f"(speaker_gain={speaker_gain:+.4f})")
        print(f"  ratio={ratio:+.3f}  "
              f"subtractive(lam=1)={row.subtractive_honesty_lam1:+.4f}")
    return row


def append_to_csv(row: HonestyRow, csv_path: str | Path) -> None:
    """Append-or-replace a single group's row in the honesty CSV.

    Idempotent — re-running the audit for an existing group overwrites that
    row instead of duplicating it, so we can re-extract one group without
    losing the others' results.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(row).keys())

    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("group") != row.group:
                    rows.append(r)

    rows.append({k: v for k, v in asdict(row).items()})
    rows.sort(key=lambda r: str(r["group"]))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
