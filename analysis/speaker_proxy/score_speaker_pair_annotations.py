"""Score two independent blinded URTIC speaker-pair annotations.

The method key is opened only at scoring time. Pairwise method metrics describe
the disagreement-enriched audit set and must not be interpreted as population
accuracy over all possible URTIC pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)


VALID = {"same": 1, "different": 0, "unsure": -1}


def load_annotation(path: Path, expected_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"pair_id", "same_speaker"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(frame.columns))}")
    if frame["pair_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate pair IDs")
    frame = frame.set_index("pair_id").reindex(expected_ids)
    if frame["same_speaker"].isna().any():
        raise ValueError(f"{path}: missing expected pair IDs")
    normalized = frame["same_speaker"].str.strip().str.lower()
    bad = sorted(set(normalized) - set(VALID))
    if bad:
        raise ValueError(f"{path}: invalid labels {bad}; use same/different/unsure")
    frame["numeric"] = normalized.map(VALID).astype(int)
    return frame


def method_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "n_consensus_pairs": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "same_speaker_recall": float(tp / (tp + fn)) if tp + fn else None,
        "different_speaker_recall": float(tn / (tn + fp)) if tn + fp else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", nargs=2, type=Path)
    parser.add_argument(
        "--key", type=Path, default=Path("results/speaker_pair_annotation_key.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/speaker_pair_annotation_scores.json")
    )
    args = parser.parse_args()

    key = json.loads(args.key.read_text(encoding="utf-8"))
    pairs = pd.DataFrame(key["pairs"]).set_index("pair_id")
    expected = list(pairs.index)
    a = load_annotation(args.annotations[0], expected)
    b = load_annotation(args.annotations[1], expected)
    both_decisive = (a["numeric"] >= 0) & (b["numeric"] >= 0)
    agreement = both_decisive & (a["numeric"] == b["numeric"])
    conflict = both_decisive & (a["numeric"] != b["numeric"])
    any_unsure = (a["numeric"] < 0) | (b["numeric"] < 0)

    y_a = a.loc[both_decisive, "numeric"].to_numpy()
    y_b = b.loc[both_decisive, "numeric"].to_numpy()
    consensus_ids = a.index[agreement]
    truth = a.loc[consensus_ids, "numeric"].to_numpy()

    predictions = {
        "ecapa_kmeans": pairs.loc[consensus_ids, "ecapa_same"].map(
            lambda value: int(value["kmeans"])
        ).to_numpy(),
        "ecapa_agglomerative": pairs.loc[consensus_ids, "ecapa_same"].map(
            lambda value: int(value["agglomerative"])
        ).to_numpy(),
        "ecapa_spectral": pairs.loc[consensus_ids, "ecapa_same"].map(
            lambda value: int(value["spectral"])
        ).to_numpy(),
        "trillsson1_kmeans": pairs.loc[consensus_ids, "trillsson_same"].astype(int).to_numpy(),
    }
    ecapa_votes = np.column_stack(
        [predictions[name] for name in ("ecapa_kmeans", "ecapa_agglomerative", "ecapa_spectral")]
    )
    predictions["ecapa_majority_consensus"] = (ecapa_votes.sum(axis=1) >= 2).astype(int)

    by_stratum = {}
    for stratum, rows in pairs.loc[consensus_ids].groupby("stratum"):
        ids = rows.index
        by_stratum[stratum] = {
            "n_consensus": int(len(ids)),
            "human_same_fraction": float(a.loc[ids, "numeric"].mean()),
        }

    report = {
        "protocol": key["protocol"],
        "annotations": [str(path) for path in args.annotations],
        "inter_rater": {
            "n_pairs": len(expected),
            "both_decisive": int(both_decisive.sum()),
            "agreed_decisive_consensus": int(agreement.sum()),
            "decisive_conflicts_requiring_adjudication": int(conflict.sum()),
            "any_unsure": int(any_unsure.sum()),
            "raw_agreement_when_both_decisive": float(np.mean(y_a == y_b)) if len(y_a) else None,
            "cohen_kappa_when_both_decisive": float(cohen_kappa_score(y_a, y_b)) if len(y_a) else None,
            "conflict_pair_ids": list(a.index[conflict]),
        },
        "method_metrics_on_agreed_decisive_pairs": {
            name: method_metrics(truth, prediction)
            for name, prediction in predictions.items()
        },
        "human_consensus_by_sampling_stratum": by_stratum,
        "interpretation_guardrail": (
            "This is a disagreement-enriched, balanced case-control audit. Compare methods on "
            "the same pairs using balanced accuracy and by-stratum behavior; do not report these "
            "values as prevalence-weighted accuracy over all URTIC pairs. Resolve conflicts by a "
            "blinded third adjudication before any final headline analysis."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
