"""Evaluation-independent threshold check for the fixed G4+CQT fusion.

For each official direction, the model and every candidate deployment
threshold are learned using only the fitting side.  The opposite official side
is touched once for evaluation.  Side-local ECAPA clustering is used only to
make speaker-proxy-disjoint OOF predictions for threshold estimation; no
embedding, grouping, score, or label from the evaluation side enters fitting.

This deliberately distinguishes three questions:
  1. Does fusion rank examples better than either branch? (AUC)
  2. Does the frozen threshold 0 convert that ranking into UAR? (UAR@0)
  3. Does a train-OOF UAR-optimal threshold transfer out of side?

The 43% predicted-cold policy is included only as a diagnostic. UAR does not
imply a target prediction prevalence, so it is never recommended by this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

from reconcile_cqt_protocol import ROOT, load_features, load_labels, model


CV_SEEDS = [42, 1, 2, 3, 5, 20260720]
N_SPLITS = 5
N_PROXY_SPEAKERS = 210
TARGET_RATE_DIAGNOSTIC = 0.43
N_BOOT = 30000


def metrics(y: np.ndarray, score: np.ndarray, tau: float) -> dict[str, float]:
    pred = (score >= tau).astype(np.int8)
    return {
        "uar": float(balanced_accuracy_score(y, pred)),
        "recall_C": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_NC": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "predicted_cold_rate": float(pred.mean()),
        "threshold": float(tau),
    }


def cluster_bootstrap_uar(
    y: np.ndarray, predictions: dict[str, np.ndarray], groups: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    """Resample proxy speakers while retaining cold/non-cold counts within mixed groups."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(unique), np.full(len(unique), 1.0 / len(unique)), size=N_BOOT
    )
    cold_total = np.asarray([np.sum((groups == group) & (y == 1)) for group in unique])
    noncold_total = np.asarray([np.sum((groups == group) & (y == 0)) for group in unique])
    draws = {}
    for name, pred in predictions.items():
        cold_correct = np.asarray([
            np.sum((groups == group) & (y == 1) & (pred == 1)) for group in unique
        ])
        noncold_correct = np.asarray([
            np.sum((groups == group) & (y == 0) & (pred == 0)) for group in unique
        ])
        draws[name] = 0.5 * (
            (weights @ cold_correct) / (weights @ cold_total)
            + (weights @ noncold_correct) / (weights @ noncold_total)
        )
    return draws


def paired_interval(draws: dict[str, np.ndarray], a: str, b: str) -> dict[str, float | list[float]]:
    delta = draws[a] - draws[b]
    return {
        "mean_delta": float(delta.mean()),
        "paired_95_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
        "probability_positive": float(np.mean(delta > 0)),
    }


def best_uar_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, score)
    uar = 0.5 * (tpr + 1.0 - fpr)
    best = np.flatnonzero(np.isclose(uar, uar.max(), rtol=0.0, atol=1e-12))
    # Stable tie-break: the least extreme threshold has the weakest tuning.
    chosen = best[np.argmin(np.abs(thresholds[best]))]
    return float(thresholds[chosen]), float(uar[chosen])


def fit_eval_scores(
    x_fit: dict[str, np.ndarray], y_fit: np.ndarray, x_eval: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    raw: dict[str, np.ndarray] = {}
    z: dict[str, np.ndarray] = {}
    for name in ("G4", "G9"):
        fitted = model(seed=42).fit(x_fit[name], y_fit)
        fit_score = fitted.decision_function(x_fit[name])
        eval_score = fitted.decision_function(x_eval[name])
        raw[name] = eval_score
        z[name] = (eval_score - fit_score.mean()) / max(fit_score.std(), 1e-8)
    return {"G4_raw": raw["G4"], "G9_raw": raw["G9"],
            "fusion": 0.5 * (z["G4"] + z["G9"])}


def oof_fusion(
    x: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, seed: int
) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    out = np.full(len(y), np.nan, dtype=np.float64)
    for train, valid in cv.split(np.zeros(len(y)), y, groups):
        z_valid: dict[str, np.ndarray] = {}
        for name in ("G4", "G9"):
            fitted = model(seed=seed).fit(x[name][train], y[train])
            train_score = fitted.decision_function(x[name][train])
            valid_score = fitted.decision_function(x[name][valid])
            z_valid[name] = ((valid_score - train_score.mean())
                             / max(train_score.std(), 1e-8))
        out[valid] = 0.5 * (z_valid["G4"] + z_valid["G9"])
    if np.isnan(out).any():
        raise RuntimeError("OOF score array is incomplete")
    return out


def side_local_groups(stems: list[str], side: str) -> tuple[np.ndarray, dict]:
    archive = np.load(ROOT / "cache/ecapa-voxceleb/ecapa_embeddings.npz", allow_pickle=True)
    all_stems = archive["stems"].astype(str)
    all_sides = archive["split"].astype(str)
    embeddings = archive["embeddings"].astype(np.float32)
    mask = all_sides == side
    side_stems = all_stems[mask]
    index = {stem: i for i, stem in enumerate(side_stems)}
    side_x = normalize(embeddings[mask])
    ordered_x = np.vstack([side_x[index[stem]] for stem in stems])
    fitted = KMeans(n_clusters=N_PROXY_SPEAKERS, n_init=10, random_state=42).fit(ordered_x)
    groups = fitted.labels_.astype(np.int32)
    return groups, {
        "construction": f"KMeans(k={N_PROXY_SPEAKERS}, n_init=10, seed=42) on {side} ECAPA only",
        "n_groups": int(len(np.unique(groups))),
    }


def direction_report(
    fit_side: str, eval_side: str,
    x: dict[str, dict[str, np.ndarray]], y: dict[str, np.ndarray],
    fit_groups: np.ndarray, eval_groups: np.ndarray, bootstrap_seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    eval_scores = fit_eval_scores(x[fit_side], y[fit_side], x[eval_side])
    seed_rows = []
    for seed in CV_SEEDS:
        score = oof_fusion(x[fit_side], y[fit_side], fit_groups, seed)
        tau, selected_uar = best_uar_threshold(y[fit_side], score)
        rate_tau = float(np.quantile(score, 1.0 - TARGET_RATE_DIAGNOSTIC))
        seed_rows.append({
            "seed": seed,
            "oof_at_zero": metrics(y[fit_side], score, 0.0),
            "oof_selected": metrics(y[fit_side], score, tau),
            "selected_threshold": tau,
            "roc_selected_uar_check": selected_uar,
            "eval_at_selected_threshold": metrics(y[eval_side], eval_scores["fusion"], tau),
            "rate43_threshold_diagnostic": rate_tau,
            "eval_at_rate43_threshold_diagnostic": metrics(y[eval_side], eval_scores["fusion"], rate_tau),
        })

    taus = np.asarray([row["selected_threshold"] for row in seed_rows])
    rate_taus = np.asarray([row["rate43_threshold_diagnostic"] for row in seed_rows])
    median_tau = float(np.median(taus))
    median_rate_tau = float(np.median(rate_taus))
    result = {
        "fit_side": fit_side,
        "eval_side": eval_side,
        "threshold_free_auc": {
            name: float(roc_auc_score(y[eval_side], eval_scores[key]))
            for name, key in (("G4", "G4_raw"), ("G9_CQT", "G9_raw"), ("fusion", "fusion"))
        },
        "standalone_raw_threshold_zero": {
            "G4": metrics(y[eval_side], eval_scores["G4_raw"], 0.0),
            "G9_CQT": metrics(y[eval_side], eval_scores["G9_raw"], 0.0),
        },
        "fusion_fixed_zero": metrics(y[eval_side], eval_scores["fusion"], 0.0),
        "fit_side_oof_thresholds": {
            "values": taus.tolist(),
            "mean": float(taus.mean()),
            "std": float(taus.std(ddof=1)),
            "median_policy_threshold": median_tau,
            "seed_rows": seed_rows,
        },
        "fusion_oof_median_threshold": metrics(y[eval_side], eval_scores["fusion"], median_tau),
        "fusion_rate43_diagnostic": metrics(y[eval_side], eval_scores["fusion"], median_rate_tau),
        "rate43_median_threshold": median_rate_tau,
        "eval_uar_across_seed_specific_oof_thresholds": [
            row["eval_at_selected_threshold"]["uar"] for row in seed_rows
        ],
    }
    result["threshold_free_auc"]["fusion_minus_G9"] = (
        result["threshold_free_auc"]["fusion"] - result["threshold_free_auc"]["G9_CQT"]
    )
    result["oof_median_minus_fixed_zero_uar"] = (
        result["fusion_oof_median_threshold"]["uar"] - result["fusion_fixed_zero"]["uar"]
    )
    bootstrap_predictions = {
        "G9_raw_zero": (eval_scores["G9_raw"] >= 0.0).astype(np.int8),
        "fusion_fixed_zero": (eval_scores["fusion"] >= 0.0).astype(np.int8),
        "fusion_oof_median_threshold": (eval_scores["fusion"] >= median_tau).astype(np.int8),
    }
    draws = cluster_bootstrap_uar(
        y[eval_side], bootstrap_predictions, eval_groups, bootstrap_seed
    )
    result["proxy_cluster_uncertainty"] = {
        "eval_groups": int(len(np.unique(eval_groups))),
        "fusion_oof_threshold_vs_G9_raw_zero": paired_interval(
            draws, "fusion_oof_median_threshold", "G9_raw_zero"
        ),
        "fusion_oof_threshold_vs_fusion_fixed_zero": paired_interval(
            draws, "fusion_oof_median_threshold", "fusion_fixed_zero"
        ),
        "fusion_fixed_zero_vs_G9_raw_zero": paired_interval(
            draws, "fusion_fixed_zero", "G9_raw_zero"
        ),
    }
    return result, draws


def main() -> None:
    labels = load_labels(str(ROOT / "dataset/ComParE2017_Cold_4students"))
    files = {side: sorted(f for f in labels if f.startswith(side + "_"))
             for side in ("train", "devel")}
    stems = {side: [Path(f).stem for f in files[side]] for side in files}
    y = {side: np.asarray([labels[f] for f in files[side]], dtype=np.int8) for side in files}
    x = {side: load_features(stems[side]) for side in stems}

    groups = {}; grouping = {}
    for side in ("train", "devel"):
        print(f"[groups] fitting side-local {side} ECAPA clustering", flush=True)
        groups[side], grouping[side] = side_local_groups(stems[side], side)
        grouping[side]["mixed_label_groups"] = int(sum(
            len(np.unique(y[side][groups[side] == group])) > 1
            for group in np.unique(groups[side])
        ))

    directions = {}; bootstrap_draws = {}
    for direction_number, (fit_side, eval_side) in enumerate(
        (("train", "devel"), ("devel", "train")), 1
    ):
        key = f"{fit_side}_to_{eval_side}"
        print(f"[threshold transfer] {key}", flush=True)
        directions[key], bootstrap_draws[key] = direction_report(
            fit_side, eval_side, x, y, groups[fit_side], groups[eval_side],
            bootstrap_seed=20260720 + direction_number,
        )

    fixed_mean = float(np.mean([row["fusion_fixed_zero"]["uar"] for row in directions.values()]))
    learned_mean = float(np.mean([row["fusion_oof_median_threshold"]["uar"] for row in directions.values()]))
    rate_mean = float(np.mean([row["fusion_rate43_diagnostic"]["uar"] for row in directions.values()]))
    learned_deltas = [row["oof_median_minus_fixed_zero_uar"] for row in directions.values()]
    bidirectional_draws = {
        name: 0.5 * (
            bootstrap_draws["train_to_devel"][name]
            + bootstrap_draws["devel_to_train"][name]
        )
        for name in ("G9_raw_zero", "fusion_fixed_zero", "fusion_oof_median_threshold")
    }
    recommendation = (
        "keep_fixed_zero" if any(delta <= 0.0 for delta in learned_deltas)
        else "oof_median_threshold_transfers_in_both_directions"
    )
    report = {
        "protocol": {
            "evaluation_independence": "all model and threshold choices use fitting side only",
            "threshold_estimation": "median of six 5-fold side-local speaker-proxy-disjoint OOF UAR optima",
            "grouping": grouping,
            "fixed_architecture": "G4 and G9 balanced LR; train-logit zscore; equal average",
            "rate43_status": "diagnostic only; UAR does not imply a target predicted-cold prevalence",
        },
        "directions": directions,
        "bidirectional_summary": {
            "fixed_zero_mean_uar": fixed_mean,
            "oof_median_threshold_mean_uar": learned_mean,
            "oof_median_minus_zero_mean_uar": learned_mean - fixed_mean,
            "rate43_diagnostic_mean_uar": rate_mean,
            "directional_oof_median_minus_zero": learned_deltas,
            "proxy_cluster_uncertainty": {
                "fusion_oof_threshold_vs_G9_raw_zero": paired_interval(
                    bidirectional_draws, "fusion_oof_median_threshold", "G9_raw_zero"
                ),
                "fusion_oof_threshold_vs_fusion_fixed_zero": paired_interval(
                    bidirectional_draws, "fusion_oof_median_threshold", "fusion_fixed_zero"
                ),
                "fusion_fixed_zero_vs_G9_raw_zero": paired_interval(
                    bidirectional_draws, "fusion_fixed_zero", "G9_raw_zero"
                ),
            },
        },
        "recommendation": recommendation,
        "limitations": [
            "True speaker IDs are unavailable; side-local ECAPA clusters are proxies.",
            "Development has historical selection exposure, so this remains a diagnostic rather than pristine model selection.",
            "A threshold selected to maximize finite-sample OOF UAR can itself overfit; median aggregation reduces but does not remove that risk.",
        ],
    }
    out = ROOT / "results/eval_independent_threshold_policy.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"bidirectional_summary": report["bidirectional_summary"],
                      "recommendation": recommendation,
                      "directions": {name: {
                          "auc": row["threshold_free_auc"],
                          "standalone": row["standalone_raw_threshold_zero"],
                          "fusion_fixed_zero": row["fusion_fixed_zero"],
                          "oof_threshold_values": row["fit_side_oof_thresholds"]["values"],
                          "fusion_oof_median_threshold": row["fusion_oof_median_threshold"],
                          "fusion_rate43_diagnostic": row["fusion_rate43_diagnostic"],
                      } for name, row in directions.items()}}, indent=2))
    print(f"[wrote] {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
