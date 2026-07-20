"""Repeated grouped-CV and paired cluster bootstrap for fixed G4+G9 fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from reconcile_cqt_protocol import CV_SEEDS, ROOT, load_features, load_labels, metric, model
from speakers.cluster import load_pseudo_speakers


N_BOOT = 20000
BOOT_SEED = 20260720


def fit_seed(y, x, groups, split_seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    pred = {name: np.full(len(y), -1, dtype=np.int8) for name in ("G4", "G9", "G4_G9_equal")}
    fold_rows = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        z = {}
        row = {"fold": fold, "models": {}}
        for name in ("G4", "G9"):
            fitted = model().fit(x[name][tr], y[tr])
            train_score = fitted.decision_function(x[name][tr])
            test_score = fitted.decision_function(x[name][te])
            pred[name][te] = (test_score >= 0).astype(np.int8)
            z[name] = (test_score - train_score.mean()) / max(train_score.std(), 1e-8)
            row["models"][name] = metric(y[te], pred[name][te])
        pred["G4_G9_equal"][te] = (0.5 * (z["G4"] + z["G9"]) >= 0).astype(np.int8)
        row["models"]["G4_G9_equal"] = metric(y[te], pred["G4_G9_equal"][te])
        fold_rows.append(row)
    return pred, fold_rows


def group_contributions(y, pred, groups, unique_groups):
    cc = np.zeros(len(unique_groups)); tc = np.zeros(len(unique_groups))
    cn = np.zeros(len(unique_groups)); tn = np.zeros(len(unique_groups))
    for i, group in enumerate(unique_groups):
        gm = groups == group; cm = gm & (y == 1); nm = gm & (y == 0)
        tc[i] = cm.sum(); cc[i] = np.sum(pred[cm] == 1)
        tn[i] = nm.sum(); cn[i] = np.sum(pred[nm] == 0)
    return cc, tc, cn, tn


def main():
    data_dir = ROOT / "dataset" / "ComParE2017_Cold_4students"
    labels = load_labels(str(data_dir))
    files = sorted(f for f in labels if f.startswith("train_"))
    stems = [Path(f).stem for f in files]
    y = np.asarray([labels[f] for f in files], dtype=np.int8)
    pseudo = load_pseudo_speakers(ROOT / "cache/pseudo_speakers/k210_seed42.tsv")
    groups = np.asarray([pseudo[s] for s in stems])
    x = load_features(stems)
    predictions = {name: [] for name in ("G4", "G9", "G4_G9_equal")}
    rows = []
    for seed in CV_SEEDS:
        pred, folds = fit_seed(y, x, groups, seed)
        for name in predictions: predictions[name].append(pred[name])
        rows.append({"split_seed": seed, "models": {name: metric(y, pred[name]) for name in pred}, "folds": folds})
        print(seed, {name: round(metric(y, pred[name])["uar"], 4) for name in pred})
    predictions = {name: np.stack(value) for name, value in predictions.items()}

    unique_groups = np.unique(groups)
    rng = np.random.default_rng(BOOT_SEED)
    weights = rng.multinomial(
        len(unique_groups), np.full(len(unique_groups), 1 / len(unique_groups)), size=N_BOOT
    )
    boot = {}
    for name, matrix in predictions.items():
        per_seed = []
        for pred in matrix:
            cc, tc, cn, tn = group_contributions(y, pred, groups, unique_groups)
            rec_c = (weights @ cc) / (weights @ tc)
            rec_nc = (weights @ cn) / (weights @ tn)
            per_seed.append(0.5 * (rec_c + rec_nc))
        boot[name] = np.mean(np.stack(per_seed), axis=0)

    summary = {}
    point = {}
    for name in predictions:
        values = [r["models"][name]["uar"] for r in rows]
        point[name] = float(np.mean(values))
        summary[name] = {
            "repeated_cv_mean_uar": point[name],
            "split_seed_std": float(np.std(values, ddof=1)),
            "split_seed_values": values,
            "group_bootstrap_95_ci_of_repeated_mean": np.quantile(boot[name], [0.025, 0.975]).tolist(),
        }
    comparisons = {}
    for baseline in ("G4", "G9"):
        delta = boot["G4_G9_equal"] - boot[baseline]
        point_delta = point["G4_G9_equal"] - point[baseline]
        seed_deltas = [
            rows[i]["models"]["G4_G9_equal"]["uar"] - rows[i]["models"][baseline]["uar"]
            for i in range(len(rows))
        ]
        comparisons[f"fusion_vs_{baseline}"] = {
            "point_delta": point_delta,
            "per_split_seed_deltas": seed_deltas,
            "paired_group_bootstrap_95_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
            "probability_positive": float(np.mean(delta > 0)),
        }
    mixed = [int(g) for g in unique_groups if len(np.unique(y[groups == g])) > 1]
    report = {
        "protocol": {
            "data": "official Train only", "development_used": False,
            "outer_cv": "six repetitions of 5-fold StratifiedGroupKFold",
            "split_seeds": CV_SEEDS, "threshold": 0,
            "fusion": "0.5 * (train-zscored G4 logit + train-zscored G9 logit)",
            "learned_fusion_parameters": 0,
            "bootstrap": f"{N_BOOT} paired resamples of inferred clusters, averaged across CV repetitions",
        },
        "data": {"chunks": len(y), "clusters": len(unique_groups), "mixed_label_clusters": mixed},
        "models": summary, "comparisons": comparisons, "runs": rows,
        "caveat": "Inferred clusters are imperfect speaker proxies; 15/210 mix labels.",
    }
    out = ROOT / "results/fixed_g4_g9_repeated_cv.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "results/fixed_g4_g9_repeated_cv_oof.npz",
        files=np.asarray(files), y=y, groups=groups, split_seeds=np.asarray(CV_SEEDS),
        **{f"pred__{name}": values for name, values in predictions.items()},
    )
    print(json.dumps({"models": summary, "comparisons": comparisons}, indent=2))
    print(f"[wrote] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
