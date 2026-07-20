"""Combined Train+Development outer-CV: monolithic refit vs 10-fold ensemble.

The outer test fold remains untouched.  A monolithic model is fit on the full
outer-training pool; the ensemble contains ten models, each fit on 9/10 of that
same outer-training pool.  Architectures are fixed G4 and fixed equal G4+G9.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from reconcile_cqt_protocol import CV_SEEDS, ROOT, load_features, load_labels, metric, model

sys.path.insert(0, str(ROOT / "model"))
from speakers.cluster import load_pseudo_speakers  # noqa: E402


N_OUTER = 5
N_ENSEMBLE = 10
N_BOOT = 20000


def branch_scores(x, y, train, test):
    raw = {}; z = {}
    for name in ("G4", "G9"):
        fitted = model().fit(x[name][train], y[train])
        train_score = fitted.decision_function(x[name][train])
        test_score = fitted.decision_function(x[name][test])
        raw[name] = test_score
        z[name] = (test_score - train_score.mean()) / max(train_score.std(), 1e-8)
    return raw, z


def fit_repetition(y, x, groups, split_seed):
    outer = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=split_seed)
    names = ("mono_G4", "ensemble_G4", "mono_G4_G9", "ensemble_G4_G9")
    pred = {name: np.full(len(y), -1, dtype=np.int8) for name in names}
    folds = []
    for fold, (otr, ote) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        raw, z = branch_scores(x, y, otr, ote)
        mono_g4 = raw["G4"]
        mono_fusion = 0.5 * (z["G4"] + z["G9"])
        pred["mono_G4"][ote] = (mono_g4 >= 0).astype(np.int8)
        pred["mono_G4_G9"][ote] = (mono_fusion >= 0).astype(np.int8)

        inner = StratifiedGroupKFold(
            n_splits=N_ENSEMBLE, shuffle=True, random_state=split_seed + 1000 + fold
        )
        member_g4 = []; member_fusion = []
        for itr_local, _ in inner.split(np.zeros(len(otr)), y[otr], groups[otr]):
            itr = otr[itr_local]
            member_raw, member_z = branch_scores(x, y, itr, ote)
            member_g4.append(member_raw["G4"])
            member_fusion.append(0.5 * (member_z["G4"] + member_z["G9"]))
        ensemble_g4 = np.mean(member_g4, axis=0)
        ensemble_fusion = np.mean(member_fusion, axis=0)
        pred["ensemble_G4"][ote] = (ensemble_g4 >= 0).astype(np.int8)
        pred["ensemble_G4_G9"][ote] = (ensemble_fusion >= 0).astype(np.int8)

        folds.append({
            "fold": fold, "n_outer_train": int(len(otr)), "n_outer_test": int(len(ote)),
            "n_outer_train_groups": int(len(np.unique(groups[otr]))),
            "n_outer_test_groups": int(len(np.unique(groups[ote]))),
            "models": {name: metric(y[ote], pred[name][ote]) for name in names},
        })
    return pred, folds


def group_parts(y, pred, groups, unique):
    cc = np.zeros(len(unique)); tc = np.zeros(len(unique))
    cn = np.zeros(len(unique)); tn = np.zeros(len(unique))
    for i, group in enumerate(unique):
        gm = groups == group; cm = gm & (y == 1); nm = gm & (y == 0)
        tc[i] = cm.sum(); cc[i] = np.sum(pred[cm] == 1)
        tn[i] = nm.sum(); cn[i] = np.sum(pred[nm] == 0)
    return cc, tc, cn, tn


def main():
    data_dir = ROOT / "dataset/ComParE2017_Cold_4students"
    labels = load_labels(str(data_dir))
    files = sorted(f for f in labels if f.startswith("train_") or f.startswith("devel_"))
    stems = [Path(f).stem for f in files]
    y = np.asarray([labels[f] for f in files], dtype=np.int8)
    x = load_features(stems)

    # Official Train and Development are speaker-disjoint. Offset cluster IDs
    # by official side so imperfect pooled clusters can never bridge the sides.
    pseudo = load_pseudo_speakers(ROOT / "cache/pseudo_speakers/pooled_k420_seed42.tsv")
    raw_group = np.asarray([pseudo[s] for s in stems], dtype=np.int64)
    side = np.asarray([0 if f.startswith("train_") else 1 for f in files], dtype=np.int8)
    groups = raw_group + side.astype(np.int64) * (int(raw_group.max()) + 1)
    unique = np.unique(groups)
    mixed = [int(g) for g in unique if len(np.unique(y[groups == g])) > 1]

    names = ("mono_G4", "ensemble_G4", "mono_G4_G9", "ensemble_G4_G9")
    predictions = {name: [] for name in names}; runs = []
    for seed in CV_SEEDS:
        pred, folds = fit_repetition(y, x, groups, seed)
        for name in names: predictions[name].append(pred[name])
        run = {"split_seed": seed, "models": {name: metric(y, pred[name]) for name in names}, "folds": folds}
        runs.append(run)
        print(seed, {name: round(run["models"][name]["uar"], 4) for name in names})
    predictions = {name: np.stack(value) for name, value in predictions.items()}

    rng = np.random.default_rng(20260720)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=N_BOOT)
    boot = {}
    for name, matrix in predictions.items():
        per_seed = []
        for p in matrix:
            cc, tc, cn, tn = group_parts(y, p, groups, unique)
            per_seed.append(0.5 * ((weights @ cc) / (weights @ tc) + (weights @ cn) / (weights @ tn)))
        boot[name] = np.mean(per_seed, axis=0)

    summary = {}
    for name in names:
        values = [r["models"][name]["uar"] for r in runs]
        summary[name] = {
            "repeated_cv_mean_uar": float(np.mean(values)),
            "split_seed_std": float(np.std(values, ddof=1)),
            "values": values,
            "paired_cluster_bootstrap_95_ci": np.quantile(boot[name], [0.025, 0.975]).tolist(),
        }
    pairs = {
        "ensemble_vs_mono_G4": ("ensemble_G4", "mono_G4"),
        "ensemble_vs_mono_G4_G9": ("ensemble_G4_G9", "mono_G4_G9"),
        "mono_fusion_vs_mono_G4": ("mono_G4_G9", "mono_G4"),
        "ensemble_fusion_vs_ensemble_G4": ("ensemble_G4_G9", "ensemble_G4"),
    }
    comparisons = {}
    for label, (a, b) in pairs.items():
        delta = boot[a] - boot[b]
        point = summary[a]["repeated_cv_mean_uar"] - summary[b]["repeated_cv_mean_uar"]
        comparisons[label] = {
            "point_delta": point,
            "paired_cluster_bootstrap_95_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
            "probability_positive": float(np.mean(delta > 0)),
            "per_split_seed_deltas": [runs[i]["models"][a]["uar"] - runs[i]["models"][b]["uar"] for i in range(len(runs))],
        }
    report = {
        "protocol": {
            "data": "official Train+Development", "test_used": False,
            "groups": "pooled_k420 proxy clusters, offset by official side",
            "outer": f"{len(CV_SEEDS)} repetitions x {N_OUTER} folds",
            "ensemble": f"{N_ENSEMBLE} inner grouped-fold models per outer test fold",
            "threshold": 0, "fusion": "fixed equal train-zscored G4+G9",
        },
        "data": {"chunks": len(y), "groups": len(unique), "mixed_label_groups": mixed,
                 "cold_chunks": int(y.sum())},
        "models": summary, "comparisons": comparisons, "runs": runs,
        "interpretation_limit": "This estimates models trained on 80% of Train+Development; full-data deployment UAR remains unobservable without Test labels.",
    }
    out = ROOT / "results/combined_refit_vs_ensemble_cv.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "results/combined_refit_vs_ensemble_oof.npz",
        files=np.asarray(files), y=y, groups=groups, split_seeds=np.asarray(CV_SEEDS),
        **{f"pred__{name}": value for name, value in predictions.items()},
    )
    print(json.dumps({"models": summary, "comparisons": comparisons}, indent=2))
    print(f"[wrote] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
