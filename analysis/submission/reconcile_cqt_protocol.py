"""Reconcile the old CQT headline with corrected Train-only outer CV.

Diagnostics only: reproduces the historical Train->Development protocol,
measures grouping/solver/training-fraction effects on official Train, and
nested-selects CQT regularisation without touching Development.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import load_labels, stratified_grouped_split  # noqa: E402
from honesty import fit_cold_probe, fit_zscore, predict_logit, sweep_tau  # noqa: E402
from speakers.cluster import load_pseudo_speakers  # noqa: E402


SEEDS = [42, 1, 2, 3, 5]
CV_SEEDS = [42, 1, 2, 3, 5, 20260720]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "uar": float(balanced_accuracy_score(y, pred)),
        "recall_C": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_NC": float(recall_score(y, pred, pos_label=0, zero_division=0)),
    }


def model(*, solver: str = "liblinear", c: float = 1.0, seed: int = 42):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, class_weight="balanced", solver=solver,
            max_iter=3000, random_state=seed,
        ),
    )


def load_features(stems: list[str]) -> dict[str, np.ndarray]:
    return {
        "G4": np.stack([
            np.load(ROOT / "cache" / "handcrafted" / "g4" / f"{s}.npy")[4:]
            for s in stems
        ]).astype(np.float32),
        "G9": np.stack([
            np.load(ROOT / "cache" / "handcrafted" / "cqt" / f"{s}.npy")
            for s in stems
        ]).astype(np.float32),
    }


def cv_fixed_zero(
    y: np.ndarray, x: dict[str, np.ndarray], groups: np.ndarray,
    *, n_splits: int, split_seed: int, solver: str = "liblinear",
) -> dict:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    pred = {name: np.full(len(y), -1, dtype=np.int8) for name in (*x.keys(), "G4_G9_equal")}
    rows = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        row = {
            "fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_test_clusters": int(len(np.unique(groups[te]))),
            "n_test_clusters_with_cold": int(sum(np.any(y[te][groups[te] == g] == 1) for g in np.unique(groups[te]))),
            "n_test_mixed_clusters": int(sum(len(np.unique(y[te][groups[te] == g])) > 1 for g in np.unique(groups[te]))),
            "n_test_cold_chunks": int(y[te].sum()),
            "cold_chunk_fraction": float(y[te].mean()),
            "models": {},
        }
        z_test = {}
        for name, values in x.items():
            fitted = model(solver=solver, seed=split_seed).fit(values[tr], y[tr])
            train_out = fitted.decision_function(values[tr])
            out = fitted.decision_function(values[te])
            z_test[name] = (out - train_out.mean()) / max(train_out.std(), 1e-8)
            p = (out >= 0).astype(np.int8)
            pred[name][te] = p
            row["models"][name] = metric(y[te], p)
        fused = 0.5 * (z_test["G4"] + z_test["G9"])
        fused_pred = (fused >= 0).astype(np.int8)
        pred["G4_G9_equal"][te] = fused_pred
        row["models"]["G4_G9_equal"] = metric(y[te], fused_pred)
        rows.append(row)
    return {
        "models": {
            name: {
                "oof": metric(y, p),
                "fold_uars": [r["models"][name]["uar"] for r in rows],
                "fold_uar_std": float(np.std([r["models"][name]["uar"] for r in rows], ddof=1)),
            }
            for name, p in pred.items()
        },
        "folds": rows,
    }


def nested_cqt_c(y: np.ndarray, g9: np.ndarray, groups: np.ndarray) -> dict:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260720)
    pred = np.full(len(y), -1, dtype=np.int8)
    rows = []
    for fold, (otr, ote) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260820 + fold)
        scores = {}
        for c in C_GRID:
            ip = np.full(len(otr), -1, dtype=np.int8)
            for itr_local, iva_local in inner.split(np.zeros(len(otr)), y[otr], groups[otr]):
                itr, iva = otr[itr_local], otr[iva_local]
                fitted = model(c=c).fit(g9[itr], y[itr])
                ip[iva_local] = (fitted.decision_function(g9[iva]) >= 0).astype(np.int8)
            scores[c] = balanced_accuracy_score(y[otr], ip)
        best_c = max(C_GRID, key=lambda c: (scores[c], -c))
        fitted = model(c=best_c).fit(g9[otr], y[otr])
        pred[ote] = (fitted.decision_function(g9[ote]) >= 0).astype(np.int8)
        rows.append({"fold": fold, "selected_C": best_c, "inner_uars": scores,
                     "outer": metric(y[ote], pred[ote])})
    return {"oof": metric(y, pred), "folds": rows}


def old_protocol(
    train_files: list[str], dev_files: list[str], labels: dict[str, int],
    pseudo: dict[str, int], xtr: dict[str, np.ndarray], xdev: dict[str, np.ndarray],
) -> dict:
    tr_index = {f: i for i, f in enumerate(train_files)}
    dv_index = {f: i for i, f in enumerate(dev_files)}
    rows = []
    for seed in SEEDS:
        fit_files, tau_files = stratified_grouped_split(
            train_files, labels, pseudo, val_frac=0.10, seed=seed
        )
        _, test_files = stratified_grouped_split(
            dev_files, labels, pseudo, val_frac=0.50, seed=seed
        )
        fit = np.asarray([tr_index[f] for f in fit_files]); tau_idx = np.asarray([tr_index[f] for f in tau_files])
        test = np.asarray([dv_index[f] for f in test_files])
        yf = np.asarray([labels[f] for f in fit_files]); yt = np.asarray([labels[f] for f in tau_files])
        ye = np.asarray([labels[f] for f in test_files])
        logits_tau = {}; logits_test = {}; z_tau = {}; z_test = {}; result = {}
        for name in ("G4", "G9"):
            clf, scaler = fit_cold_probe(xtr[name][fit], yf, seed=seed)
            lf = predict_logit(clf, scaler, xtr[name][fit])
            logits_tau[name] = predict_logit(clf, scaler, xtr[name][tau_idx])
            logits_test[name] = predict_logit(clf, scaler, xdev[name][test])
            zp = fit_zscore(lf)
            z_tau[name] = zp.apply(logits_tau[name]); z_test[name] = zp.apply(logits_test[name])
            threshold, _ = sweep_tau(logits_tau[name], yt)
            result[name] = {"tau": threshold, **metric(ye, (logits_test[name] >= threshold).astype(np.int8))}
        fused_tau = 0.5 * (z_tau["G4"] + z_tau["G9"])
        fused_test = 0.5 * (z_test["G4"] + z_test["G9"])
        threshold, _ = sweep_tau(fused_tau, yt)
        result["G4_G9_equal"] = {"tau": threshold, **metric(ye, (fused_test >= threshold).astype(np.int8))}
        rows.append({
            "split_seed": seed, "n_train_fit": int(len(fit)), "n_train_threshold": int(len(tau_idx)),
            "n_devel_test": int(len(test)), "n_threshold_cold_clusters": int(len(np.unique(np.asarray([pseudo[f[:-4]] for f in tau_files])[yt == 1]))),
            "n_devel_test_cold_clusters": int(len(np.unique(np.asarray([pseudo[f[:-4]] for f in test_files])[ye == 1]))),
            "models": result,
        })
    summary = {}
    for name in ("G4", "G9", "G4_G9_equal"):
        vals = [r["models"][name]["uar"] for r in rows]
        summary[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
    return {"summary": summary, "splits": rows}


def full_train_to_devel(ytr, ydv, xtr, xdev) -> dict:
    logits = {}; zdev = {}
    result = {}
    for name in ("G4", "G9"):
        fitted = model(solver="lbfgs").fit(xtr[name], ytr)
        train_score = fitted.decision_function(xtr[name])
        dev_score = fitted.decision_function(xdev[name])
        logits[name] = dev_score
        zdev[name] = (dev_score - train_score.mean()) / max(train_score.std(), 1e-8)
        result[name] = metric(ydv, (dev_score >= 0).astype(np.int8))
    result["G4_G9_equal"] = metric(ydv, (0.5 * (zdev["G4"] + zdev["G9"]) >= 0).astype(np.int8))
    return result


def main() -> None:
    started = time.time()
    data_dir = ROOT / "dataset" / "ComParE2017_Cold_4students"
    labels = load_labels(str(data_dir))
    train_files = sorted(f for f in labels if f.startswith("train_"))
    dev_files = sorted(f for f in labels if f.startswith("devel_"))
    train_stems = [Path(f).stem for f in train_files]; dev_stems = [Path(f).stem for f in dev_files]
    ytr = np.asarray([labels[f] for f in train_files]); ydv = np.asarray([labels[f] for f in dev_files])
    print("[load] G4/G9 Train+Development")
    xtr = load_features(train_stems); xdev = load_features(dev_stems)

    grouping_paths = {
        "k210_train_only_original": ROOT / "cache/pseudo_speakers/k210_seed42.tsv",
        "pooled_k420_original": ROOT / "cache/pseudo_speakers/pooled_k420_seed42.tsv",
        "ablation_train_only_k210": ROOT / "cache/pseudo_speakers/ablation_train_only_k210_seed42.tsv",
        "ablation_pooled_k420": ROOT / "cache/pseudo_speakers/ablation_pooled_k420_seed42.tsv",
        "pooled_k420_no_develtest": ROOT / "cache/pseudo_speakers/ablation_pooled_k420_no_develtest_seed42.tsv",
    }
    groupings = {name: load_pseudo_speakers(path) for name, path in grouping_paths.items()}
    group_arrays = {name: np.asarray([mapping[s] for s in train_stems]) for name, mapping in groupings.items()}

    grouping_sensitivity = {}
    for name, groups in group_arrays.items():
        seed_rows = []
        for seed in CV_SEEDS:
            print(f"[outer] {name} seed={seed}")
            seed_rows.append(cv_fixed_zero(ytr, xtr, groups, n_splits=5, split_seed=seed))
        grouping_sensitivity[name] = {
            model_name: {
                "mean_oof_uar": float(np.mean([r["models"][model_name]["oof"]["uar"] for r in seed_rows])),
                "std_across_split_seeds": float(np.std([r["models"][model_name]["oof"]["uar"] for r in seed_rows], ddof=1)),
                "values": [r["models"][model_name]["oof"]["uar"] for r in seed_rows],
            }
            for model_name in ("G4", "G9", "G4_G9_equal")
        }
        grouping_sensitivity[name]["seed42_folds"] = seed_rows[0]["folds"]
        grouping_sensitivity[name]["corrected_seed_folds"] = seed_rows[-1]["folds"]

    primary_groups = group_arrays["k210_train_only_original"]
    training_fraction = {}
    for n_splits in (5, 10):
        rows = [cv_fixed_zero(ytr, xtr, primary_groups, n_splits=n_splits, split_seed=s) for s in CV_SEEDS]
        training_fraction[str(n_splits)] = {
            "train_fraction": 1 - 1 / n_splits,
            **{name: {
                "mean_oof_uar": float(np.mean([r["models"][name]["oof"]["uar"] for r in rows])),
                "std": float(np.std([r["models"][name]["oof"]["uar"] for r in rows], ddof=1)),
                "values": [r["models"][name]["oof"]["uar"] for r in rows],
            } for name in ("G4", "G9", "G4_G9_equal")},
        }

    solver_comparison = {
        solver: cv_fixed_zero(ytr, xtr, primary_groups, n_splits=5, split_seed=20260720, solver=solver)["models"]
        for solver in ("liblinear", "lbfgs")
    }
    print("[nested] CQT regularisation")
    cqt_regularisation = nested_cqt_c(ytr, xtr["G9"], primary_groups)
    print("[historical] reproduce old pooled Train->half-Development protocol")
    historical = old_protocol(
        train_files, dev_files, labels, groupings["pooled_k420_original"], xtr, xdev
    )
    train_to_full_devel = full_train_to_devel(ytr, ydv, xtr, xdev)
    devel_to_full_train = full_train_to_devel(ydv, ytr, xdev, xtr)
    dev_groups = np.asarray([groupings["pooled_k420_original"][s] for s in dev_stems])
    dev_cv_rows = [cv_fixed_zero(ydv, xdev, dev_groups, n_splits=5, split_seed=s) for s in CV_SEEDS]
    within_development = {
        name: {
            "mean_oof_uar": float(np.mean([r["models"][name]["oof"]["uar"] for r in dev_cv_rows])),
            "std": float(np.std([r["models"][name]["oof"]["uar"] for r in dev_cv_rows], ddof=1)),
            "values": [r["models"][name]["oof"]["uar"] for r in dev_cv_rows],
        }
        for name in ("G4", "G9", "G4_G9_equal")
    }

    corrected = np.load(ROOT / "results/corrected_outer_cv_linear_oof.npz")
    threshold_effect = {}
    for name in ("G4_gain_invariant", "G9_CQT"):
        score = corrected[f"score__{name}"]; tuned = corrected[f"pred__{name}"]
        threshold_effect[name] = {
            "fixed_zero": metric(ytr, (score >= 0).astype(np.int8)),
            "inner_tuned": metric(ytr, tuned),
        }

    report = {
        "question": "Why did old CQT/G4 estimates exceed corrected outer CV?",
        "facts_about_prior_artifacts": {
            "A5j_features": "G4+G5, not G4+G9/CQT",
            "A5g_cqt_shadow_values": [0.672438105221124, 0.6145858601014458, 0.6504401593199153, 0.5776660502672496, 0.5937825103265043],
            "A5g_cqt_shadow_mean": 0.6217825370472478,
            "A5g_cqt_shadow_std": 0.03927245842373333,
            "A5g_seed42_handcrafted_G4_G9": 0.6743494392865462,
            "A5g_evaluation": "half of official Development; candidate family had already been inspected on Development",
        },
        "grouping_sensitivity_train_only_outer_cv": grouping_sensitivity,
        "training_fraction_diagnostic": training_fraction,
        "solver_comparison": solver_comparison,
        "nested_cqt_regularisation": cqt_regularisation,
        "historical_protocol_reproduction": historical,
        "full_train_to_full_development_fixed_zero": train_to_full_devel,
        "full_development_to_full_train_fixed_zero": devel_to_full_train,
        "within_development_outer_cv": within_development,
        "threshold_effect_corrected_outer_oof": threshold_effect,
        "limitations": [
            "Pseudo-speaker clusters are inferred and 15/210 primary clusters mix labels.",
            "Historical Development results are diagnostic only because Development was repeatedly inspected.",
            "Learning-curve comparisons change fold granularity as well as training fraction.",
        ],
        "elapsed_minutes": (time.time() - started) / 60,
    }
    out = ROOT / "results" / "cqt_protocol_reconciliation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[wrote] {out.relative_to(ROOT)} elapsed={report['elapsed_minutes']:.1f}m")


if __name__ == "__main__":
    main()
