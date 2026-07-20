"""Test a low-capacity G4-anchored residual architecture with nested group CV.

The score is z(G4) + alpha*z(candidate), with alpha constrained to [0, 1].
Alpha=0 is the G4-only fallback.  Alpha and the decision threshold are chosen
only from inner speaker-group OOF predictions; official Development is unused.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from rerun_corrected_outer_cv import (
    ROOT, SEED, OUTER_FOLDS, INNER_FOLDS, best_tau, estimator, load_labels,
    load_npy, load_pseudo_speakers, load_torch, metric, score_model,
)


ALPHAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


def fit_z(model, x: np.ndarray) -> tuple[float, float]:
    s = score_model(model, x)
    return float(s.mean()), float(max(s.std(), 1e-8))


def apply_z(model, x: np.ndarray, pars: tuple[float, float]) -> np.ndarray:
    return (score_model(model, x) - pars[0]) / pars[1]


def main() -> None:
    started = time.time()
    data_dir = ROOT / "dataset" / "ComParE2017_Cold_4students"
    labels = load_labels(str(data_dir))
    pseudo = load_pseudo_speakers(ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv")
    files = sorted(f for f in labels if f.startswith("train_"))
    stems = [Path(f).stem for f in files]
    y = np.asarray([labels[f] for f in files], dtype=np.int64)
    groups = np.asarray([pseudo[s] for s in stems], dtype=np.int64)
    features = {
        "G4_gain_invariant": load_npy(stems, "g4", slice(4, None)),
        "G5_modulation": load_npy(stems, "modulation"),
        "G9_CQT": load_npy(stems, "cqt"),
        "eGeMAPS88": load_npy(stems, "egemaps"),
        "signature10": load_npy(stems, "signature"),
        "HeAR512": load_torch(stems, "google_hear-pytorch"),
    }
    anchor = "G4_gain_invariant"
    candidates = [n for n in features if n != anchor]
    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    splits = list(outer.split(np.zeros(len(y)), y, groups))
    pred = {n: np.full(len(y), -1, dtype=np.int8) for n in candidates}
    score = {n: np.full(len(y), np.nan) for n in candidates}
    folds = []

    for fold, (otr, ote) in enumerate(splits):
        inner = StratifiedGroupKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 100 + fold
        )
        inner_splits = list(inner.split(np.zeros(len(otr)), y[otr], groups[otr]))
        inner_z = {n: np.full(len(otr), np.nan) for n in features}
        for itr_local, iva_local in inner_splits:
            itr, iva = otr[itr_local], otr[iva_local]
            for name, x in features.items():
                model = estimator().fit(x[itr], y[itr])
                inner_z[name][iva_local] = apply_z(model, x[iva], fit_z(model, x[itr]))

        outer_models = {n: estimator().fit(x[otr], y[otr]) for n, x in features.items()}
        outer_z = {
            n: apply_z(outer_models[n], x[ote], fit_z(outer_models[n], x[otr]))
            for n, x in features.items()
        }
        row = {"outer_fold": fold, "models": {}}
        for name in candidates:
            best = None
            for alpha in ALPHAS:
                s = inner_z[anchor] + alpha * inner_z[name]
                tau, uar = best_tau(y[otr], s)
                candidate = (uar, -alpha, alpha, tau)
                if best is None or candidate > best:
                    best = candidate
            inner_uar, _, alpha, tau = best
            out_score = outer_z[anchor] + alpha * outer_z[name]
            out_pred = (out_score >= tau).astype(np.int8)
            score[name][ote] = out_score
            pred[name][ote] = out_pred
            row["models"][name] = {
                "alpha": alpha, "tau": tau, "inner_oof_uar": inner_uar,
                **metric(y[ote], out_pred),
            }
            print(f"fold={fold+1} residual={name:<14} alpha={alpha:<4} "
                  f"outer_UAR={row['models'][name]['uar']:.4f}")
        folds.append(row)

    summary = {}
    for name in candidates:
        fold_uars = [r["models"][name]["uar"] for r in folds]
        alphas = [r["models"][name]["alpha"] for r in folds]
        summary[name] = {
            "outer_oof": metric(y, pred[name]),
            "fold_uar_mean": float(np.mean(fold_uars)),
            "fold_uar_std": float(np.std(fold_uars, ddof=1)),
            "fold_uars": fold_uars, "selected_alphas": alphas,
            "zero_alpha_folds": int(sum(a == 0 for a in alphas)),
        }
    ranking = sorted(summary, key=lambda n: summary[n]["outer_oof"]["uar"], reverse=True)
    report = {
        "rung_id": "g4_anchored_residual",
        "protocol": {
            "selection_pool": "official Train only", "development_used": False,
            "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
            "architecture": "z(G4) + alpha*z(candidate)", "alpha_grid": ALPHAS,
            "selection": "alpha and threshold selected on inner speaker-group OOF",
        },
        "folds": folds, "summary": summary, "ranking": ranking,
        "elapsed_minutes": (time.time() - started) / 60,
    }
    out = ROOT / "results" / "g4_anchored_residual.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "results" / "g4_anchored_residual_oof.npz",
        files=np.asarray(files), y=y, groups=groups,
        **{f"score__{n}": v for n, v in score.items()},
        **{f"pred__{n}": v for n, v in pred.items()},
    )
    print("\n=== G4-ANCHORED RESIDUAL RANKING ===")
    for name in ranking:
        s = summary[name]
        print(f"{name:<14} UAR={s['outer_oof']['uar']:.4f} "
              f"fold={s['fold_uar_mean']:.4f}+/-{s['fold_uar_std']:.4f} "
              f"alphas={s['selected_alphas']}")
    print(f"[wrote] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
