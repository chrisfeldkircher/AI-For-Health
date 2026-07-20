"""Corrected nested speaker-group CV for the reusable cached feature shortlist.

Selection is entirely inside official Train. Five outer pseudo-speaker folds
estimate performance; four inner pseudo-speaker folds produce out-of-fold base
logits for threshold selection and leakage-safe stacked fusion. Official
Development is never loaded by this script.

Outputs: results/corrected_outer_cv_linear.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import load_labels  # noqa: E402
from speakers.cluster import load_pseudo_speakers  # noqa: E402


SEED = 20260720
OUTER_FOLDS = 5
INNER_FOLDS = 4


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "uar": float(balanced_accuracy_score(y, pred)),
        "recall_C": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_NC": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "accuracy": float(np.mean(y == pred)),
    }


def best_tau(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, score, pos_label=1)
    uars = 0.5 * (tpr + 1.0 - fpr)
    i = int(np.nanargmax(uars))
    tau = float(thresholds[i])
    if not np.isfinite(tau):
        tau = float(np.nextafter(np.max(score), np.inf))
    return tau, float(uars[i])


def estimator(c: float = 1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=SEED,
        ),
    )


def score_model(model, x: np.ndarray) -> np.ndarray:
    return model.decision_function(x).astype(np.float64)


def load_npy(stems: list[str], subdir: str, sl: slice | None = None) -> np.ndarray:
    base = ROOT / "cache" / "handcrafted" / subdir
    x = np.stack([np.load(base / f"{stem}.npy") for stem in stems]).astype(np.float32)
    return x[:, sl] if sl is not None else x


def load_torch(stems: list[str], backbone: str) -> np.ndarray:
    base = ROOT / "cache" / backbone / "pooled"
    return np.stack([
        torch.load(base / f"{stem}.pt", weights_only=True, map_location="cpu").numpy()
        for stem in stems
    ]).astype(np.float32)


def same_identity_centroid_probe(
    x: np.ndarray, identities: np.ndarray, seed: int = SEED
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    fit, ev = [], []
    for identity in np.unique(identities):
        idx = np.flatnonzero(identities == identity)
        rng.shuffle(idx)
        n_ev = min(max(1, int(round(0.2 * len(idx)))), len(idx) - 1)
        fit.extend(idx[n_ev:]); ev.extend(idx[:n_ev])
    fit = np.asarray(fit); ev = np.asarray(ev)
    scaler = StandardScaler().fit(x[fit])
    xf = normalize(scaler.transform(x[fit]), axis=1)
    xe = normalize(scaler.transform(x[ev]), axis=1)
    ids = np.unique(identities[fit])
    cent = normalize(np.vstack([xf[identities[fit] == i].mean(0) for i in ids]), axis=1)
    pred = ids[(xe @ cent.T).argmax(1)]
    return {
        "top1": float(np.mean(pred == identities[ev])),
        "n_fit": int(len(fit)), "n_eval": int(len(ev)),
        "uniform_chance": float(1 / len(ids)),
    }


def main() -> None:
    t0 = time.time()
    data_dir = ROOT / "dataset" / "ComParE2017_Cold_4students"
    labels = load_labels(str(data_dir))
    pseudo = load_pseudo_speakers(ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv")
    files = sorted(f for f in labels if f.startswith("train_"))
    stems = [Path(f).stem for f in files]
    y = np.asarray([labels[f] for f in files], dtype=np.int64)
    groups = np.asarray([pseudo[s] for s in stems], dtype=np.int64)

    print(f"[data] official Train only: chunks={len(files)} groups={len(np.unique(groups))} "
          f"cold={int(y.sum())}")
    print("[load] cached candidate features")
    features = {
        "G4_gain_invariant": load_npy(stems, "g4", slice(4, None)),
        "G5_modulation": load_npy(stems, "modulation"),
        "G9_CQT": load_npy(stems, "cqt"),
        "eGeMAPS88": load_npy(stems, "egemaps"),
        "signature10": load_npy(stems, "signature"),
        "HeAR512": load_torch(stems, "google_hear-pytorch"),
    }
    for name, x in features.items():
        if not np.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values")
        print(f"  {name:<20} {x.shape}")

    outer = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED
    )
    outer_splits = list(outer.split(np.zeros(len(y)), y, groups))
    fold_rows = []
    all_outer_scores = {name: np.full(len(y), np.nan) for name in features}
    all_outer_pred = {name: np.full(len(y), -1, dtype=np.int8) for name in features}
    fusion_specs = {
        "fusion_G4_G9": ["G4_gain_invariant", "G9_CQT"],
        "fusion_G4_G5_G9": ["G4_gain_invariant", "G5_modulation", "G9_CQT"],
        "fusion_G4_G9_HeAR": ["G4_gain_invariant", "G9_CQT", "HeAR512"],
        "fusion_all_compact": [
            "G4_gain_invariant", "G5_modulation", "G9_CQT",
            "eGeMAPS88", "signature10", "HeAR512",
        ],
    }
    for name in fusion_specs:
        all_outer_scores[name] = np.full(len(y), np.nan)
        all_outer_pred[name] = np.full(len(y), -1, dtype=np.int8)

    for outer_id, (outer_train, outer_test) in enumerate(outer_splits):
        print(f"\n=== OUTER {outer_id + 1}/{OUTER_FOLDS} "
              f"train={len(outer_train)} test={len(outer_test)} "
              f"test_groups={len(np.unique(groups[outer_test]))} ===")
        inner = StratifiedGroupKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 100 + outer_id
        )
        inner_local = list(inner.split(
            np.zeros(len(outer_train)), y[outer_train], groups[outer_train]
        ))
        inner_fold_id = np.full(len(outer_train), -1, dtype=np.int8)
        inner_oof = {
            name: np.full(len(outer_train), np.nan, dtype=np.float64)
            for name in features
        }

        # Inner OOF base logits: used only for threshold and meta-model fitting.
        for inner_id, (itr_local, iva_local) in enumerate(inner_local):
            itr, iva = outer_train[itr_local], outer_train[iva_local]
            inner_fold_id[iva_local] = inner_id
            for name, x in features.items():
                m = estimator().fit(x[itr], y[itr])
                inner_oof[name][iva_local] = score_model(m, x[iva])
        assert np.all(inner_fold_id >= 0)
        assert all(np.isfinite(v).all() for v in inner_oof.values())

        row = {
            "outer_fold": outer_id,
            "n_train": int(len(outer_train)), "n_test": int(len(outer_test)),
            "n_train_groups": int(len(np.unique(groups[outer_train]))),
            "n_test_groups": int(len(np.unique(groups[outer_test]))),
            "models": {},
        }

        outer_base_scores = {}
        for name, x in features.items():
            tau, inner_uar = best_tau(y[outer_train], inner_oof[name])
            m = estimator().fit(x[outer_train], y[outer_train])
            score = score_model(m, x[outer_test])
            pred = (score >= tau).astype(np.int8)
            outer_base_scores[name] = score
            all_outer_scores[name][outer_test] = score
            all_outer_pred[name][outer_test] = pred
            row["models"][name] = {
                "tau": tau, "inner_oof_uar_at_tau": inner_uar,
                **metric(y[outer_test], pred),
            }
            print(f"  {name:<20} outer_UAR={row['models'][name]['uar']:.4f} "
                  f"inner={inner_uar:.4f}")

        # Nested stacking: base scores are OOF; meta scores are cross-fitted
        # across the inner folds before selecting a threshold.
        for fusion_name, members in fusion_specs.items():
            z_inner = np.column_stack([inner_oof[m] for m in members])
            meta_oof = np.full(len(outer_train), np.nan)
            for inner_id in range(INNER_FOLDS):
                va = inner_fold_id == inner_id
                tr = ~va
                meta = estimator(c=0.1).fit(z_inner[tr], y[outer_train][tr])
                meta_oof[va] = score_model(meta, z_inner[va])
            tau, inner_uar = best_tau(y[outer_train], meta_oof)
            meta = estimator(c=0.1).fit(z_inner, y[outer_train])
            z_test = np.column_stack([outer_base_scores[m] for m in members])
            score = score_model(meta, z_test)
            pred = (score >= tau).astype(np.int8)
            all_outer_scores[fusion_name][outer_test] = score
            all_outer_pred[fusion_name][outer_test] = pred
            row["models"][fusion_name] = {
                "members": members, "tau": tau,
                "inner_meta_oof_uar_at_tau": inner_uar,
                "meta_coefficients": meta.named_steps["logisticregression"].coef_[0].tolist(),
                **metric(y[outer_test], pred),
            }
            print(f"  {fusion_name:<20} outer_UAR={row['models'][fusion_name]['uar']:.4f} "
                  f"inner={inner_uar:.4f}")
        fold_rows.append(row)

    summary = {}
    for name in all_outer_pred:
        assert np.all(all_outer_pred[name] >= 0)
        fold_uars = [r["models"][name]["uar"] for r in fold_rows]
        summary[name] = {
            "outer_oof": metric(y, all_outer_pred[name]),
            "fold_uar_mean": float(np.mean(fold_uars)),
            "fold_uar_std": float(np.std(fold_uars, ddof=1)),
            "fold_uars": fold_uars,
        }

    # Same-identity leakage measure for standalone inputs. This is separate
    # from cold-classifier outer CV and has the correct closed-set semantics.
    speaker_probe = {
        name: same_identity_centroid_probe(x, groups)
        for name, x in features.items()
    }
    ranked = sorted(summary, key=lambda n: summary[n]["outer_oof"]["uar"], reverse=True)
    report = {
        "rung_id": "corrected_outer_cv_linear",
        "protocol": {
            "selection_pool": "official Train only",
            "official_development_used": False,
            "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
            "groups": "k210_seed42 train pseudo-speakers (train NN cohesion 0.974)",
            "base_model": "StandardScaler + balanced liblinear logistic regression",
            "threshold": "selected on inner out-of-fold scores",
            "fusion": "cross-fitted inner-OOF regularized logistic stacking",
        },
        "folds": fold_rows,
        "summary": summary,
        "same_identity_speaker_probe": speaker_probe,
        "ranking": ranked,
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    out = ROOT / "results" / "corrected_outer_cv_linear.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    npz_out = ROOT / "results" / "corrected_outer_cv_linear_oof.npz"
    np.savez_compressed(
        npz_out,
        files=np.asarray(files), y=y, groups=groups,
        **{f"score__{name}": values for name, values in all_outer_scores.items()},
        **{f"pred__{name}": values for name, values in all_outer_pred.items()},
    )
    print("\n=== CORRECTED OUTER-OOF RANKING ===")
    for name in ranked:
        s = summary[name]
        print(f"  {name:<22} UAR={s['outer_oof']['uar']:.4f}  "
              f"fold={s['fold_uar_mean']:.4f}+/-{s['fold_uar_std']:.4f}")
    print(f"[wrote] {out.relative_to(ROOT)} + {npz_out.relative_to(ROOT)}  "
          f"elapsed={report['elapsed_minutes']:.1f} min")


if __name__ == "__main__":
    main()
