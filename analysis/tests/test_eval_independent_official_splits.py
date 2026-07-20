"""Eval-independent fixed G4+CQT test on official speaker-disjoint sides.

Models are trained on one official side and evaluated on the other. No pseudo
grouping is used for training, splitting, fusion or thresholding. Proxy groups
are used only for paired uncertainty resampling after predictions are frozen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from reconcile_cqt_protocol import ROOT, load_features, load_labels, metric, model

sys.path.insert(0, str(ROOT / "model"))
from speakers.cluster import load_pseudo_speakers  # noqa: E402


N_BOOT = 30000
SEED = 20260720


def fit_direction(x_fit, y_fit, x_eval, y_eval):
    score = {}; z = {}; pred = {}
    for name in ("G4", "G9"):
        fitted = model().fit(x_fit[name], y_fit)
        fit_score = fitted.decision_function(x_fit[name])
        eval_score = fitted.decision_function(x_eval[name])
        score[name] = eval_score
        z[name] = (eval_score - fit_score.mean()) / max(fit_score.std(), 1e-8)
        pred[name] = (eval_score >= 0).astype(np.int8)
    score["G4_G9_equal"] = 0.5 * (z["G4"] + z["G9"])
    pred["G4_G9_equal"] = (score["G4_G9_equal"] >= 0).astype(np.int8)
    return score, pred, {name: metric(y_eval, pred[name]) for name in pred}


def parts(y, pred, groups, unique):
    cc = np.zeros(len(unique)); tc = np.zeros(len(unique))
    cn = np.zeros(len(unique)); tn = np.zeros(len(unique))
    for i, group in enumerate(unique):
        gm = groups == group; cm = gm & (y == 1); nm = gm & (y == 0)
        tc[i] = cm.sum(); cc[i] = np.sum(pred[cm] == 1)
        tn[i] = nm.sum(); cn[i] = np.sum(pred[nm] == 0)
    return cc, tc, cn, tn


def cluster_boot(y, predictions, groups, rng):
    unique = np.unique(groups)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=N_BOOT)
    out = {}
    for name, pred in predictions.items():
        cc, tc, cn, tn = parts(y, pred, groups, unique)
        out[name] = 0.5 * ((weights @ cc) / (weights @ tc) + (weights @ cn) / (weights @ tn))
    return out


def chunk_boot(y, predictions, rng):
    cold = np.flatnonzero(y == 1); noncold = np.flatnonzero(y == 0)
    ci = rng.choice(cold, size=(N_BOOT, len(cold)), replace=True)
    ni = rng.choice(noncold, size=(N_BOOT, len(noncold)), replace=True)
    out = {}
    for name, pred in predictions.items():
        out[name] = 0.5 * ((pred[ci] == 1).mean(1) + (pred[ni] == 0).mean(1))
    return out


def compare(draws, a="G4_G9_equal", b="G4"):
    delta = draws[a] - draws[b]
    return {"paired_95_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
            "probability_positive": float(np.mean(delta > 0))}


def main():
    data_dir = ROOT / "dataset/ComParE2017_Cold_4students"
    labels = load_labels(str(data_dir))
    files = {
        side: sorted(f for f in labels if f.startswith(side + "_"))
        for side in ("train", "devel")
    }
    stems = {side: [Path(f).stem for f in fs] for side, fs in files.items()}
    y = {side: np.asarray([labels[f] for f in fs], dtype=np.int8) for side, fs in files.items()}
    x = {side: load_features(stems[side]) for side in stems}
    directions = {
        "Train_to_Development": ("train", "devel"),
        "Development_to_Train": ("devel", "train"),
    }
    frozen = {}
    for name, (fit_side, eval_side) in directions.items():
        score, pred, metrics = fit_direction(x[fit_side], y[fit_side], x[eval_side], y[eval_side])
        frozen[name] = {"fit": fit_side, "eval": eval_side, "score": score, "pred": pred, "metrics": metrics}
        print(name, {k: round(v["uar"], 4) for k, v in metrics.items()})

    group_maps = {
        "k210_train_fit": load_pseudo_speakers(ROOT / "cache/pseudo_speakers/k210_seed42.tsv"),
        "pooled_k420": load_pseudo_speakers(ROOT / "cache/pseudo_speakers/pooled_k420_seed42.tsv"),
    }
    rng = np.random.default_rng(SEED)
    uncertainty = {}
    for map_name, mapping in group_maps.items():
        per_direction = {}; draw_by_direction = {}
        for direction, item in frozen.items():
            eval_side = item["eval"]
            groups = np.asarray([mapping[s] for s in stems[eval_side]])
            draws = cluster_boot(y[eval_side], item["pred"], groups, rng)
            draw_by_direction[direction] = draws
            per_direction[direction] = {
                "fusion_vs_G4": compare(draws, "G4_G9_equal", "G4"),
                "fusion_vs_G9": compare(draws, "G4_G9_equal", "G9"),
                "eval_groups": int(len(np.unique(groups))),
                "mixed_label_groups": int(sum(len(np.unique(y[eval_side][groups == g])) > 1 for g in np.unique(groups))),
            }
        average_draws = {
            model_name: 0.5 * (
                draw_by_direction["Train_to_Development"][model_name]
                + draw_by_direction["Development_to_Train"][model_name]
            )
            for model_name in ("G4", "G9", "G4_G9_equal")
        }
        uncertainty[map_name] = {
            "directions": per_direction,
            "bidirectional_average": {
                "fusion_vs_G4": compare(average_draws, "G4_G9_equal", "G4"),
                "fusion_vs_G9": compare(average_draws, "G4_G9_equal", "G9"),
            },
        }

    chunk_uncertainty = {}; chunk_draws = {}
    for direction, item in frozen.items():
        draws = chunk_boot(y[item["eval"]], item["pred"], rng)
        chunk_draws[direction] = draws
        chunk_uncertainty[direction] = {
            "fusion_vs_G4": compare(draws, "G4_G9_equal", "G4"),
            "fusion_vs_G9": compare(draws, "G4_G9_equal", "G9"),
        }
    average_chunk = {
        name: 0.5 * (chunk_draws["Train_to_Development"][name] + chunk_draws["Development_to_Train"][name])
        for name in ("G4", "G9", "G4_G9_equal")
    }
    chunk_uncertainty["bidirectional_average"] = {
        "fusion_vs_G4": compare(average_chunk, "G4_G9_equal", "G4"),
        "fusion_vs_G9": compare(average_chunk, "G4_G9_equal", "G9"),
    }

    point = {
        direction: {
            "metrics": item["metrics"],
            "fusion_delta_vs_G4": item["metrics"]["G4_G9_equal"]["uar"] - item["metrics"]["G4"]["uar"],
            "fusion_delta_vs_G9": item["metrics"]["G4_G9_equal"]["uar"] - item["metrics"]["G9"]["uar"],
        }
        for direction, item in frozen.items()
    }
    point["bidirectional_mean"] = {
        name: float(np.mean([frozen[d]["metrics"][name]["uar"] for d in directions]))
        for name in ("G4", "G9", "G4_G9_equal")
    }
    report = {
        "protocol": {
            "official_splits": "Train and Development are speaker-independent by challenge design",
            "model_selection_in_this_test": "none", "threshold": 0,
            "fusion": "fixed equal train-zscored G4+G9",
            "pseudo_groups_used_for_models_or_splits": False,
            "pseudo_groups_used_only_for_uncertainty_resampling": True,
        },
        "point_results": point,
        "proxy_cluster_uncertainty": uncertainty,
        "chunk_uncertainty_anti_conservative": chunk_uncertainty,
        "limitations": [
            "Development was inspected in earlier project iterations, so this is a protocol diagnostic rather than a pristine model-selection test.",
            "True speaker IDs are unavailable; proxy-cluster bootstrap intervals depend on the proxy map.",
            "Chunk bootstrap is anti-conservative because chunks from one subject are dependent.",
        ],
    }
    out = ROOT / "results/eval_independent_official_split_fusion.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "results/eval_independent_official_split_fusion_predictions.npz",
        train_files=np.asarray(files["train"]), devel_files=np.asarray(files["devel"]),
        y_train=y["train"], y_devel=y["devel"],
        **{f"pred__{direction}__{model_name}": item["pred"][model_name]
           for direction, item in frozen.items() for model_name in item["pred"]},
    )
    print(json.dumps({"point_results": point, "proxy_cluster_uncertainty": uncertainty}, indent=2))
    print(f"[wrote] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
