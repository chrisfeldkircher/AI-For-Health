"""Paired pseudo-speaker bootstrap for the corrected outer-CV reruns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
N_BOOT = 20000
SEED = 20260720


def uar(y: np.ndarray, pred: np.ndarray) -> float:
    return 0.5 * (np.mean(pred[y == 1] == 1) + np.mean(pred[y == 0] == 0))


def main() -> None:
    linear = np.load(ROOT / "results" / "corrected_outer_cv_linear_oof.npz")
    foundation = np.load(ROOT / "results" / "corrected_outer_cv_foundations_oof.npz")
    residual = np.load(ROOT / "results" / "g4_anchored_residual_oof.npz")
    for other in (foundation, residual):
        if not np.array_equal(linear["files"], other["files"]):
            raise RuntimeError("OOF file orders differ")
        if not np.array_equal(linear["y"], other["y"]):
            raise RuntimeError("OOF labels differ")
    y = linear["y"]
    groups = linear["groups"]
    if not np.array_equal(groups, foundation["groups"]):
        raise RuntimeError("OOF group assignments differ")

    models = {
        "G4_fixed_tau0": (linear["score__G4_gain_invariant"] >= 0).astype(np.int8),
        "G4_inner_tuned_tau": linear["pred__G4_gain_invariant"],
        "G9_CQT": linear["pred__G9_CQT"],
        "unconstrained_stack": linear["pred__fusion_G4_G9_HeAR"],
        "G4_anchor_G5": residual["pred__G5_modulation"],
        "HuBERT_large": foundation["pred__HuBERT_large"],
        "G4_anchor_HuBERT": foundation["pred__G4_anchor_plus_HuBERT_large"],
        "WavLM_large": foundation["pred__WavLM_large"],
        "G4_anchor_WavLM": foundation["pred__G4_anchor_plus_WavLM_large"],
    }
    unique_groups = np.unique(groups)
    # Speaker clusters should not cross labels; this is essential for grouped CV.
    mixed = [int(g) for g in unique_groups if len(np.unique(y[groups == g])) != 1]
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(
        len(unique_groups), np.full(len(unique_groups), 1 / len(unique_groups)),
        size=N_BOOT,
    )

    draws: dict[str, np.ndarray] = {}
    summary = {}
    for name, pred in models.items():
        correct_c = np.zeros(len(unique_groups)); total_c = np.zeros(len(unique_groups))
        correct_nc = np.zeros(len(unique_groups)); total_nc = np.zeros(len(unique_groups))
        for i, group in enumerate(unique_groups):
            group_mask = groups == group
            cold_mask = group_mask & (y == 1)
            noncold_mask = group_mask & (y == 0)
            total_c[i] = cold_mask.sum()
            correct_c[i] = np.sum(pred[cold_mask] == 1)
            total_nc[i] = noncold_mask.sum()
            correct_nc[i] = np.sum(pred[noncold_mask] == 0)
        rec_c = (weights @ correct_c) / (weights @ total_c)
        rec_nc = (weights @ correct_nc) / (weights @ total_nc)
        values = 0.5 * (rec_c + rec_nc)
        draws[name] = values
        summary[name] = {
            "uar": uar(y, pred),
            "recall_C": float(np.mean(pred[y == 1] == 1)),
            "recall_NC": float(np.mean(pred[y == 0] == 0)),
            "bootstrap_95_ci": np.quantile(values, [0.025, 0.975]).tolist(),
        }

    comparisons = {}
    for name, values in draws.items():
        if name == "G4_fixed_tau0":
            continue
        delta = values - draws["G4_fixed_tau0"]
        comparisons[name] = {
            "point_delta_uar_vs_G4": summary[name]["uar"] - summary["G4_fixed_tau0"]["uar"],
            "paired_bootstrap_95_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
            "probability_better_than_G4": float(np.mean(delta > 0)),
        }

    report = {
        "protocol": {
            "unit": "k210_seed42 pseudo-speaker cluster",
            "paired_bootstrap_draws": N_BOOT,
            "official_development_used": False,
            "warning": "clusters are inferred proxies because true speaker IDs are unavailable",
        },
        "data": {"chunks": int(len(y)), "groups": int(len(unique_groups)), "mixed_label_groups": mixed},
        "models": summary, "comparisons_vs_G4": comparisons,
    }
    out = ROOT / "results" / "corrected_rerun_bootstrap.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Single-partition corrected rerun (superseded)", "",
        "**Do not use this ranking alone for architecture selection.** The later repeated-partition "
        "reconciliation found that the chosen outer split seed was pessimistic for CQT and that a "
        "fixed equal G4+CQT fusion averaged 0.623 UAR. See `cqt_protocol_reconciliation.json` and "
        "`fixed_g4_g9_repeated_cv.json`.", "",
        "All selection was restricted to official Train. Official Development was not used.", "",
        "| Model | OOF UAR | 95% group-bootstrap CI | Delta vs G4 | P(better than G4) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in models:
        s = summary[name]; ci = s["bootstrap_95_ci"]
        if name == "G4_fixed_tau0":
            delta = "baseline"; probability = "—"
        else:
            c = comparisons[name]
            delta = f"{c['point_delta_uar_vs_G4']:+.3f}"
            probability = f"{c['probability_better_than_G4']:.3f}"
        lines.append(f"| {name} | {s['uar']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {delta} | {probability} |")
    lines += [
        "", "## Decision", "",
        "This single partition supports excluding the current neural branches, but it does not justify "
        "discarding CQT or freezing G4 alone. The repeated-partition analysis supersedes that decision. "
        "Do not submit the current unconstrained learned stack or foundation-head architecture.",
        "", "## Final-fit rule", "",
        "After architecture and hyperparameters are frozen, refit once on official Train+Development "
        "and predict Test with the fixed zero threshold. Do not compare architectures, fusion weights, "
        "thresholds, or seeds on Development again after this point.",
    ]
    md = ROOT / "results" / "corrected_rerun_summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {out.relative_to(ROOT)} and {md.relative_to(ROOT)}")
    for name in models:
        print(name, summary[name], comparisons.get(name))


if __name__ == "__main__":
    main()
