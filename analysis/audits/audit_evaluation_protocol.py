"""Reproducible audit of the Cold pipeline's performance estimate.

This checks metric math, effective speaker sample sizes, repeated use of the
nominal devel_test holdout, sensitivity to pseudo-speaker grouping, uncertainty
reported across model seeds versus data splits, and final labeled-data usage.

The official corpus facts are from Schuller et al., INTERSPEECH 2017, section
2.2 and Table 1: 630 subjects; speaker-independent Train/Development/Test
partitions with 210 speakers each; Train and Development each contain 37 cold
and 173 non-cold participants.  Section 3 notes that final test models use
Train + Development.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
MODEL_ROOT = ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from data.cached_dataset import load_labels, stratified_grouped_split  # noqa: E402
from features.train import compute_uar  # noqa: E402
from honesty.fusion import uar as fusion_uar  # noqa: E402
from speakers.cluster import load_pseudo_speakers  # noqa: E402


def _metric_check() -> dict:
    rng = np.random.default_rng(20260720)
    rows = []
    for n in (11, 101, 1000):
        y = rng.integers(0, 2, size=n)
        pred = rng.integers(0, 2, size=n)
        values = {
            "pipeline_compute_uar": float(compute_uar(pred, y)),
            "pipeline_fusion_uar": float(fusion_uar(y, pred)),
            "sklearn_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        }
        rows.append({"n": n, **values})
    max_delta = max(max(r.values()) - min(r.values()) for r in (
        {k: v for k, v in row.items() if k != "n"} for row in rows
    ))
    return {"pass": bool(max_delta < 1e-12), "max_abs_delta": max_delta, "cases": rows}


def _notebook_reuse() -> dict:
    nb = json.loads((ROOT / "model" / "run.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"]
    mentions = [s for s in code if "devel_test" in s]
    decision_markers = ("decision", "admit", "gate", "delta", "REF_", "reference")
    decision_uses = [s for s in mentions if any(marker in s for marker in decision_markers)]
    return {
        "n_code_cells": len(code),
        "n_code_cells_referencing_devel_test": len(mentions),
        "n_devel_test_cells_with_decision_or_comparison_markers": len(decision_uses),
        "holdout_remained_one_shot": len(decision_uses) <= 1,
        "note": "Static source audit; it counts experiment cells, not executions.",
    }


def _submission_integrity() -> dict:
    path = ROOT / "Feldkircher_Lee_Chouksey_submission_1.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["file_name"] for r in rows]
    labels = [r["label"] for r in rows]
    expected = sorted(p.name for p in (
        ROOT / "dataset" / "ComParE2017_Cold_4students" / "wav"
    ).glob("test_*.wav"))
    return {
        "n_rows": len(rows),
        "unique_files": len(set(names)),
        "valid_labels_only": set(labels) <= {"C", "NC"},
        "matches_test_wav_set": set(names) == set(expected),
        "predicted_cold_rate": float(np.mean(np.asarray(labels) == "C")),
    }


def main() -> None:
    labels = load_labels(str(ROOT / "dataset" / "ComParE2017_Cold_4students"))
    pseudo = load_pseudo_speakers(ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv")
    train_files = sorted(f for f in labels if f.startswith("train_"))
    devel_files = sorted(f for f in labels if f.startswith("devel_"))
    train_fit, train_threshold = stratified_grouped_split(
        train_files, labels, pseudo, val_frac=0.10, seed=42
    )

    def split_summary(files: list[str]) -> dict:
        groups = {pseudo[Path(f).stem] for f in files}
        cold_groups = {pseudo[Path(f).stem] for f in files if labels[f] == 1}
        return {
            "n_chunks": len(files),
            "n_cold_chunks": int(sum(labels[f] for f in files)),
            "n_pseudo_speakers": len(groups),
            "n_cold_pseudo_speakers": len(cold_groups),
        }

    sensitivity = json.loads((ROOT / "results" / "A5e_k_sensitivity.json").read_text())
    shadow = json.loads((ROOT / "results" / "A5b_k2_shadow_splits.json").read_text())
    verification = json.loads((ROOT / "results" / "speaker_pipeline_verification.json").read_text())
    probe_audit = json.loads((ROOT / "results" / "audit_speaker_probe_protocol.json").read_text())

    grouping_uars = {
        name: float(row["uar_devel_test"]["mean"])
        for name, row in sensitivity["groupings"].items()
    }
    model_seed_std = float(sensitivity["groupings"]["k210_shipped"]["uar_devel_test"]["std"])
    shadow_std = float(shadow["aggregate"]["ensemble_uar_across_shadow"]["std"])
    grouping_spread = float(max(grouping_uars.values()) - min(grouping_uars.values()))

    # Illustrative sampling uncertainty if chunks from one subject are strongly
    # correlated. This is not a formal CI for the submitted model; it shows why
    # treating thousands of chunks as independent is overconfident.
    recall_c = float(shadow["aggregate"]["ensemble_recall_C_across_shadow"]["mean"])
    recall_nc = float(shadow["aggregate"]["ensemble_recall_NC_across_shadow"]["mean"])
    se_chunk = 0.5 * np.sqrt(
        recall_c * (1 - recall_c) / 1011 + recall_nc * (1 - recall_nc) / 8585
    )
    se_subject = 0.5 * np.sqrt(
        recall_c * (1 - recall_c) / 37 + recall_nc * (1 - recall_nc) / 173
    )

    total_labeled = len(train_files) + len(devel_files)
    report = {
        "rung_id": "audit_evaluation_protocol",
        "metric_math": _metric_check(),
        "official_protocol": {
            "source": "schuller17_interspeech.pdf, Table 1, section 2.2, section 3",
            "speaker_independent_partitions": True,
            "speakers_per_partition": 210,
            "train_devel_cold_non_cold_speakers_each": [37, 173],
            "expected_final_fit": "Train + Development after model selection",
        },
        "effective_tuning_sample": {
            "train_fit": split_summary(train_fit),
            "train_threshold": split_summary(train_threshold),
            "risk": "beta and tau are selected using only three cold pseudo-speakers",
        },
        "devel_split_soundness": {
            "nearest_neighbor_same_cluster": verification["checks"]["V7_fragmentation"]["top1_nn_same_cluster"]["devel"],
            "same_speaker_proxy_cross_side_rate": verification["checks"]["V8_split_speaker_leakage"]["per_split"]["devel"]["nn_on_other_side"],
            "literal_speaker_disjoint_claim_supported": False,
        },
        "speaker_probe_validity": {
            "held_chunk_same_identity_top1": probe_audit["held_chunk_same_identity"]["top1"],
            "cross_official_pool_top1": probe_audit["cross_official_pool_nearest_train_centroid_labels"]["top1"],
            "understatement_ratio": probe_audit["same_identity_to_cross_pool_top1_ratio"],
            "cross_pool_majority_baseline": probe_audit["cross_official_pool_nearest_train_centroid_labels"]["majority_baseline"],
            "valid_identity_leakage_gate": False,
        },
        "holdout_reuse": _notebook_reuse(),
        "estimate_sensitivity": {
            "uar_by_grouping": grouping_uars,
            "grouping_max_minus_min": grouping_spread,
            "canonical_minus_shadow_mean": float(shadow["delta_canonical_vs_shadow_mean"]),
            "model_seed_std": model_seed_std,
            "shadow_split_std": shadow_std,
            "shadow_to_model_seed_std_ratio": shadow_std / model_seed_std,
            "grouping_spread_to_model_seed_std_ratio": grouping_spread / model_seed_std,
        },
        "uncertainty_scale_illustration": {
            "naive_chunk_iid_se_uar": float(se_chunk),
            "subject_count_se_uar": float(se_subject),
            "naive_95pct_half_width": float(1.96 * se_chunk),
            "subject_count_95pct_half_width": float(1.96 * se_subject),
            "note": "Illustrative binomial scales, not a replacement for a cluster bootstrap.",
        },
        "final_training_coverage": {
            "labeled_train_chunks": len(train_files),
            "labeled_devel_chunks": len(devel_files),
            "submitted_head_fit_chunks": len(train_fit),
            "total_available_labeled_chunks": total_labeled,
            "fraction_used_for_final_parameter_fit": len(train_fit) / total_labeled,
            "official_final_refit_followed": False,
        },
        "submission_integrity": _submission_integrity(),
        "verdict": {
            "uar_implementation_correct": True,
            "reported_internal_uar_is_reliable_hidden_test_estimate": False,
            "dominant_problems": [
                "nominal devel_test repeatedly reused for selection and comparison",
                "speaker probe evaluated across unseen identities with nearest-centroid proxy labels",
                "beta/tau tuned on only three cold pseudo-speakers",
                "uncertainty reported across model seeds rather than outer speaker samples",
                "final model not refit on Train + Development",
            ],
        },
        "recommended_protocol": [
            "Use grouped nested CV inside Train for all architecture, beta, tau, and epoch choices.",
            "Evaluate the locked pipeline once on the entire official Development partition.",
            "Use within-identity held-chunk or pairwise verification for speaker leakage; do not classify unseen speaker IDs.",
            "Report outer-fold or cluster-bootstrap uncertainty, not only training-seed standard deviation.",
            "After locking, refit or cross-fold ensemble on all Train + Development labels before Test inference.",
        ],
    }

    out = ROOT / "results" / "audit_evaluation_protocol.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print("effective train_threshold:", report["effective_tuning_sample"]["train_threshold"])
    print("estimate sensitivity:", report["estimate_sensitivity"])
    print("final fit fraction:", f"{report['final_training_coverage']['fraction_used_for_final_parameter_fit']:.1%}")
    print("[wrote]", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
