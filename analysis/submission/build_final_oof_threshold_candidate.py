"""Build a non-destructive final G4+CQT candidate with a grouped-OOF threshold.

The architecture and equal branch weights stay frozen.  A single scalar
threshold is the median of the UAR-optimal thresholds from six repeated
five-fold OOF runs on Train+Development.  Proxy speaker IDs are clustered
within each official side and then offset, so Test data never participates.

The existing threshold-zero candidate and original submission are read for
validation/comparison only and are never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from analyze_fusion_threshold_policy import (
    CV_SEEDS,
    best_uar_threshold,
    metrics,
    oof_fusion,
    side_local_groups,
)
from reconcile_cqt_protocol import ROOT, load_features, load_labels


def read_submission(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["file_name"] for row in rows], np.asarray([row["label"] for row in rows])


def write_submission(path: Path, files: list[str], labels: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "label"])
        writer.writerows(zip(files, labels))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    labels_map = load_labels(str(ROOT / "dataset/ComParE2017_Cold_4students"))
    fit_files = sorted(f for f in labels_map if f.startswith("train_") or f.startswith("devel_"))
    fit_stems = [Path(f).stem for f in fit_files]
    y = np.asarray([labels_map[f] for f in fit_files], dtype=np.int8)
    x = load_features(fit_stems)

    groups = np.empty(len(fit_files), dtype=np.int32)
    group_report = {}
    offset = 0
    for side in ("train", "devel"):
        take = np.asarray([f.startswith(side + "_") for f in fit_files])
        side_stems = [stem for stem, keep in zip(fit_stems, take) if keep]
        print(f"[groups] fitting {side}-local proxy groups", flush=True)
        side_groups, detail = side_local_groups(side_stems, side)
        groups[take] = side_groups + offset
        detail["offset"] = offset
        detail["mixed_label_groups"] = int(sum(
            len(np.unique(y[take][side_groups == group])) > 1
            for group in np.unique(side_groups)
        ))
        group_report[side] = detail
        offset += int(side_groups.max()) + 1

    rows = []
    for seed in CV_SEEDS:
        print(f"[OOF] split seed {seed}", flush=True)
        score = oof_fusion(x, y, groups, seed)
        tau, selected_uar = best_uar_threshold(y, score)
        rows.append({
            "seed": seed,
            "threshold": tau,
            "fixed_zero": metrics(y, score, 0.0),
            "selected_on_same_oof_diagnostic": metrics(y, score, tau),
            "roc_selected_uar_check": selected_uar,
        })
    thresholds = np.asarray([row["threshold"] for row in rows])
    final_threshold = float(np.median(thresholds))

    bundle_path = ROOT / "results/final_G4_G9_monolithic.joblib"
    bundle = joblib.load(bundle_path)
    if bundle["fit_files"] != fit_files:
        raise RuntimeError("Final bundle fit-file order does not match current Train+Development order")

    wav_dir = ROOT / "dataset/ComParE2017_Cold_4students/wav"
    test_files = sorted(path.name for path in wav_dir.glob("test_*.wav"))
    test_stems = [Path(f).stem for f in test_files]
    test_x = load_features(test_stems)
    z = {}
    for name in ("G4", "G9"):
        raw = bundle["models"][name].decision_function(test_x[name])
        params = bundle["z_params"][name]
        z[name] = (raw - params["mean"]) / max(params["std"], 1e-8)
    score = 0.5 * (z["G4"] + z["G9"])
    fixed_labels = np.where(score >= 0.0, "C", "NC")
    selected_labels = np.where(score >= final_threshold, "C", "NC")

    fixed_path = ROOT / "results/submission_candidate_G4_G9_fixed_monolithic.csv"
    fixed_files, fixed_csv_labels = read_submission(fixed_path)
    if fixed_files != test_files or not np.array_equal(fixed_csv_labels, fixed_labels):
        raise RuntimeError("Loaded final bundle does not exactly reproduce the existing threshold-zero CSV")

    out_csv = ROOT / "results/submission_candidate_G4_G9_oof_threshold_monolithic.csv"
    write_submission(out_csv, test_files, selected_labels)
    check_files, check_labels = read_submission(out_csv)
    if check_files != test_files or not np.array_equal(check_labels, selected_labels):
        raise RuntimeError("Round-trip submission validation failed")
    if len(set(check_files)) != len(test_files) or set(np.unique(check_labels)) - {"C", "NC"}:
        raise RuntimeError("Submission contains duplicate filenames or invalid labels")

    changed = selected_labels != fixed_labels
    report = {
        "architecture": "monolithic balanced-LR G4 + G9; train-logit zscore; fixed equal average",
        "threshold_policy": "median of six repeated 5-fold grouped-OOF UAR-optimal thresholds",
        "test_data_used_for_threshold_or_model_selection": False,
        "proxy_grouping": group_report,
        "oof_thresholds": thresholds.tolist(),
        "threshold_mean": float(thresholds.mean()),
        "threshold_std": float(thresholds.std(ddof=1)),
        "final_median_threshold": final_threshold,
        "oof_rows": rows,
        "candidate": {
            "path": str(out_csv),
            "sha256": sha256(out_csv),
            "rows": len(test_files),
            "predicted_cold_rate": float(np.mean(selected_labels == "C")),
            "agreement_with_fixed_zero": float(np.mean(selected_labels == fixed_labels)),
            "changed_predictions": int(changed.sum()),
            "changed_C_to_NC": int(np.sum(changed & (fixed_labels == "C") & (selected_labels == "NC"))),
            "changed_NC_to_C": int(np.sum(changed & (fixed_labels == "NC") & (selected_labels == "C"))),
        },
        "fixed_zero_fallback": {
            "path": str(fixed_path),
            "predicted_cold_rate": float(np.mean(fixed_labels == "C")),
        },
        "performance_warning": "Test labels are unavailable; no Test UAR is computed or inferred.",
    }
    out_json = ROOT / "results/final_oof_threshold_candidate.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "oof_thresholds": report["oof_thresholds"],
        "final_median_threshold": final_threshold,
        "candidate": report["candidate"],
        "fixed_zero_fallback": report["fixed_zero_fallback"],
    }, indent=2))
    print(f"[wrote] {out_csv.relative_to(ROOT)} and {out_json.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
