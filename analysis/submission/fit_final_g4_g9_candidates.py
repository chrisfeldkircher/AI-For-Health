"""Fit frozen G4+G9 final candidates on Train+Development and predict Test.

Creates new candidate files only; never overwrites the existing submission.
Performance is not estimated here because Test labels are unavailable.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from reconcile_cqt_protocol import ROOT, load_features, load_labels, model

sys.path.insert(0, str(ROOT / "model"))
from data.data import _load_audio  # noqa: E402
from features import cqt_features  # noqa: E402
from speakers.cluster import load_pseudo_speakers  # noqa: E402


CACHE = ROOT / "cache/handcrafted/cqt"
WAV_DIR = ROOT / "dataset/ComParE2017_Cold_4students/wav"
CLIP_SECONDS = 8.0
ENSEMBLE_FOLDS = 10


def extract_one(stem: str) -> int:
    target = CACHE / f"{stem}.npy"
    if target.exists():
        return 0
    audio, sr = _load_audio(str(WAV_DIR / f"{stem}.wav"))
    audio = audio[: int(CLIP_SECONDS * sr)]
    feature = cqt_features(audio, sr=sr).astype(np.float32)
    np.save(target, feature)
    return 1


def ensure_test_cqt(stems: list[str]) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [stem for stem in stems if not (CACHE / f"{stem}.npy").exists()]
    if todo:
        from joblib import Parallel, delayed
        n_jobs = max(1, min(12, (os.cpu_count() or 4) - 2))
        print(f"[CQT] extracting {len(todo)} Test files with n_jobs={n_jobs}", flush=True)
        written = Parallel(n_jobs=n_jobs, backend="loky", batch_size=16, verbose=10)(
            delayed(extract_one)(stem) for stem in todo
        )
        n_written = int(sum(written))
    else:
        n_jobs = 0; n_written = 0
    missing = [stem for stem in stems if not (CACHE / f"{stem}.npy").exists()]
    if missing:
        raise RuntimeError(f"CQT Test cache incomplete: {len(missing)} missing")
    return {"needed": len(stems), "previously_cached": len(stems) - len(todo),
            "written": n_written, "n_jobs": n_jobs}


def fit_branches(x, y, train, test_x):
    fitted = {}; test_raw = {}; test_z = {}; z_params = {}
    for name in ("G4", "G9"):
        pipe = model().fit(x[name][train], y[train])
        train_score = pipe.decision_function(x[name][train])
        test_score = pipe.decision_function(test_x[name])
        mu = float(train_score.mean()); sigma = float(max(train_score.std(), 1e-8))
        fitted[name] = pipe; test_raw[name] = test_score
        test_z[name] = (test_score - mu) / sigma
        z_params[name] = {"mean": mu, "std": sigma}
    return fitted, test_raw, test_z, z_params


def labels_from_score(score):
    return np.where(score >= 0, "C", "NC")


def write_submission(path: Path, files: list[str], labels: np.ndarray):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "label"])
        writer.writerows(zip(files, labels))


def agreement(a, b):
    return float(np.mean(a == b))


def main():
    data_dir = ROOT / "dataset/ComParE2017_Cold_4students"
    labels = load_labels(str(data_dir))
    fit_files = sorted(f for f in labels if f.startswith("train_") or f.startswith("devel_"))
    test_files = sorted(path.name for path in WAV_DIR.glob("test_*.wav"))
    fit_stems = [Path(f).stem for f in fit_files]; test_stems = [Path(f).stem for f in test_files]
    y = np.asarray([labels[f] for f in fit_files], dtype=np.int8)
    extraction = ensure_test_cqt(test_stems)
    print("[load] final Train+Development and Test features", flush=True)
    x = load_features(fit_stems); test_x = load_features(test_stems)
    all_idx = np.arange(len(y))

    fitted, mono_raw, mono_z, mono_z_params = fit_branches(x, y, all_idx, test_x)
    mono_g4_score = mono_raw["G4"]
    mono_fusion_score = 0.5 * (mono_z["G4"] + mono_z["G9"])
    mono_g4_label = labels_from_score(mono_g4_score)
    mono_fusion_label = labels_from_score(mono_fusion_score)

    pseudo = load_pseudo_speakers(ROOT / "cache/pseudo_speakers/pooled_k420_seed42.tsv")
    raw_group = np.asarray([pseudo[s] for s in fit_stems], dtype=np.int64)
    side = np.asarray([0 if f.startswith("train_") else 1 for f in fit_files], dtype=np.int8)
    groups = raw_group + side.astype(np.int64) * (int(raw_group.max()) + 1)
    cv = StratifiedGroupKFold(n_splits=ENSEMBLE_FOLDS, shuffle=True, random_state=42)
    ensemble_g4_scores = []; ensemble_fusion_scores = []; ensemble_members = []
    for member, (train, _) in enumerate(cv.split(np.zeros(len(y)), y, groups), 1):
        member_fit, member_raw, member_z, member_z_params = fit_branches(x, y, train, test_x)
        ensemble_g4_scores.append(member_raw["G4"])
        ensemble_fusion_scores.append(0.5 * (member_z["G4"] + member_z["G9"]))
        ensemble_members.append({"models": member_fit, "z_params": member_z_params,
                                 "n_train": int(len(train))})
        print(f"[ensemble] fitted member {member}/{ENSEMBLE_FOLDS}", flush=True)
    ensemble_g4_score = np.mean(ensemble_g4_scores, axis=0)
    ensemble_fusion_score = np.mean(ensemble_fusion_scores, axis=0)
    ensemble_g4_label = labels_from_score(ensemble_g4_score)
    ensemble_fusion_label = labels_from_score(ensemble_fusion_score)

    results_dir = ROOT / "results"
    mono_csv = results_dir / "submission_candidate_G4_G9_fixed_monolithic.csv"
    ensemble_csv = results_dir / "submission_candidate_G4_G9_fixed_ensemble10.csv"
    fallback_csv = results_dir / "submission_candidate_G4_only_monolithic.csv"
    write_submission(mono_csv, test_files, mono_fusion_label)
    write_submission(ensemble_csv, test_files, ensemble_fusion_label)
    write_submission(fallback_csv, test_files, mono_g4_label)
    joblib.dump(
        {"models": fitted, "z_params": mono_z_params, "fusion": "equal", "threshold": 0.0,
         "fit_files": fit_files},
        results_dir / "final_G4_G9_monolithic.joblib", compress=3,
    )
    joblib.dump(
        {"members": ensemble_members, "fusion": "equal", "threshold": 0.0,
         "fit_files": fit_files, "grouping": "pooled_k420 offset by official side"},
        results_dir / "final_G4_G9_ensemble10.joblib", compress=3,
    )

    old_path = ROOT / "Feldkircher_Lee_Chouksey_submission_1.csv"
    old = None
    if old_path.exists():
        with old_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        old_map = {r["file_name"]: r["label"] for r in rows}
        old = np.asarray([old_map[f] for f in test_files])
    report = {
        "fit": {"chunks": len(fit_files), "cold_chunks": int(y.sum()),
                "train_chunks": sum(f.startswith("train_") for f in fit_files),
                "development_chunks": sum(f.startswith("devel_") for f in fit_files)},
        "test": {"chunks": len(test_files), "cqt_extraction": extraction},
        "candidates": {
            "G4_only_monolithic": {"cold_rate": float(np.mean(mono_g4_label == "C"))},
            "G4_G9_monolithic": {"cold_rate": float(np.mean(mono_fusion_label == "C"))},
            "G4_only_ensemble10": {"cold_rate": float(np.mean(ensemble_g4_label == "C")),
                                   "agreement_with_monolithic": agreement(ensemble_g4_label, mono_g4_label),
                                   "score_correlation": float(np.corrcoef(ensemble_g4_score, mono_g4_score)[0, 1])},
            "G4_G9_ensemble10": {"cold_rate": float(np.mean(ensemble_fusion_label == "C")),
                                  "agreement_with_monolithic": agreement(ensemble_fusion_label, mono_fusion_label),
                                  "score_correlation": float(np.corrcoef(ensemble_fusion_score, mono_fusion_score)[0, 1])},
        },
        "old_submission_comparison": None if old is None else {
            "old_cold_rate": float(np.mean(old == "C")),
            "agreement_old_vs_G4_G9_monolithic": agreement(old, mono_fusion_label),
            "agreement_old_vs_G4_only": agreement(old, mono_g4_label),
        },
        "performance_warning": "No Test UAR is computed or inferred; Test labels are unavailable.",
        "files": {"monolithic_fusion": str(mono_csv), "ensemble_fusion": str(ensemble_csv),
                  "G4_fallback": str(fallback_csv)},
    }
    out = results_dir / "final_candidate_prediction_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[wrote] {out.relative_to(ROOT)} and three candidate CSVs", flush=True)


if __name__ == "__main__":
    main()
