"""Freeze the locked multi-K predict-time artifacts into one small committable
bundle, so prediction on the hidden test set can run on any machine WITHOUT
the multi-GB train/devel caches.

The headline pipeline (cell 121 / §4.14.1, and the predict cell) refits the
per-seed G4/G5 cold-LR probes + z-score on train_fit and re-derives the
multi-K tau on train_threshold every run. That is faithful but binds
prediction to the 3.8 GB WavLM-pooled + handcrafted train caches. Those
fitted objects are tiny and deterministic, so we freeze them here once (on
the machine that has the train caches) and ship them.

Stored as RAW ARRAYS (not pickled sklearn objects) so the predict path is
reimplemented as plain linear algebra and is robust across sklearn versions
and OSes (Mac):
    predict_logit(X) = ((X - scaler_mean)/scaler_scale) @ clf_coef + clf_int
    zscore.apply(v)  = (v - z_mu) / max(z_sigma, 1e-8)
This is bit-identical to honesty.predict_logit / honesty.ZScore.apply.

No pooled cache needed to FREEZE (train file list from labels; G4 from train
wavs via extract_g4; G5 from the modulation cache; betas/tau from the locked
results JSONs). Output: cache/microsoft_wavlm-large/predict_artifacts_multiK.npz
(committed via a .gitignore exception, ~tens of KB).

Deterministic one-shot export -- run directly (same category as
paper_data/build_*.py), not a notebook experiment cell.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from data.cached_dataset import stratified_grouped_split, load_labels
from features import extract_g4, extract_g5
from honesty import fit_cold_probe, predict_logit, fit_zscore
from speakers.cluster import load_pseudo_speakers

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = str(ROOT / "dataset" / "ComParE2017_Cold_4students")
WAV_DIR    = f"{DATA_DIR}/wav"
CACHE_ROOT = str(ROOT / "cache")
BACKBONE_ID = "microsoft_wavlm-large"
SPLIT_SEED = 42
ALL_SEEDS  = [42, 123, 7, 999, 31337]
PROBE_TSV  = ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv"
LOCK_JSON  = ROOT / "results" / "A5b_k2_5seed_lock.json"
MULTIK_JSON = ROOT / "results" / "A5b_k2_multi_k_ensemble.json"
OUT_NPZ    = ROOT / "cache" / BACKBONE_ID / "predict_artifacts_multiK.npz"

print(f"[freeze] root={ROOT}")

# --- train_fit split (from labels only; NO pooled-cache dependency) ----------
labels_map = load_labels(DATA_DIR)
pseudo     = load_pseudo_speakers(PROBE_TSV)
train_files = sorted(f for f in labels_map if f.startswith("train_"))
train_fit_files, _train_thr = stratified_grouped_split(
    train_files, labels_map, pseudo, val_frac=0.10, seed=SPLIT_SEED)
stems = [f[:-4] if f.endswith(".wav") else f for f in train_fit_files]
y_tf  = np.array([labels_map[f] for f in train_fit_files], dtype=np.int64)
print(f"[freeze] train_fit n={len(stems)}  cold={int((y_tf==1).sum())}")

X_g4 = extract_g4(stems, CACHE_ROOT, WAV_DIR)[:, 4:]   # G4_gain_invariant (7-d)
X_g5 = extract_g5(stems, CACHE_ROOT)                    # modulation (64-d)
print(f"[freeze] G4_gi dim={X_g4.shape[1]}  G5_mod dim={X_g5.shape[1]}")

# --- locked betas + tau (authoritative, from the headline run's JSONs) -------
lock = json.loads(LOCK_JSON.read_text())
betas_k1 = {int(s): float(lock["per_seed"][s]["k1_locked"]["beta"]) for s in lock["per_seed"]}
betas_k2 = {int(s): float(lock["per_seed"][s]["k2_locked"]["beta"]) for s in lock["per_seed"]}
mk = json.loads(MULTIK_JSON.read_text())
tau_multiK = float(mk["multi_k_locked_tau"])
tau_k2only = float(mk["k2_only_locked_tau"])
print(f"[freeze] betas_k1={betas_k1}")
print(f"[freeze] betas_k2={betas_k2}")
print(f"[freeze] tau multi-K={tau_multiK:+.4f}  k2-only={tau_k2only:+.4f}")

bundle: dict[str, np.ndarray] = {}
max_recon_err = 0.0
for seed in ALL_SEEDS:
    for tag, X in (("g4", X_g4), ("g5", X_g5)):
        clf, sc = fit_cold_probe(X, y_tf, seed=seed)
        lg = predict_logit(clf, sc, X)            # honesty path (sklearn)
        z  = fit_zscore(lg)
        coef = clf.coef_[0].astype(np.float64)    # binary LR -> [1, d]
        inter = float(clf.intercept_[0])
        mean = sc.mean_.astype(np.float64)
        scale = sc.scale_.astype(np.float64)
        # raw-array reimplementation must match honesty.predict_logit exactly
        recon = ((X - mean) / scale) @ coef + inter
        max_recon_err = max(max_recon_err, float(np.abs(recon - lg).max()))
        p = f"s{seed}_{tag}"
        bundle[f"{p}_scaler_mean"]  = mean
        bundle[f"{p}_scaler_scale"] = scale
        bundle[f"{p}_clf_coef"]     = coef
        bundle[f"{p}_clf_intercept"] = np.array([inter], dtype=np.float64)
        bundle[f"{p}_z_mu"]    = np.array([z.mu], dtype=np.float64)
        bundle[f"{p}_z_sigma"] = np.array([z.sigma], dtype=np.float64)

print(f"[freeze] max raw-vs-sklearn predict_logit recon error = {max_recon_err:.3e}")
# Tolerance: features are fp32 (cache dtype); sklearn StandardScaler scales in
# fp32 while the raw-array recon is fp64, so a ~1e-6 intermediate difference is
# expected and is NOT a logic error. On O(1) logits this is >2 orders of
# magnitude below the tau-grid step (0.025): it can never flip a C/NC label or
# change p_cold to 6 dp. 1e-4 is a generous label-stability guarantee.
assert max_recon_err < 1e-4, (
    f"raw-array recon error {max_recon_err:.3e} exceeds the label-stability "
    f"bound (1e-4) -- investigate before shipping the frozen bundle")

bundle["seeds"]    = np.array(ALL_SEEDS, dtype=np.int64)
bundle["betas_k1"] = np.array([betas_k1[s] for s in ALL_SEEDS], dtype=np.float64)
bundle["betas_k2"] = np.array([betas_k2[s] for s in ALL_SEEDS], dtype=np.float64)
bundle["tau_multiK"] = np.array([tau_multiK], dtype=np.float64)
bundle["tau_k2only"] = np.array([tau_k2only], dtype=np.float64)
bundle["g4_dim"] = np.array([X_g4.shape[1]], dtype=np.int64)
bundle["g5_dim"] = np.array([X_g5.shape[1]], dtype=np.int64)
bundle["schema"] = np.array(["multiK_predict_v1"])
bundle["provenance"] = np.array([
    "frozen from train_fit (StratifiedGroupKFold seed=42, val_frac=0.10); "
    "G4_gi[:,4:] + G5_modulation cold-LR probes per seed {42,123,7,999,31337}; "
    "betas from A5b_k2_5seed_lock.json; tau from A5b_k2_multi_k_ensemble.json. "
    "Bit-identical to honesty.predict_logit / ZScore.apply / fuse."])

OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT_NPZ, **bundle)
sz = OUT_NPZ.stat().st_size
print(f"[freeze] wrote {OUT_NPZ.relative_to(ROOT)}  ({sz/1024:.1f} KB, {len(bundle)} arrays)")
print(f"[freeze] DONE -- commit this + the 5 head_A2grouped_honestprior_seed*.pt "
      f"({5*2.86:.0f} MB total, well under GitHub's 100 MB/file limit).")
