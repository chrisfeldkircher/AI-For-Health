"""Experiment 1 (label-free): does re-standardizing the per-group z-term on the
TEST pool collapse the fused-logit mean back toward 0 and kill the +34 tail?

Decisive fork from the fusion-fragility hypothesis. Uses ONLY:
  - cached test G4/G5 feature arrays (cache/handcrafted/{g4,modulation}/test_*.npy)
  - the frozen probe bundle (predict_artifacts_multiK.npz): per seed x group raw
    scaler/clf arrays + z_mu/z_sigma + betas
  - the existing fused predictions (results/test_predictions_multiK.csv)

NO WavLM forward pass and NO test labels are needed, because re-standardization
only touches the group z-terms; the A2 (WavLM head) contribution is unchanged:

  fused_seed = A2_seed + (0.5*b1 + 0.25*b2)*z4_seed + 0.25*b2*z5_seed
  new_fused  = CSV_multiK + mean_seed[ contrib(test-z) - contrib(train-z) ]

where contrib uses the same betas; only the (mu, sigma) used to standardize the
per-group LOGITS changes from frozen-train to test-pool.

Outputs distributional diagnostics (means, stds, quantiles, tail mass) -- NOT
leaderboard metrics -- exactly as the reviewer specified.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

CACHE = Path("cache")
G4_DIR = CACHE / "handcrafted" / "g4"
G5_DIR = CACHE / "handcrafted" / "modulation"
NPZ = CACHE / "microsoft_wavlm-large" / "predict_artifacts_multiK.npz"
CSV = Path("results") / "test_predictions_multiK.csv"
TAU = -1.625

# ---- 0. test stem order: take it from the CSV so everything aligns -----------
import csv as _csv
rows = list(_csv.DictReader(CSV.open(newline="", encoding="utf-8")))
stems = [r["file_name"][:-4] if r["file_name"].endswith(".wav") else r["file_name"] for r in rows]
csv_multiK = np.array([float(r["ensemble_logit_multiK"]) for r in rows], dtype=np.float64)
N = len(stems)
print(f"[load] {N} test rows from CSV; CSV multiK mean={csv_multiK.mean():.3f} "
      f"min={csv_multiK.min():.2f} max={csv_multiK.max():.2f}")

# ---- 1. load cached test G4/G5 feature arrays --------------------------------
def _load_stack(d, stems):
    arrs = [np.load(d / f"{s}.npy") for s in stems]
    return np.vstack([a.reshape(1, -1) for a in arrs]).astype(np.float64)

X_g4_full = _load_stack(G4_DIR, stems)
X_g4 = X_g4_full[:, 4:]                  # G4_gain_invariant (matches freeze: [:,4:])
X_g5 = _load_stack(G5_DIR, stems)
print(f"[load] G4 full dim={X_g4_full.shape[1]} -> gi dim={X_g4.shape[1]}; G5 dim={X_g5.shape[1]}")

# ---- 2. frozen probe bundle --------------------------------------------------
b = np.load(NPZ, allow_pickle=True)
seeds = b["seeds"].tolist()
bk1 = {int(s): float(b["betas_k1"][i]) for i, s in enumerate(seeds)}
bk2 = {int(s): float(b["betas_k2"][i]) for i, s in enumerate(seeds)}

def probe_logit(X, seed, tag):
    p = f"s{seed}_{tag}"
    mean = b[f"{p}_scaler_mean"]; scale = b[f"{p}_scaler_scale"]
    coef = b[f"{p}_clf_coef"]; inter = float(b[f"{p}_clf_intercept"][0])
    return ((X - mean) / scale) @ coef + inter

# ---- 3. per-group test logits + the train z-params ---------------------------
print("\n[shift] per-group TEST logit moments vs FROZEN-TRAIN z-params:")
print("        (z_mu/z_sigma are the train_fit group-logit mean/std)")
print(f"{'seed':>6} {'grp':>4} {'train_mu':>9} {'train_sig':>9} {'test_mu':>9} {'test_sig':>9} {'mean_shift_in_train_sigma':>26}")
g4_logit = {}
g5_logit = {}
for s in seeds:
    for tag, X, store in (("g4", X_g4, g4_logit), ("g5", X_g5, g5_logit)):
        lg = probe_logit(X, s, tag)
        store[s] = lg
        zmu = float(b[f"s{s}_{tag}_z_mu"][0]); zsig = float(b[f"s{s}_{tag}_z_sigma"][0])
        shift_in_sigma = (lg.mean() - zmu) / zsig
        print(f"{s:>6} {tag:>4} {zmu:>9.4f} {zsig:>9.4f} {lg.mean():>9.4f} {lg.std():>9.4f} {shift_in_sigma:>26.3f}")

# ---- 4. decompose: group contribution to the fused logit (frozen-train z) -----
# contrib_seed = (0.5*b1 + 0.25*b2)*z4 + 0.25*b2*z5  with z = (logit - z_mu)/z_sigma
def zscore(lg, seed, tag):
    zmu = float(b[f"s{seed}_{tag}_z_mu"][0]); zsig = float(b[f"s{seed}_{tag}_z_sigma"][0])
    return (lg - zmu) / max(zsig, 1e-8)

def zscore_testpool(lg):
    return (lg - lg.mean()) / max(lg.std(), 1e-8)   # re-standardize on the 9551 test rows

contrib_train = np.zeros(N)
contrib_test  = np.zeros(N)
for s in seeds:
    w4 = 0.5 * bk1[s] + 0.25 * bk2[s]
    w5 = 0.25 * bk2[s]
    z4_tr = zscore(g4_logit[s], s, "g4"); z5_tr = zscore(g5_logit[s], s, "g5")
    z4_te = zscore_testpool(g4_logit[s]); z5_te = zscore_testpool(g5_logit[s])
    contrib_train += (w4 * z4_tr + w5 * z5_tr)
    contrib_test  += (w4 * z4_te + w5 * z5_te)
contrib_train /= len(seeds)
contrib_test  /= len(seeds)

# A2 (WavLM head) ensemble contribution, backed out exactly:
A2 = csv_multiK - contrib_train

print("\n[decompose] ensemble fused-logit components on test:")
def stats(name, x):
    q = np.quantile(x, [0.01, 0.10, 0.50, 0.90, 0.99])
    print(f"  {name:<26} mean={x.mean():>8.3f}  std={x.std():>7.3f}  "
          f"q01={q[0]:>7.2f} q10={q[1]:>7.2f} q50={q[2]:>7.2f} q90={q[3]:>7.2f} q99={q[4]:>7.2f}  max={x.max():>6.2f}")
stats("A2 (WavLM head)", A2)
stats("group contrib (train-z)", contrib_train)
stats("group contrib (test-z)", contrib_test)
stats("CSV fused multiK", csv_multiK)

# ---- 5. re-fuse with test-pool standardization -------------------------------
new_fused = csv_multiK + (contrib_test - contrib_train)
print("\n[re-fuse] effect of re-standardizing the group z-term on the test pool:")
stats("fused FROZEN-train z (orig)", csv_multiK)
stats("fused TEST-pool z (re-std)", new_fused)

def predc(x):
    return float((x >= TAU).mean())
print(f"\n  predicted-C rate @ tau={TAU}:  frozen-z = {predc(csv_multiK):.4f}   test-z = {predc(new_fused):.4f}")
print(f"  tail mass (logit > +10):       frozen-z = {(csv_multiK>10).mean():.4f}   test-z = {(new_fused>10).mean():.4f}")
print(f"  tail mass (logit > +20):       frozen-z = {(csv_multiK>20).mean():.4f}   test-z = {(new_fused>20).mean():.4f}")
print(f"  left mass  (logit < -10):      frozen-z = {(csv_multiK< -10).mean():.4f}   test-z = {(new_fused< -10).mean():.4f}")

# rank correlation between orig and re-standardized fusion (how much ranking moves)
from scipy.stats import spearmanr, kendalltau
rho, _ = spearmanr(csv_multiK, new_fused)
print(f"\n  Spearman(orig, re-std) = {rho:.4f}   (1.0 = ranking unchanged)")

# How much does the group contribution std inflate vs train design intent?
# By design on train, group contrib std should be ~ sqrt((w4)^2 + (w5)^2) since z~N(0,1) indep.
import math
for s in seeds[:1]:
    w4 = 0.5 * bk1[s] + 0.25 * bk2[s]; w5 = 0.25 * bk2[s]
    design_std = math.sqrt(w4**2 + w5**2)
    print(f"  [seed {s}] design group-contrib std (train, z~N(0,1)) ~= {design_std:.2f}")
print(f"  observed ensemble group-contrib std on test (train-z) = {contrib_train.std():.2f}")
print(f"  observed ensemble group-contrib std on test (test-z)  = {contrib_test.std():.2f}")
