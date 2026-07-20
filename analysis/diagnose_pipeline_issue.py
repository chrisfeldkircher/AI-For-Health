"""
Independent, from-raw-data reproduction of the Cold-pipeline failure mode.

Question (Christoph's track = speaker-identity handling): what made the locked
system score UAR 0.6205 on the hidden test vs a shadow-mean ~0.69 internally,
and how do we not repeat it on the next submission?

Hypothesis under test:
  The pseudo-speaker GROUPING that Chris's pipeline produced (KMeans fit on TRAIN
  ONLY, devel/test assigned by nearest train-centroid) is faithful on train but
  FRAGMENTS devel/test speakers. That makes the devel cross-validation (used to
  pick beta/tau and to report the shadow UAR) not actually speaker-disjoint, so
  the internal estimate is optimistic AND a speaker-confounded backbone head can
  masquerade as a cold detector. The pooled-fit grouping fixes the leakage; under
  it the honest estimate drops to ~0.62 == the hidden test, and the WavLM/backbone
  logit collapses onto the pure-speaker baseline.

This script reproduces, from the on-disk caches only (no stored JSONs trusted):
  PART 1  Grouping leakage (root cause): V7 top-1 NN same-cluster per split, and
          V8 same-speaker leakage across the actual grouped devel split, for the
          shipped train-only groupings vs the pooled fix.
  PART 2  Inflation mechanism: a faithful minimal fusion (backbone-cold logit
          + beta * speaker logit, tau threshold) tuned on devel_val and reported
          on devel_test under a shadow protocol, run under the LEAKY vs HONEST
          grouping. Shows the leaky grouping inflates the shadow UAR and drives
          beta UP (speaker logit rewarded), while the honest grouping drives
          beta -> 0 (speaker logit is dead weight on disjoint speakers).

Run:  <datascience python> diagnose_pipeline_issue.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import stratified_grouped_split, load_labels  # noqa: E402

NPZ = ROOT / "cache" / "ecapa-voxceleb" / "ecapa_embeddings.npz"
META = ROOT / "paper_data" / "eda" / "chunk_metadata.csv"
WAVLM_PCA = ROOT / "paper_data" / "eda" / "wavlm_a25_pca128_fp16.npy"
DATA_DIR = str(ROOT / "dataset" / "ComParE2017_Cold_4students")
PS = ROOT / "cache" / "pseudo_speakers"

GROUPINGS = {
    "shipped_train_only_k210": PS / "k210_seed42.tsv",            # what drove the headline
    "ablation_train_only_k420": PS / "ablation_train_only_k420_seed42.tsv",
    "ablation_pooled_k210": PS / "ablation_pooled_k210_seed42.tsv",
    "pooled_k420_FIX": PS / "pooled_k420_seed42.tsv",             # the honest fix
}
LEAKY = "shipped_train_only_k210"
HONEST = "pooled_k420_FIX"
SEED = 42
SHADOW_SEEDS = [42, 123, 7, 999, 31337]
BETA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]


def load_grouping(tsv: Path) -> dict[str, int]:
    out = {}
    with tsv.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            stem, _sp, clu = line.rstrip("\n").split("\t")
            out[stem] = int(clu)
    return out


def uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted average recall = balanced accuracy over {NC=0, C=1}."""
    rec = []
    for c in (0, 1):
        m = y_true == c
        rec.append((y_pred[m] == c).mean() if m.any() else 0.0)
    return float(np.mean(rec))


# ----------------------------------------------------------------------------
# Load raw artifacts
# ----------------------------------------------------------------------------
print("=" * 78)
print("LOADING RAW ARTIFACTS")
print("=" * 78)
d = np.load(NPZ, allow_pickle=True)
stems_all = d["stems"].astype(str)
split_all = d["split"].astype(str)
emb_all = d["embeddings"].astype(np.float32)
ecapa = {s: emb_all[i] for i, s in enumerate(stems_all)}
print(f"  ECAPA embeddings: {emb_all.shape}  splits={dict(zip(*np.unique(split_all, return_counts=True)))}")

meta = list(csv.DictReader(META.open(encoding="utf-8")))
meta_stems = [r["file_stem"] for r in meta]
cold = {r["file_stem"]: int(r["cold_label"]) for r in meta}
print(f"  metadata (labeled train+devel): {len(meta)} rows")

wavlm = np.load(WAVLM_PCA).astype(np.float32)   # aligned to metadata row order
assert wavlm.shape[0] == len(meta), f"wavlm rows {wavlm.shape[0]} != meta {len(meta)}"
wavlm_map = {meta_stems[i]: wavlm[i] for i in range(len(meta))}
print(f"  WavLM A2.5 PCA-128 (backbone cold feature): {wavlm.shape}")

groupings = {name: load_grouping(p) for name, p in GROUPINGS.items() if p.exists()}
print(f"  groupings loaded: {list(groupings.keys())}")
labels_map = load_labels(DATA_DIR)


# ----------------------------------------------------------------------------
# PART 1 - ROOT CAUSE: grouping leakage (V7 fragmentation + V8 split leakage)
# ----------------------------------------------------------------------------
def v7_top1_same_cluster(split_name: str, groups: dict[str, int]):
    m = split_all == split_name
    if not all(s in groups for s in stems_all[m]):
        return None  # grouping does not cover this split
    X = normalize(emb_all[m], axis=1)
    lab = np.array([groups[s] for s in stems_all[m]])
    n = X.shape[0]
    same, B = 0, 2048
    for i0 in range(0, n, B):
        sim = X[i0:i0 + B] @ X.T
        for r in range(sim.shape[0]):
            sim[r, i0 + r] = -2.0
        same += int((lab[sim.argmax(1)] == lab[i0:i0 + B]).sum())
    return same / n


def v8_split_leakage(split_name: str, groups: dict[str, int], frac: float):
    files = sorted(f for f in labels_map if f.startswith(split_name + "_"))
    a, b = stratified_grouped_split(files, labels_map, groups, val_frac=frac, seed=SEED)
    a_st = set(f[:-4] for f in a)
    order = [f[:-4] for f in files]
    X = normalize(np.vstack([ecapa[s] for s in order]), axis=1)
    side = np.array([0 if s in a_st else 1 for s in order], np.int8)
    n = X.shape[0]
    nn_side = np.empty(n, np.int8)
    B = 2048
    for i0 in range(0, n, B):
        sim = X[i0:i0 + B] @ X.T
        for r in range(sim.shape[0]):
            sim[r, i0 + r] = -2.0
        nn_side[i0:i0 + B] = side[sim.argmax(1)]
    ga = {groups[s] for s in a_st}
    gb = {groups[f[:-4]] for f in b}
    return float((nn_side != side).mean()), len(ga & gb)


print("\n" + "=" * 78)
print("PART 1 - ROOT CAUSE: is the grouping speaker-honest on held-out splits?")
print("=" * 78)
print(f"{'grouping':<28} {'V7 train':>9} {'V7 devel':>9} {'V7 test':>9} "
      f"{'V8 devel':>9} {'overlap':>8}")
print("-" * 78)
part1 = {}
for name, g in groupings.items():
    v7 = {sp: v7_top1_same_cluster(sp, g) for sp in ("train", "devel", "test")}
    v8_devel, overlap = v8_split_leakage("devel", g, 0.50)
    part1[name] = {"v7": v7, "v8_devel": v8_devel, "overlap": overlap}
    fmt = lambda x: "     n/a" if x is None else f"{x:>9.3f}"
    print(f"{name:<28} {fmt(v7['train'])} {fmt(v7['devel'])} {fmt(v7['test'])} "
          f"{v8_devel:>9.3f} {overlap:>8d}")
print("-" * 78)
print("V7 = frac of clips whose nearest same-split neighbour (~same true speaker)")
print("     shares its pseudo-speaker label. Low devel/test => fragmentation.")
print("V8 devel = frac of devel clips whose nearest neighbour lands on the OTHER")
print("     side of the grouped devel_val/devel_test split => same-speaker leakage.")


# ----------------------------------------------------------------------------
# PART 2 - MECHANISM: does the leaky grouping inflate the shadow UAR by
# rewarding a speaker-confounded logit?
# ----------------------------------------------------------------------------
def fit_logit(feat_map, standardize=True):
    """Balanced LR feat->cold, fit on TRAIN; return decision_function on devel."""
    tr = [s for s in meta_stems if s.startswith("train_")]
    dv = [s for s in meta_stems if s.startswith("devel_")]
    Xtr = np.vstack([feat_map[s] for s in tr]); ytr = np.array([cold[s] for s in tr])
    Xdv = np.vstack([feat_map[s] for s in dv])
    if standardize:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xdv = sc.transform(Xtr), sc.transform(Xdv)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Xtr, ytr)
    ztr = lr.decision_function(Xtr)
    zdv = lr.decision_function(Xdv)
    # z-score the logit on train stats so the beta grid is scale-fair
    mu, sd = ztr.mean(), ztr.std() + 1e-9
    return dv, (zdv - mu) / sd, np.array([cold[s] for s in dv])


print("\n" + "=" * 78)
print("PART 2 - MECHANISM: speaker-masquerade inflation under leaky vs honest grouping")
print("=" * 78)
print("Fitting backbone-cold logit (WavLM A2.5 PCA-128) and speaker logit (ECAPA)...")
dv_stems, z_cold, y_dev = fit_logit(wavlm_map)
dv_stems2, z_spk, _ = fit_logit(ecapa)
assert dv_stems == dv_stems2
stem2i = {s: i for i, s in enumerate(dv_stems)}

# standalone honest bars (beta=0 / speaker-only), grouped devel_test, shadow-mean
def standalone_uar(z, groups):
    outs = []
    for sd in SHADOW_SEEDS:
        files = sorted(f for f in labels_map if f.startswith("devel_"))
        val, test = stratified_grouped_split(files, labels_map, groups, val_frac=0.5, seed=sd)
        vi = np.array([stem2i[f[:-4]] for f in val]); ti = np.array([stem2i[f[:-4]] for f in test])
        # tune tau on val, report on test
        best_tau, best_u = 0.0, -1
        for tau in np.linspace(z[vi].min(), z[vi].max(), 81):
            u = uar(y_dev[vi], (z[vi] > tau).astype(int))
            if u > best_u:
                best_u, best_tau = u, tau
        outs.append(uar(y_dev[ti], (z[ti] > best_tau).astype(int)))
    return float(np.mean(outs)), float(np.std(outs))


def fusion_shadow(groups):
    """Shadow protocol: tune (beta, tau) on devel_val, report devel_test UAR."""
    uars, betas = [], []
    files = sorted(f for f in labels_map if f.startswith("devel_"))
    for sd in SHADOW_SEEDS:
        val, test = stratified_grouped_split(files, labels_map, groups, val_frac=0.5, seed=sd)
        vi = np.array([stem2i[f[:-4]] for f in val]); ti = np.array([stem2i[f[:-4]] for f in test])
        best = (-1, 0.0, 0.0)  # uar_val, beta, tau
        for beta in BETA_GRID:
            s_val = z_cold[vi] + beta * z_spk[vi]
            for tau in np.linspace(s_val.min(), s_val.max(), 81):
                u = uar(y_dev[vi], (s_val > tau).astype(int))
                if u > best[0]:
                    best = (u, beta, tau)
        _, beta, tau = best
        s_test = z_cold[ti] + beta * z_spk[ti]
        uars.append(uar(y_dev[ti], (s_test > tau).astype(int)))
        betas.append(beta)
    return float(np.mean(uars)), float(np.std(uars)), betas


print("\nStandalone honest bars (grouped devel_test, shadow-mean over 5 seeds):")
for label, z in [("backbone WavLM-cold logit alone", z_cold), ("ECAPA speaker logit alone", z_spk)]:
    m, s = standalone_uar(z, groupings[HONEST])
    print(f"  {label:<34}: {m:.4f} +/- {s:.4f}")

print("\nFusion  s = z_cold + beta * z_spk,  (beta, tau) tuned on devel_val:")
print(f"{'grouping':<28} {'shadow UAR':>18} {'mean beta*':>12} {'betas':>0}")
print("-" * 78)
part2 = {}
for name in (LEAKY, HONEST):
    m, s, betas = fusion_shadow(groupings[name])
    part2[name] = {"uar_mean": m, "uar_std": s, "betas": betas}
    print(f"{name:<28} {m:>10.4f} +/- {s:<5.4f} {np.mean(betas):>12.2f}   {betas}")
print("-" * 78)

# ----------------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
lv7d = part1[LEAKY]["v7"]["devel"]; hv7d = part1[HONEST]["v7"]["devel"]
lv8 = part1[LEAKY]["v8_devel"]; hv8 = part1[HONEST]["v8_devel"]
print(f"[root cause] devel fragmentation V7: leaky {lv7d:.3f} -> honest {hv7d:.3f}")
print(f"[root cause] devel split leakage V8: leaky {lv8:.3f} -> honest {hv8:.3f}")
lu = part2[LEAKY]["uar_mean"]; hu = part2[HONEST]["uar_mean"]
lb = float(np.mean(part2[LEAKY]["betas"])); hb = float(np.mean(part2[HONEST]["betas"]))
print(f"[mechanism ] fusion shadow UAR:      leaky {lu:.4f} -> honest {hu:.4f}   (delta {hu-lu:+.4f})")
print(f"[mechanism ] mean selected beta*:    leaky {lb:.2f}  -> honest {hb:.2f}")
print(f"             (higher beta under leaky = speaker logit rewarded because")
print(f"              devel_val speakers reappear in devel_test)")
print(f"[hidden test] locked system on true held-out speakers: 0.6205")
