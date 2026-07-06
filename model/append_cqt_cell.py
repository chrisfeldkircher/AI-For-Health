"""Append the A5b K=3 CQT (G9) experiment cell to run.ipynb.

Idempotent: if a cell whose rung_id is 'A5b_k3_cqt_5seed' already exists, this
does nothing. Otherwise it appends a markdown header + code cell at the end.
Does NOT execute anything. Run from model/:  python append_cqt_cell.py
"""
import ast
import json
import uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5b_k3_cqt_5seed"

MARKDOWN = "## A5b §4.11.2.2 — K=3 with G9 (constant-Q transform)\n\n" \
    "Tests whether a constant-Q spectral group (finer low-frequency resolution\n" \
    "than the mel front end G5 and WavLM already see) adds cold signal on top of\n" \
    "the locked K=2, and whether it passes the speaker-honesty gate. Same protocol\n" \
    "as the eGeMAPS and HuBERT candidate cells. You run this cell; I do not."

CODE = r'''# A5b K=3 candidate: G9 = constant-Q transform (CQT) spectral group (Tier-2 4.11.2.2).
# Mirrors the eGeMAPS K=3 cell exactly, with two additions:
#   (a) STEP 1 extracts the CQT cache for train+devel if missing (parallel, CPU).
#   (b) STEP 2 runs the standalone honesty audit for G9 (cold UAR + speaker top-1)
#       so we see immediately whether G9 raises UAR by leaking speaker identity.
# Two configs: A = K=2 with G9 replacing G5; B = K=3 = A2 + G4_gi + G5_mod + G9.
# 5 seeds {42,123,7,999,31337}. Output: results/A5b_k3_cqt_5seed.json
# Cost: first run extracts CQT for ~19k train+devel clips (parallel, ~10-20 min),
# then ~3-5 min on cached features; re-runs skip extraction.

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cached_dataset import (
    PooledCacheDataset, stratified_grouped_split, load_labels,
)
from data.data import _load_audio
from features import (
    LayerWeightedPooledHead, extract_g4, extract_g5, extract_g9, cqt_features,
)
from features.train import _pooled_collate, predict_probs
from honesty import (
    cold_probe, speaker_probe,
    fit_cold_probe, predict_logit, fit_zscore, fuse,
    sweep_tau, evaluate_at_tau,
)
from speakers.cluster import load_pseudo_speakers

DATA_DIR   = "../dataset/ComParE2017_Cold_4students"
WAV_DIR    = f"{DATA_DIR}/wav"
CACHE_ROOT = "../cache"
BACKBONE   = "microsoft_wavlm-large"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED = 42
ALL_SEEDS  = [42, 123, 7, 999, 31337]
CLIP_SECS  = 8.0
N_JOBS     = -1
PROBE_TSV  = Path(f"{CACHE_ROOT}/pseudo_speakers/k210_seed42.tsv")
CQT_DIR    = Path(f"{CACHE_ROOT}/handcrafted/cqt")
OUT_JSON   = "../results/A5b_k3_cqt_5seed.json"

BETA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
             2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

REF_K2_LOCKED_UAR_MEAN = 0.7037
REF_K2_LOCKED_UAR_STD  = 0.0060
K3_ADMISSION_THRESHOLD = REF_K2_LOCKED_UAR_MEAN + 0.005   # 0.7087
REF_A2_SPEAKER_TOP1    = 0.0501                            # A2.5 fused speaker probe top-1
SPEAKER_CHANCE         = 1.0 / 210.0                       # ~0.0048

t_start = time.time()
print(f"[device] {DEVICE}")
print(f"[K=2 5-seed reference] {REF_K2_LOCKED_UAR_MEAN:.4f} +/- {REF_K2_LOCKED_UAR_STD:.4f}")
print(f"[K=3 admission threshold] > {K3_ADMISSION_THRESHOLD:.4f}")
print(f"[speaker gate] G9 speaker top-1 should not exceed A2 ref {REF_A2_SPEAKER_TOP1:.4f} "
      f"(chance {SPEAKER_CHANCE:.4f})")

# =============================================================================
# STEP 0: splits + labels + pseudo-speakers
# =============================================================================
full_train = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, split="train")
full_devel = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, split="devel")
labels_map = load_labels(DATA_DIR)
pseudo     = load_pseudo_speakers(PROBE_TSV)
train_fit_files, train_thr_files  = stratified_grouped_split(
    full_train.files, labels_map, pseudo, val_frac=0.10, seed=SPLIT_SEED)
devel_val_files, devel_test_files = stratified_grouped_split(
    full_devel.files, labels_map, pseudo, val_frac=0.50, seed=SPLIT_SEED)

def _stems(files): return [f[:-4] if f.endswith(".wav") else f for f in files]
SPLITS = {
    "train_fit":       (train_fit_files,  _stems(train_fit_files)),
    "train_threshold": (train_thr_files,  _stems(train_thr_files)),
    "devel_val":       (devel_val_files,  _stems(devel_val_files)),
    "devel_test":      (devel_test_files, _stems(devel_test_files)),
}
y = {name: np.array([labels_map[f] for f in files], dtype=np.int64)
     for name, (files, _) in SPLITS.items()}
spk = {name: np.array([pseudo[s] for s in stems], dtype=np.int64)
       for name, (_, stems) in SPLITS.items()}

# =============================================================================
# STEP 1: ensure CQT cache for the stems we need (train+devel), then load G9
# =============================================================================
print(f"\n=== STEP 1: CQT cache ({CQT_DIR}) ===")
CQT_DIR.mkdir(parents=True, exist_ok=True)
need_stems = sorted({s for _, (_, stems) in SPLITS.items() for s in stems})
todo = [s for s in need_stems if not (CQT_DIR / f"{s}.npy").exists()]
print(f"  {len(need_stems)} stems needed; {len(need_stems) - len(todo)} cached; {len(todo)} to compute")

def _one_cqt(stem):
    target = CQT_DIR / f"{stem}.npy"
    if target.exists():
        return 0
    audio, sr = _load_audio(str(Path(WAV_DIR) / f"{stem}.wav"))  # float32, 16 kHz
    x = audio[:int(CLIP_SECS * sr)]                              # trim to <=8 s, no pad (matches G5)
    feat = cqt_features(x, sr=sr).astype(np.float32)
    np.save(target, feat)
    return 1

if todo:
    try:
        from joblib import Parallel, delayed
        Parallel(n_jobs=N_JOBS, backend="loky", verbose=5, batch_size=16)(
            delayed(_one_cqt)(s) for s in todo)
    except ImportError:
        print("  [warn] joblib not available -- serial fallback")
        for s in todo:
            _one_cqt(s)
n_now = sum(1 for s in need_stems if (CQT_DIR / f"{s}.npy").exists())
assert n_now == len(need_stems), f"CQT cache incomplete: {n_now}/{len(need_stems)}"
print(f"  CQT cache ready ({(time.time()-t_start)/60:.1f} min so far)")

X_g4 = {s: extract_g4(stems, CACHE_ROOT, WAV_DIR)[:, 4:] for s, (_, stems) in SPLITS.items()}
X_g5 = {s: extract_g5(stems, CACHE_ROOT)                 for s, (_, stems) in SPLITS.items()}
X_g9 = {s: extract_g9(stems, CACHE_ROOT)                 for s, (_, stems) in SPLITS.items()}
print(f"  feature dims: G4_gi={X_g4['train_fit'].shape[1]}  "
      f"G5={X_g5['train_fit'].shape[1]}  G9_cqt={X_g9['train_fit'].shape[1]}")

# =============================================================================
# STEP 2: standalone honesty audit of G9 (does it leak speaker identity?)
# =============================================================================
print(f"\n=== STEP 2: G9 standalone honesty audit (train_fit -> devel_val) ===")
g9_cold = cold_probe(X_g9["train_fit"], y["train_fit"], X_g9["devel_val"], y["devel_val"])
g9_spk  = speaker_probe(X_g9["train_fit"], spk["train_fit"], X_g9["devel_val"], spk["devel_val"])
print(f"  G9 cold UAR        = {g9_cold.uar:.4f}  (rec_C={g9_cold.recall_pos:.3f}, rec_NC={g9_cold.recall_neg:.3f})")
print(f"  G9 speaker top-1   = {g9_spk.top1:.4f}  (nmi={g9_spk.nmi:.3f}, chance={SPEAKER_CHANCE:.4f})")
speaker_gate_pass = g9_spk.top1 <= REF_A2_SPEAKER_TOP1
print(f"  speaker gate (top-1 <= A2 ref {REF_A2_SPEAKER_TOP1:.4f}): "
      f"{'PASS' if speaker_gate_pass else 'FAIL -- G9 carries speaker identity'}")

def _load_a2hp_head(seed):
    sample = full_train[0]["pooled"]
    nl, sd = sample.shape
    head = LayerWeightedPooledHead(n_layers=nl, stat_dim=sd, proj_dim=128,
                                   n_classes=2, dropout=0.5).to(DEVICE)
    state = torch.load(f"{CACHE_ROOT}/{BACKBONE}/head_A2grouped_honestprior_seed{seed}.pt",
                       weights_only=True, map_location=DEVICE)
    head.load_state_dict(state["state_dict"])
    head.eval()
    return head

def _a2_logit_on_split(head, files):
    ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, file_list=files)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                        collate_fn=_pooled_collate)
    p, _ = predict_probs(head, loader, DEVICE)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))

def _ms(xs):
    return {"mean": float(st.mean(xs)),
            "std":  float(st.stdev(xs)) if len(xs) > 1 else 0.0, "n": len(xs)}

# =============================================================================
# STEP 3: precompute A2.5 logits + per-group z_logits (G4, G5, G9) per seed
# =============================================================================
print(f"\n=== STEP 3: precompute A2.5 logits + z_logits per seed ===")
per_seed = {}
for seed in ALL_SEEDS:
    print(f"  seed {seed} ...")
    head = _load_a2hp_head(seed)
    a2_logit = {s: _a2_logit_on_split(head, files) for s, (files, _) in SPLITS.items()}

    def _zg(X):
        clf, sc = fit_cold_probe(X["train_fit"], y["train_fit"], seed=seed)
        lg = {s: predict_logit(clf, sc, X[s]) for s in SPLITS}
        z  = fit_zscore(lg["train_fit"])
        return {s: z.apply(lg[s]) for s in SPLITS}

    z_g4, z_g5, z_g9 = _zg(X_g4), _zg(X_g5), _zg(X_g9)
    tau_g9a, _ = sweep_tau(z_g9["train_threshold"], y["train_threshold"])
    g9_alone = evaluate_at_tau(z_g9["devel_test"], y["devel_test"], tau_g9a)
    per_seed[seed] = {"a2_logit": a2_logit, "z_g4": z_g4, "z_g5": z_g5, "z_g9": z_g9,
                      "g9_standalone_uar_devel_test": float(g9_alone["uar"])}

def _sweep(groups_key_lists):
    """groups_key_lists: list of per-seed z-logit dict keys to fuse with A2."""
    runs = []
    for seed in ALL_SEEDS:
        c = per_seed[seed]
        rows = []
        for beta in BETA_GRID:
            zt = [c[k]["train_threshold"] for k in groups_key_lists]
            zd = [c[k]["devel_test"]      for k in groups_key_lists]
            tau, uar_thr = sweep_tau(fuse(c["a2_logit"]["train_threshold"], zt, beta),
                                     y["train_threshold"])
            dt = evaluate_at_tau(fuse(c["a2_logit"]["devel_test"], zd, beta),
                                 y["devel_test"], tau)
            rows.append({"beta": float(beta), "tau": float(tau),
                         "tau_at_edge": bool((tau <= -3.95) or (tau >= 3.95)),
                         "uar_train_threshold": float(uar_thr), "devel_test": dt})
        locked = max(rows, key=lambda r: r["uar_train_threshold"])
        runs.append({"seed": seed, "sweep": rows, "locked": locked})
    return runs

# =============================================================================
# STEP 4: config A (K=2 replace G5 with G9) + config B (K=3 add G9)
# =============================================================================
print(f"\n=== STEP 4: config A (A2+G4+G9) and config B (A2+G4+G5+G9) ===")
configA = _sweep(["z_g4", "z_g9"])
configB = _sweep(["z_g4", "z_g5", "z_g9"])
A_uars = [r["locked"]["devel_test"]["uar"] for r in configA]; A_agg = _ms(A_uars)
B_uars = [r["locked"]["devel_test"]["uar"] for r in configB]; B_agg = _ms(B_uars)
A_betas = [r["locked"]["beta"] for r in configA]
B_betas = [r["locked"]["beta"] for r in configB]
print(f"  Config A (K=2 A2+G4+G9): {A_agg['mean']:.4f} +/- {A_agg['std']:.4f}  betas={A_betas}")
print(f"  Config B (K=3 A2+G4+G5+G9): {B_agg['mean']:.4f} +/- {B_agg['std']:.4f}  betas={B_betas}")

admit_k3 = B_agg["mean"] > K3_ADMISSION_THRESHOLD and speaker_gate_pass
if admit_k3:
    decision = "k3_cqt_admitted"
elif B_agg["mean"] > K3_ADMISSION_THRESHOLD and not speaker_gate_pass:
    decision = "k3_cqt_uar_ok_but_speaker_gate_fail"
elif A_agg["mean"] > REF_K2_LOCKED_UAR_MEAN + 0.005 and speaker_gate_pass:
    decision = "k2_with_cqt_admitted"
else:
    decision = "k2_g4_g5_stays_canonical"

print(f"\n=== VERDICT ===")
print(f"  G9 speaker gate:  {'PASS' if speaker_gate_pass else 'FAIL'}  (top-1 {g9_spk.top1:.4f})")
print(f"  K=3 UAR admit (> {K3_ADMISSION_THRESHOLD:.4f}): {'YES' if B_agg['mean'] > K3_ADMISSION_THRESHOLD else 'NO'}  ({B_agg['mean']:.4f})")
print(f"  DECISION: {decision}")

elapsed = (time.time() - t_start) / 60.0
out = {
    "rung_id": "A5b_k3_cqt_5seed",
    "description": (
        "A5b K=3 candidate G9 = constant-Q transform (168-d: 84 CQT bins x "
        "{mean,std} of per-bin dB over time). Config A = K=2 with G9 replacing "
        "G5_modulation; Config B = K=3 (A2 + G4_gi + G5_mod + G9). 5 seeds, "
        "per-seed argmax beta* lock on train_threshold, reported on devel_test. "
        "Two-dimensional honesty gate: ADMIT K=3 iff mean devel_test UAR > "
        "0.7087 AND G9 standalone speaker-probe top-1 <= A2 ref 0.0501."
    ),
    "split_seed": SPLIT_SEED,
    "all_seeds": ALL_SEEDS,
    "split_sizes": {s: len(SPLITS[s][0]) for s in SPLITS},
    "beta_grid": BETA_GRID,
    "feature_dims": {"g4_gain_invariant": int(X_g4["train_fit"].shape[1]),
                     "g5_modulation": int(X_g5["train_fit"].shape[1]),
                     "g9_cqt": int(X_g9["train_fit"].shape[1])},
    "k2_locked_reference": {"mean": REF_K2_LOCKED_UAR_MEAN, "std": REF_K2_LOCKED_UAR_STD},
    "k3_admission_threshold": K3_ADMISSION_THRESHOLD,
    "g9_standalone_honesty": {
        "cold_uar_devel_val": float(g9_cold.uar),
        "cold_recall_pos": float(g9_cold.recall_pos),
        "cold_recall_neg": float(g9_cold.recall_neg),
        "speaker_top1_devel_val": float(g9_spk.top1),
        "speaker_nmi_devel_val": float(g9_spk.nmi),
        "speaker_chance": SPEAKER_CHANCE,
        "reference_a2_speaker_top1": REF_A2_SPEAKER_TOP1,
        "speaker_gate_pass": bool(speaker_gate_pass),
        "g9_standalone_uar_devel_test_per_seed": {
            str(s): per_seed[s]["g9_standalone_uar_devel_test"] for s in ALL_SEEDS},
    },
    "config_a_k2_replacement": {"description": "K=2 with G9 replacing G5_modulation",
                                "runs": configA, "locked_betas": A_betas,
                                "uar_devel_test": A_agg,
                                "delta_vs_k2_locked_mean": A_agg["mean"] - REF_K2_LOCKED_UAR_MEAN},
    "config_b_k3_addition": {"description": "K=3 = A2 + G4_gi + G5_mod + G9_cqt",
                             "runs": configB, "locked_betas": B_betas,
                             "uar_devel_test": B_agg,
                             "delta_vs_k2_locked_mean": B_agg["mean"] - REF_K2_LOCKED_UAR_MEAN,
                             "admit_k3": bool(admit_k3)},
    "decision": decision,
    "elapsed_minutes": elapsed,
}
Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_JSON).write_text(json.dumps(out, indent=2))
print(f"\n[wrote] {OUT_JSON}")
print(f"[done] total wall time = {elapsed:.2f} min")
'''


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code" and '"A5b_k3_cqt_5seed"' in "".join(c["source"]):
            print("[skip] A5b_k3_cqt_5seed cell already present; nothing appended.")
            return
    # validate the code parses before inserting
    ast.parse(CODE)

    def _mk(cell_type, src):
        cell = {"cell_type": cell_type, "metadata": {},
                "source": src.splitlines(keepends=True), "id": uuid.uuid4().hex[:8]}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return cell

    nb["cells"].append(_mk("markdown", MARKDOWN))
    nb["cells"].append(_mk("code", CODE))
    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"[appended] markdown + code cell for {RUNG}; notebook now has {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
