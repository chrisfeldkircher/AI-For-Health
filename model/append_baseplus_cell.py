"""Append the A5b K=3 WavLM-base-plus (layer-weighted) experiment cell to run.ipynb.

Idempotent: skips if a cell with rung_id 'A5b_k3_baseplus_lw_5seed' exists.
Does NOT execute anything. Run from model/:  python append_baseplus_cell.py
"""
import ast
import json
import uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5b_k3_baseplus_lw_5seed"

MARKDOWN = "## A5b §4.12.4 — K=3 with WavLM-base-plus + learned layer-weighted softmax\n\n" \
    "Same design as the HuBERT-base cell (§4.12.3): per-layer honesty audit, A2.5-style\n" \
    "head with honesty-prior init (5 seeds), M14 standalone pre-flight, K=3 fusion sweep.\n" \
    "Purpose: (1) put Ming's base-plus branch on the shared protocol so the layer story is\n" \
    "comparable, (2) test whether a second, cheaper WavLM tier adds fusion value.\n" \
    "First run extracts the base-plus pooled cache for train+devel (GPU, ~40-80 min).\n" \
    "You run this cell; I do not."

CODE = r'''# A5b K=3 with WavLM-base-plus + learned layer-weighted softmax (Tier-2 4.12.4).
# Clone of the HuBERT-base cell (4.12.3) with the backbone swapped: per-layer audit
# + base-plus-A2.5 head training (5 seeds, honesty-prior init T*sub@1) + standalone
# UAR + M14 pre-flight + K=3 fusion sweep. Also puts Ming's base-plus branch on the
# shared protocol (per-layer cold-vs-speaker audit on the SAME probes/splits).
# Output: results/A5b_k3_baseplus_lw_5seed.json + results/A5d_baseplus_layer_honesty.csv
# Cost: first run extracts base-plus pooled for train+devel (GPU, ~40-80 min);
# afterwards ~50 min for the 5-seed head training. Re-runs skip extraction.

import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.data import AudioDataset
from data.cached_dataset import (
    PooledCacheDataset, stratified_grouped_split, load_labels,
)
from features import LayerWeightedPooledHead, extract_g4, extract_g5
from features.backbone import build_backbone
from features.extract import extract_pooled
from features.train import (
    _pooled_collate, predict_probs, evaluate, make_balanced_sampler,
)
from honesty import (
    cold_probe, speaker_probe,
    fit_cold_probe, predict_logit, fit_zscore, fuse,
    sweep_tau, evaluate_at_tau,
)
from speakers.cluster import load_pseudo_speakers

DATA_DIR       = "../dataset/ComParE2017_Cold_4students"
WAV_DIR        = f"{DATA_DIR}/wav"
CACHE_ROOT     = "../cache"
WAVLM_BACKBONE = "microsoft_wavlm-large"
BP_BACKBONE    = "microsoft_wavlm-base-plus"
BP_BUILD_ALIAS = "wavlm-base-plus"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED     = 42
ALL_SEEDS      = [42, 123, 7, 999, 31337]
CLIP_SECS      = 8.0
BATCH_EXTRACT  = 4
PROBE_TSV      = Path(f"{CACHE_ROOT}/pseudo_speakers/k210_seed42.tsv")
BP_CACHE       = Path(f"{CACHE_ROOT}/{BP_BACKBONE}/pooled")
LOCK_JSON      = "../results/A5b_k2_5seed_lock.json"
AUDIT_OUT      = "../results/A5d_baseplus_layer_honesty.csv"
OUT_JSON       = "../results/A5b_k3_baseplus_lw_5seed.json"

T_INV          = 50.0    # honesty-prior temperature, mirrors A2.5_WavLM + HuBERT cells
N_BP_LAYERS    = 13      # 12 transformer layers + embeddings
BP_STAT_DIM    = 3072    # 4 stats x 768

BETA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
             2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

REF_K2_LOCKED_UAR_MEAN = 0.7037
REF_K2_LOCKED_UAR_STD  = 0.0060
K3_ADMISSION_THRESHOLD = REF_K2_LOCKED_UAR_MEAN + 0.005   # 0.7087
STANDALONE_FLOOR_DEFINITE_FAIL   = 0.55
STANDALONE_FLOOR_ADMIT_PLAUSIBLE = 0.61
REF_HUBERT_LW_STANDALONE = None   # fill from A5b_k3_hubert_lw_5seed.json print if desired

t_start = time.time()
print(f"[device] {DEVICE}")
print(f"[backbone] {BP_BACKBONE} ({N_BP_LAYERS} layers x {BP_STAT_DIM})")
print(f"[K=2 reference] {REF_K2_LOCKED_UAR_MEAN:.4f} +/- {REF_K2_LOCKED_UAR_STD:.4f}")
print(f"[K=3 admission] mean K=3 UAR > {K3_ADMISSION_THRESHOLD:.4f}")
print(f"[M14 floors] definite-FAIL < {STANDALONE_FLOOR_DEFINITE_FAIL}; "
      f"admit-plausible >= {STANDALONE_FLOOR_ADMIT_PLAUSIBLE}")

# =============================================================================
# STEP 0: splits + labels + pseudo-speakers (canonical k210 grouping -- keep all
# candidate comparisons under the same grouping as the locked references).
# =============================================================================
full_train = PooledCacheDataset(DATA_DIR, CACHE_ROOT, WAVLM_BACKBONE, split="train")
full_devel = PooledCacheDataset(DATA_DIR, CACHE_ROOT, WAVLM_BACKBONE, split="devel")
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
# STEP 0.5: ensure the base-plus pooled cache for train+devel (idempotent).
# NOTE: a 10-file stub from an early smoke run exists with the same config
# ([13, 3072] = 4 stats x 768, CLIP_SECS=8.0); skip_existing=True reuses it.
# If you suspect the stub was made with a different pad config, delete
# cache/microsoft_wavlm-base-plus/pooled/ first and re-run.
# =============================================================================
need_stems = sorted({s for _, (_, stems) in SPLITS.items() for s in stems})
have = sum(1 for s in need_stems if (BP_CACHE / f"{s}.pt").exists())
print(f"\n[step0.5] base-plus pooled cache: {have}/{len(need_stems)} present")
if have < len(need_stems):
    print(f"  extracting WavLM-base-plus pooled (pad={CLIP_SECS}s, batch={BATCH_EXTRACT}) -- GPU ...")
    backbone = build_backbone(BP_BUILD_ALIAS, device=DEVICE)
    for split_name in ("train", "devel"):
        ds_audio = AudioDataset(data_dir=DATA_DIR, split=split_name, use_mel=False,
                                use_opensmile=False, pad_or_truncate_secs=CLIP_SECS)
        extract_pooled(backbone=backbone, dataset=ds_audio, cache_root=CACHE_ROOT,
                       batch_size=BATCH_EXTRACT, skip_existing=True)
    del backbone
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
have = sum(1 for s in need_stems if (BP_CACHE / f"{s}.pt").exists())
assert have == len(need_stems), f"base-plus cache incomplete: {have}/{len(need_stems)}"
print(f"  cache ready ({(time.time()-t_start)/60:.1f} min so far)")

def _load_bp_full(stems):
    """Load cached base-plus [13, 3072] per stem (un-mean-pooled across layers)."""
    out = np.zeros((len(stems), N_BP_LAYERS, BP_STAT_DIM), dtype=np.float32)
    for i, stem in enumerate(stems):
        x = torch.load(BP_CACHE / f"{stem}.pt", weights_only=True, map_location="cpu")
        out[i] = x.float().numpy()
    return out

print("  loading base-plus full per-layer pooled per split ...")
X_bp_full = {s: _load_bp_full(stems) for s, (_, stems) in SPLITS.items()}
print(f"  base-plus shape: train_fit {X_bp_full['train_fit'].shape}  "
      f"devel_test {X_bp_full['devel_test'].shape}")

# =============================================================================
# STEP 1: per-layer base-plus honesty audit -> sub@1 vector for honesty prior.
# This is the shared-protocol version of the base-plus layer story: cold gain
# vs speaker gain per layer, same probes and splits as A5d_WavLM / A5d_HuBERT.
# =============================================================================
print(f"\n=== STEP 1: per-layer base-plus honesty audit ===")
X_audit_train = {L: X_bp_full["train_fit"][:, L, :] for L in range(N_BP_LAYERS)}
X_audit_eval  = {L: X_bp_full["devel_val"][:, L, :] for L in range(N_BP_LAYERS)}
y_cold_train, y_cold_eval = y["train_fit"], y["devel_val"]
y_spk_train,  y_spk_eval  = spk["train_fit"], spk["devel_val"]

audit_rows = []
for L in range(N_BP_LAYERS):
    cold_res = cold_probe(X_audit_train[L], y_cold_train,
                          X_audit_eval[L], y_cold_eval,
                          C=1.0, max_iter=2000, seed=SPLIT_SEED)
    spk_res = speaker_probe(X_audit_train[L], y_spk_train,
                            X_audit_eval[L], y_spk_eval,
                            C=1.0, max_iter=2000, seed=SPLIT_SEED)
    label_gain   = cold_res.uar - 0.50
    speaker_gain = spk_res.top1 - 1.0 / max(spk_res.n_classes, 1)
    sub_at_1     = label_gain - speaker_gain
    audit_rows.append({"layer": L, "cold_uar": float(cold_res.uar),
                       "speaker_top1": float(spk_res.top1),
                       "n_pseudo": int(spk_res.n_classes),
                       "label_gain": float(label_gain),
                       "speaker_gain": float(speaker_gain),
                       "sub_at_1": float(sub_at_1)})
    print(f"  L{L:02d}  cold_uar={cold_res.uar:.4f}  spk_top1={spk_res.top1:.4f}  "
          f"label_gain={label_gain:+.4f}  speaker_gain={speaker_gain:+.4f}  "
          f"sub@1={sub_at_1:+.4f}")

sub_at_1_vec = np.array([r["sub_at_1"] for r in audit_rows], dtype=np.float64)
top5 = np.argsort(sub_at_1_vec)[::-1][:5]
print(f"  top-5 cold-honest layers (by sub@1): "
      f"{[(int(L), float(sub_at_1_vec[L])) for L in top5]}")

import csv
Path(AUDIT_OUT).parent.mkdir(parents=True, exist_ok=True)
with open(AUDIT_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
    w.writeheader()
    for r in audit_rows:
        w.writerow(r)
print(f"  [wrote] {AUDIT_OUT}")

# =============================================================================
# STEP 2: base-plus-A2.5 head training -- per seed with honesty-prior init.
# =============================================================================
print(f"\n=== STEP 2: base-plus-A2.5 head training (5 seeds, honesty-prior init) ===")

class _BPPooledDataset(Dataset):
    def __init__(self, X, files, labels):
        self.X, self.files, self.labels = X, files, labels
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return {"pooled": torch.from_numpy(self.X[i]).float(),
                "label": torch.tensor(int(self.labels[i]), dtype=torch.long),
                "file_name": self.files[i]}
    def get_labels(self):
        return [int(L) for L in self.labels]

ds_train_fit = _BPPooledDataset(X_bp_full["train_fit"], train_fit_files, y["train_fit"])
ds_devel_val = _BPPooledDataset(X_bp_full["devel_val"], devel_val_files, y["devel_val"])

per_seed_results = []
per_seed_bp_logit = {s: {} for s in ALL_SEEDS}

for seed in ALL_SEEDS:
    print(f"\n  --- seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = LayerWeightedPooledHead(n_layers=N_BP_LAYERS, stat_dim=BP_STAT_DIM,
                                   proj_dim=128, n_classes=2, dropout=0.5).to(DEVICE)
    with torch.no_grad():
        head.layer_weights.copy_(torch.from_numpy(T_INV * sub_at_1_vec).float())
    head.scaler.to(DEVICE)
    fit_loader = DataLoader(ds_train_fit, batch_size=256, shuffle=False,
                            num_workers=0, collate_fn=_pooled_collate)
    head.scaler.fit(fit_loader, verbose=False)
    sampler = make_balanced_sampler(ds_train_fit, seed=seed)
    train_loader = DataLoader(ds_train_fit, batch_size=64, sampler=sampler,
                              num_workers=0, collate_fn=_pooled_collate)
    val_loader = DataLoader(ds_devel_val, batch_size=256, shuffle=False,
                            num_workers=0, collate_fn=_pooled_collate)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(head.param_groups(base_lr=1e-3), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=25)

    best_val_uar, best_epoch, best_state, patience = -1.0, -1, None, 0
    for epoch in range(1, 26):
        head.train()
        n_seen, run_loss = 0, 0.0
        for batch in train_loader:
            pooled = batch["pooled"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits, _ = head(pooled)
            loss = loss_fn(logits, labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optim.step()
            bs = labels.size(0)
            n_seen += bs
            run_loss += loss.item() * bs
        scheduler.step()
        val_uar, _, _, _, _ = evaluate(head, val_loader, DEVICE)
        improved = val_uar > best_val_uar
        print(f"    [ep{epoch:02d}] train_loss={run_loss/max(n_seen,1):.4f}  "
              f"val_UAR={val_uar:.4f}{'  *' if improved else ''}")
        if improved:
            best_val_uar, best_epoch = val_uar, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                print(f"    early stop at ep{epoch}")
                break
    head.load_state_dict(best_state)
    print(f"    best_val_UAR={best_val_uar:.4f} at ep{best_epoch}")

    ckpt_path = Path(f"{CACHE_ROOT}/{BP_BACKBONE}/head_A25_honestprior_seed{seed}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "val_uar": best_val_uar,
                "epoch": best_epoch, "n_layers": N_BP_LAYERS,
                "stat_dim": BP_STAT_DIM, "proj_dim": 128,
                "honesty_prior_T": T_INV, "sub_at_1": sub_at_1_vec.tolist()},
               ckpt_path)
    print(f"    [wrote ckpt] {ckpt_path}")

    head.eval()
    with torch.no_grad():
        for split_name, (files, stems) in SPLITS.items():
            ds = _BPPooledDataset(X_bp_full[split_name], files, y[split_name])
            loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                                collate_fn=_pooled_collate)
            p, _ = predict_probs(head, loader, DEVICE)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            per_seed_bp_logit[seed][split_name] = np.log(p / (1.0 - p))

    with torch.no_grad():
        final_w = head.layer_softmax().detach().cpu().numpy()
    cos_init_final = float(sub_at_1_vec.dot(final_w) /
                           (np.linalg.norm(sub_at_1_vec) * np.linalg.norm(final_w) + 1e-8))
    top5_final = np.argsort(final_w)[::-1][:5]
    per_seed_results.append({"seed": seed, "best_val_uar": float(best_val_uar),
                             "best_epoch": int(best_epoch),
                             "final_layer_weights": final_w.tolist(),
                             "cos_sub_at_1_vs_final": cos_init_final,
                             "top5_layers_final": [(int(L), float(final_w[L])) for L in top5_final]})
    print(f"    cos(sub@1, final_w) = {cos_init_final:.4f}  "
          f"top5_final={[(int(L), float(final_w[L])) for L in top5_final]}")

# =============================================================================
# STEP 3: standalone base-plus-A2.5 UAR per seed (M14 pre-flight).
# =============================================================================
print(f"\n=== STEP 3: base-plus-A2.5 standalone UAR per seed (M14 pre-flight) ===")
standalone_per_seed = []
for seed in ALL_SEEDS:
    tau, uar_thr = sweep_tau(per_seed_bp_logit[seed]["train_threshold"], y["train_threshold"])
    ev_dt = evaluate_at_tau(per_seed_bp_logit[seed]["devel_test"], y["devel_test"], tau)
    standalone_per_seed.append({"seed": seed, "tau": float(tau),
                                "uar_train_thr": float(uar_thr),
                                "uar_devel_test": float(ev_dt["uar"]),
                                "recall_C": float(ev_dt["recall_C"]),
                                "recall_NC": float(ev_dt["recall_NC"])})
    print(f"  seed {seed}: standalone base-plus-A2.5 UAR = {ev_dt['uar']:.4f}  tau*={tau:+.3f}")

standalone_uars = [r["uar_devel_test"] for r in standalone_per_seed]
standalone_mean = float(np.mean(standalone_uars))
standalone_std  = float(np.std(standalone_uars, ddof=1)) if len(standalone_uars) > 1 else 0.0
print(f"\n  base-plus-A2.5 standalone 5-seed: {standalone_mean:.4f} +/- {standalone_std:.4f}")
print(f"  vs HuBERT-base mean-pooled 0.5396; vs WavLM-Large-A2.5 ~0.656")

if standalone_mean < STANDALONE_FLOOR_DEFINITE_FAIL:
    pre_flight_decision = "skip_definite_fail"
    print(f"\n  M14 verdict: {standalone_mean:.4f} < {STANDALONE_FLOOR_DEFINITE_FAIL} -> definite FAIL; skip K=3")
elif standalone_mean >= STANDALONE_FLOOR_ADMIT_PLAUSIBLE:
    pre_flight_decision = "proceed_admit_plausible"
    print(f"\n  M14 verdict: {standalone_mean:.4f} >= {STANDALONE_FLOOR_ADMIT_PLAUSIBLE} -> admit plausible; run K=3")
else:
    pre_flight_decision = "proceed_borderline"
    print(f"\n  M14 verdict: borderline; run K=3 for confirmation")
run_k3_sweep = pre_flight_decision != "skip_definite_fail"

# =============================================================================
# STEP 4: K=3 fusion sweep (if M14 pre-flight passes).
# =============================================================================
def _ms(xs):
    return {"mean": float(st.mean(xs)),
            "std": float(st.stdev(xs)) if len(xs) > 1 else 0.0, "n": len(xs)}

print("\n  extracting G4_gain_invariant + G5_modulation ...")
X_g4 = {s: extract_g4(stems, CACHE_ROOT, WAV_DIR)[:, 4:] for s, (_, stems) in SPLITS.items()}
X_g5 = {s: extract_g5(stems, CACHE_ROOT) for s, (_, stems) in SPLITS.items()}

def _load_a2hp_wavlm_head(seed):
    sample = full_train[0]["pooled"]
    nl, sd = sample.shape
    head = LayerWeightedPooledHead(n_layers=nl, stat_dim=sd, proj_dim=128,
                                   n_classes=2, dropout=0.5).to(DEVICE)
    state = torch.load(f"{CACHE_ROOT}/{WAVLM_BACKBONE}/head_A2grouped_honestprior_seed{seed}.pt",
                       weights_only=True, map_location=DEVICE)
    head.load_state_dict(state["state_dict"])
    head.eval()
    return head

def _wavlm_a25_logit_on_split(head, files):
    ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, WAVLM_BACKBONE, file_list=files)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0,
                        collate_fn=_pooled_collate)
    p, _ = predict_probs(head, loader, DEVICE)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))

per_seed_k3_runs = []
if run_k3_sweep:
    print(f"\n=== STEP 4: K=3 sweep with base-plus-A2.5 logit as third group ===")
    for seed in ALL_SEEDS:
        print(f"\n  --- seed {seed} ---")
        wavlm_head = _load_a2hp_wavlm_head(seed)
        a2_logit = {s: _wavlm_a25_logit_on_split(wavlm_head, files)
                    for s, (files, _) in SPLITS.items()}

        clf_g4, sc_g4 = fit_cold_probe(X_g4["train_fit"], y["train_fit"], seed=seed)
        lg4 = {s: predict_logit(clf_g4, sc_g4, X_g4[s]) for s in SPLITS}
        z4 = fit_zscore(lg4["train_fit"])
        z_g4 = {s: z4.apply(lg4[s]) for s in SPLITS}

        clf_g5, sc_g5 = fit_cold_probe(X_g5["train_fit"], y["train_fit"], seed=seed)
        lg5 = {s: predict_logit(clf_g5, sc_g5, X_g5[s]) for s in SPLITS}
        z5 = fit_zscore(lg5["train_fit"])
        z_g5 = {s: z5.apply(lg5[s]) for s in SPLITS}

        zbp = fit_zscore(per_seed_bp_logit[seed]["train_fit"])
        z_bp = {s: zbp.apply(per_seed_bp_logit[seed][s]) for s in SPLITS}

        sweep_rows = []
        for beta in BETA_GRID:
            fused_thr = fuse(a2_logit["train_threshold"],
                             [z_g4["train_threshold"], z_g5["train_threshold"],
                              z_bp["train_threshold"]], beta)
            tau, uar_thr = sweep_tau(fused_thr, y["train_threshold"])
            fused_dt = fuse(a2_logit["devel_test"],
                            [z_g4["devel_test"], z_g5["devel_test"],
                             z_bp["devel_test"]], beta)
            dt_eval = evaluate_at_tau(fused_dt, y["devel_test"], tau)
            sweep_rows.append({"beta": float(beta), "tau": float(tau),
                               "tau_at_edge": bool((tau <= -3.95) or (tau >= 3.95)),
                               "uar_train_threshold": float(uar_thr),
                               "devel_test": dt_eval})
        locked = max(sweep_rows, key=lambda r: r["uar_train_threshold"])
        delta_k2 = locked["devel_test"]["uar"] - REF_K2_LOCKED_UAR_MEAN
        print(f"  seed {seed}: locked beta*={locked['beta']:.2f}  tau*={locked['tau']:+.3f}  "
              f"devel_test UAR={locked['devel_test']['uar']:.4f}  d_K2={delta_k2:+.4f}")
        per_seed_k3_runs.append({"seed": seed, "sweep": sweep_rows, "locked": locked,
                                 "delta_uar_vs_k2_locked": float(delta_k2)})

    k3_uars = [r["locked"]["devel_test"]["uar"] for r in per_seed_k3_runs]
    k3_agg = _ms(k3_uars)
    admit_k3 = k3_agg["mean"] > K3_ADMISSION_THRESHOLD
    print(f"\n  K=3 (A2 + G4 + G5 + base-plus-A2.5) LOCKED: "
          f"{k3_agg['mean']:.4f} +/- {k3_agg['std']:.4f}  "
          f"{'ADMIT' if admit_k3 else 'no admit'}")
else:
    k3_agg, admit_k3 = None, False

decision = ("k3_baseplus_admitted" if admit_k3 else
            ("skip_definite_fail" if not run_k3_sweep else "k2_stays_canonical"))
print(f"\n=== VERDICT: {decision} ===")

elapsed = (time.time() - t_start) / 60.0
out = {
    "rung_id": "A5b_k3_baseplus_lw_5seed",
    "description": (
        "A5b K=3 with WavLM-base-plus + learned layer-weighted softmax; clone of "
        "the HuBERT-base cell (4.12.3) with the backbone swapped. Per-layer "
        "honesty audit (shared protocol; comparable with Ming's base-plus layer "
        "story), A2.5-style head with honesty-prior init T*sub@1 (5 seeds), "
        "standalone M14 pre-flight (floors 0.55 / 0.61), K=3 sweep on top of the "
        "locked K=2. Admission: mean K=3 devel_test UAR > 0.7087."),
    "split_seed": SPLIT_SEED,
    "all_seeds": ALL_SEEDS,
    "split_sizes": {s: len(SPLITS[s][0]) for s in SPLITS},
    "beta_grid": BETA_GRID,
    "backbone": BP_BACKBONE,
    "n_layers": N_BP_LAYERS, "stat_dim": BP_STAT_DIM,
    "honesty_prior_T": T_INV,
    "layer_audit": audit_rows,
    "head_training": per_seed_results,
    "standalone": {"per_seed": standalone_per_seed,
                   "mean": standalone_mean, "std": standalone_std},
    "m14_pre_flight_decision": pre_flight_decision,
    "k2_locked_reference": {"mean": REF_K2_LOCKED_UAR_MEAN, "std": REF_K2_LOCKED_UAR_STD},
    "k3_admission_threshold": K3_ADMISSION_THRESHOLD,
    "k3_runs": per_seed_k3_runs,
    "k3_aggregate": k3_agg,
    "admit_k3": bool(admit_k3),
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
        if c["cell_type"] == "code" and '"A5b_k3_baseplus_lw_5seed"' in "".join(c["source"]):
            print("[skip] A5b_k3_baseplus_lw_5seed cell already present; nothing appended.")
            return
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
