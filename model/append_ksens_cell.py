"""Append the A5e k-sensitivity cell to run.ipynb (idempotent, does not run).
Run from model/:  python append_ksens_cell.py
"""
import ast, json, uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5e_k_sensitivity"

MARKDOWN = "## A5e §5.7 — pseudo-speaker k sensitivity of the locked K=2\n\n" \
    "Answers the standing question: k=100/210/420 were built as a robustness bracket\n" \
    "around the ~210-speakers-per-split assumption, but the sensitivity check was never\n" \
    "run. This re-locks the K=2 pipeline (5 seeds, beta swept on train_threshold, tau\n" \
    "locked, reported on devel_test) under each grouping, plus the pooled_k420 grouping.\n" \
    "If the 5-seed devel_test UAR is stable across groupings, conclusions do not hinge on\n" \
    "getting k exactly right. You run this cell; I do not."

CODE = r'''# A5e: k-sensitivity of the locked K=2 (A2.5 + G4_gi + G5_mod), 5 seeds.
# Re-locks the SAME pipeline under 4 pseudo-speaker groupings (k100, k210, k420,
# pooled_k420) by swapping only the grouping tsv that defines the grouped splits.
# A2.5 logits come from the committed heads (grouping-independent at inference);
# G4/G5 cold probes + z-scores are refit on each grouping's train_fit; beta swept
# and tau locked on that grouping's train_threshold; reported on devel_test.
# Output: results/A5e_k_sensitivity.json   Cost: ~5-8 min on cached features.

import json, statistics as st, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from features import LayerWeightedPooledHead, extract_g4, extract_g5
from features.train import _pooled_collate, predict_probs
from honesty import fit_cold_probe, predict_logit, fit_zscore, fuse, sweep_tau, evaluate_at_tau
from speakers.cluster import load_pseudo_speakers

DATA_DIR="../dataset/ComParE2017_Cold_4students"; WAV_DIR=f"{DATA_DIR}/wav"
CACHE_ROOT="../cache"; BACKBONE="microsoft_wavlm-large"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED=42; ALL_SEEDS=[42,123,7,999,31337]
OUT_JSON="../results/A5e_k_sensitivity.json"
BETA_GRID=[0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0]
GROUPINGS = {
    "k100":        f"{CACHE_ROOT}/pseudo_speakers/k100_seed42.tsv",
    "k210_shipped":f"{CACHE_ROOT}/pseudo_speakers/k210_seed42.tsv",
    "k420":        f"{CACHE_ROOT}/pseudo_speakers/k420_seed42.tsv",
    "pooled_k420": f"{CACHE_ROOT}/pseudo_speakers/pooled_k420_seed42.tsv",
}
t0=time.time(); print(f"[device] {DEVICE}")
labels_map = load_labels(DATA_DIR)
full_train = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, split="train")
full_devel = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, split="devel")
sample = full_train[0]["pooled"]; NL, SD = sample.shape

def _stems(fs): return [f[:-4] if f.endswith(".wav") else f for f in fs]

# cache A2.5 logits per seed on the full train + devel file lists ONCE
# (inference is grouping-independent; the grouping only re-buckets these files).
def a2_logit_map(files):
    """dict stem->ensemble-per-seed logit array is overkill; instead return a
    dict split-agnostic: we compute per-seed logit for each file, keyed by stem."""
    return None  # placeholder; we compute per seed below

print("[precompute] A2.5 per-seed logits over all train+devel files ...")
all_train = full_train.files; all_devel = full_devel.files
def _a2_logit(head, files):
    ds = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, file_list=files)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=_pooled_collate)
    p,_ = predict_probs(head, loader, DEVICE); p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))
a2_by_seed = {}
for seed in ALL_SEEDS:
    head = LayerWeightedPooledHead(n_layers=NL, stat_dim=SD, proj_dim=128, n_classes=2, dropout=0.5).to(DEVICE)
    stt = torch.load(f"{CACHE_ROOT}/{BACKBONE}/head_A2grouped_honestprior_seed{seed}.pt",
                     weights_only=True, map_location=DEVICE)
    head.load_state_dict(stt["state_dict"]); head.eval()
    lt = _a2_logit(head, all_train); ld = _a2_logit(head, all_devel)
    a2_by_seed[seed] = {**{s: lt[i] for i,s in enumerate(_stems(all_train))},
                        **{s: ld[i] for i,s in enumerate(_stems(all_devel))}}
    del head
    print(f"  seed {seed} A2.5 logits cached")

def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0,"n":len(xs)}

results = {}
for gname, tsv in GROUPINGS.items():
    if not Path(tsv).exists():
        print(f"[skip] {gname}: {tsv} missing"); continue
    print(f"\n=== grouping {gname} ===")
    pseudo = load_pseudo_speakers(Path(tsv))
    tf, tt = stratified_grouped_split(all_train, labels_map, pseudo, val_frac=0.10, seed=SPLIT_SEED)
    dv, dtst = stratified_grouped_split(all_devel, labels_map, pseudo, val_frac=0.50, seed=SPLIT_SEED)
    SPL = {"train_fit":(tf,_stems(tf)),"train_threshold":(tt,_stems(tt)),
           "devel_val":(dv,_stems(dv)),"devel_test":(dtst,_stems(dtst))}
    y = {n: np.array([labels_map[f] for f in fs],dtype=np.int64) for n,(fs,_) in SPL.items()}
    Xg4 = {n: extract_g4(sm,CACHE_ROOT,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
    Xg5 = {n: extract_g5(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
    uars=[]; betas=[]
    for seed in ALL_SEEDS:
        a2 = {n: np.array([a2_by_seed[seed][s] for s in sm]) for n,(_,sm) in SPL.items()}
        c4,s4 = fit_cold_probe(Xg4["train_fit"], y["train_fit"], seed=seed)
        l4={n:predict_logit(c4,s4,Xg4[n]) for n in SPL}; z4=fit_zscore(l4["train_fit"]); Z4={n:z4.apply(l4[n]) for n in SPL}
        c5,s5 = fit_cold_probe(Xg5["train_fit"], y["train_fit"], seed=seed)
        l5={n:predict_logit(c5,s5,Xg5[n]) for n in SPL}; z5=fit_zscore(l5["train_fit"]); Z5={n:z5.apply(l5[n]) for n in SPL}
        best=None
        for beta in BETA_GRID:
            tau,uthr = sweep_tau(fuse(a2["train_threshold"],[Z4["train_threshold"],Z5["train_threshold"]],beta), y["train_threshold"])
            dt = evaluate_at_tau(fuse(a2["devel_test"],[Z4["devel_test"],Z5["devel_test"]],beta), y["devel_test"], tau)
            if best is None or uthr>best[0]: best=(uthr,beta,tau,dt["uar"])
        uars.append(best[3]); betas.append(best[1])
    agg=_ms(uars)
    results[gname]={"uar_devel_test":agg,"locked_betas":betas,
                    "n_speakers_used":len(set(pseudo.values())),
                    "split_sizes":{n:len(SPL[n][0]) for n in SPL}}
    print(f"  5-seed devel_test UAR = {agg['mean']:.4f} +/- {agg['std']:.4f}  betas={betas}")

ref = results.get("k210_shipped",{}).get("uar_devel_test",{}).get("mean")
print(f"\n=== SUMMARY (ref = k210_shipped {ref}) ===")
for g,r in results.items():
    m=r["uar_devel_test"]["mean"]; d = (m-ref) if ref is not None else float("nan")
    print(f"  {g:<14s} {m:.4f} +/- {r['uar_devel_test']['std']:.4f}   d_vs_k210={d:+.4f}")
spread = (max(r["uar_devel_test"]["mean"] for r in results.values())
          - min(r["uar_devel_test"]["mean"] for r in results.values())) if results else 0.0
print(f"  max-min spread across groupings = {spread:.4f}  (shadow sigma ~0.0157)")

out={"rung_id":"A5e_k_sensitivity",
     "description":"Sensitivity of the locked K=2 (A2.5+G4_gi+G5_mod) 5-seed "
        "devel_test UAR to the pseudo-speaker grouping (k100/k210/k420/pooled_k420). "
        "Same pipeline, only the grouping tsv that defines the grouped splits changes. "
        "Small spread => conclusions do not hinge on the exact speaker count.",
     "split_seed":SPLIT_SEED,"all_seeds":ALL_SEEDS,"beta_grid":BETA_GRID,
     "groupings":results,"reference":"k210_shipped",
     "max_min_spread":spread,"shadow_sigma_ref":0.0157,
     "elapsed_minutes":(time.time()-t0)/60.0}
Path(OUT_JSON).parent.mkdir(parents=True,exist_ok=True)
Path(OUT_JSON).write_text(json.dumps(out,indent=2))
print(f"\n[wrote] {OUT_JSON}")
'''

def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"]=="code" and '"A5e_k_sensitivity"' in "".join(c["source"]):
            print("[skip] A5e_k_sensitivity already present."); return
    ast.parse(CODE)
    def mk(t,s):
        c={"cell_type":t,"metadata":{},"source":s.splitlines(keepends=True),"id":uuid.uuid4().hex[:8]}
        if t=="code": c["execution_count"]=None; c["outputs"]=[]
        return c
    nb["cells"].append(mk("markdown",MARKDOWN)); nb["cells"].append(mk("code",CODE))
    NB.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
    print(f"[appended] {RUNG}; notebook now has {len(nb['cells'])} cells.")

if __name__=="__main__": main()
