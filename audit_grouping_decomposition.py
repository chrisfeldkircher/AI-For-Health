"""Decompose the k-sensitivity swing: is it the A2.5 base, or the fusion gain,
or just beta-selection noise? For each grouping, 5-seed mean devel_test UAR of:
  - A2.5 alone (tau on that grouping's train_threshold)
  - fusion at the LOCKED k210 betas (fixed, no re-selection)
  - fusion at RE-SELECTED beta (what A5e did)
If A2.5-alone is stable but fusion@fixed collapses on honest groupings, the
fusion gain is grouping-specific. If A2.5-alone itself swings, it's partition
difficulty. Also runs pooled across 5 split seeds to check 0.628 is not one
unlucky partition. Output: results/audit_grouping_decomposition.json
"""
from __future__ import annotations
import json, sys, statistics as st
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from features import LayerWeightedPooledHead, extract_g4, extract_g5
from features.train import _pooled_collate, predict_probs
from honesty import fit_cold_probe, predict_logit, fit_zscore, fuse, sweep_tau, evaluate_at_tau
from speakers.cluster import load_pseudo_speakers

DATA_DIR=str(ROOT/"dataset"/"ComParE2017_Cold_4students"); WAV_DIR=f"{DATA_DIR}/wav"
CACHE=str(ROOT/"cache"); BK="microsoft_wavlm-large"
DEV="cuda" if torch.cuda.is_available() else "cpu"; SEEDS=[42,123,7,999,31337]
PS=ROOT/"cache"/"pseudo_speakers"
GROUPINGS={"k100":PS/"k100_seed42.tsv","k210_shipped":PS/"k210_seed42.tsv",
           "k420":PS/"k420_seed42.tsv","pooled_k420":PS/"pooled_k420_seed42.tsv"}
BETA_GRID=[0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0]
LOCK=json.loads((ROOT/"results"/"A5b_k2_5seed_lock.json").read_text())
LOCKED_BETA={int(s):float(LOCK["per_seed"][s]["k2_locked"]["beta"]) for s in LOCK["per_seed"]}

labels=load_labels(DATA_DIR)
ftr=PooledCacheDataset(DATA_DIR,CACHE,BK,split="train"); fdv=PooledCacheDataset(DATA_DIR,CACHE,BK,split="devel")
NL,SD=ftr[0]["pooled"].shape
def _st(fs): return [f[:-4] if f.endswith(".wav") else f for f in fs]
def _a2(head,files):
    ds=PooledCacheDataset(DATA_DIR,CACHE,BK,file_list=files)
    p,_=predict_probs(head,DataLoader(ds,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),DEV)
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
print("[precompute] A2.5 per-seed logits over all train+devel ...")
A2={}
for seed in SEEDS:
    h=LayerWeightedPooledHead(n_layers=NL,stat_dim=SD,proj_dim=128,n_classes=2,dropout=0.5).to(DEV)
    h.load_state_dict(torch.load(f"{CACHE}/{BK}/head_A2grouped_honestprior_seed{seed}.pt",weights_only=True,map_location=DEV)["state_dict"]); h.eval()
    lt=_a2(h,ftr.files); ld=_a2(h,fdv.files)
    A2[seed]={**{s:lt[i] for i,s in enumerate(_st(ftr.files))},**{s:ld[i] for i,s in enumerate(_st(fdv.files))}}
    del h; print(f"  seed {seed}")
def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0}

def eval_grouping(pseudo, split_seed=42):
    tf,tt=stratified_grouped_split(ftr.files,labels,pseudo,val_frac=0.10,seed=split_seed)
    dv,dt=stratified_grouped_split(fdv.files,labels,pseudo,val_frac=0.50,seed=split_seed)
    SPL={"train_fit":(tf,_st(tf)),"train_threshold":(tt,_st(tt)),"devel_test":(dt,_st(dt))}
    y={n:np.array([labels[f] for f in fs]) for n,(fs,_) in SPL.items()}
    Xg4={n:extract_g4(sm,CACHE,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
    Xg5={n:extract_g5(sm,CACHE) for n,(_,sm) in SPL.items()}
    alone,fixed,resel=[],[],[]
    for seed in SEEDS:
        a2={n:np.array([A2[seed][s] for s in sm]) for n,(_,sm) in SPL.items()}
        ta,_=sweep_tau(a2["train_threshold"],y["train_threshold"]); alone.append(evaluate_at_tau(a2["devel_test"],y["devel_test"],ta)["uar"])
        c4,s4=fit_cold_probe(Xg4["train_fit"],y["train_fit"],seed=seed); l4={n:predict_logit(c4,s4,Xg4[n]) for n in SPL}; z4=fit_zscore(l4["train_fit"]); Z4={n:z4.apply(l4[n]) for n in SPL}
        c5,s5=fit_cold_probe(Xg5["train_fit"],y["train_fit"],seed=seed); l5={n:predict_logit(c5,s5,Xg5[n]) for n in SPL}; z5=fit_zscore(l5["train_fit"]); Z5={n:z5.apply(l5[n]) for n in SPL}
        b=LOCKED_BETA[seed]
        tf_,_=sweep_tau(fuse(a2["train_threshold"],[Z4["train_threshold"],Z5["train_threshold"]],b),y["train_threshold"])
        fixed.append(evaluate_at_tau(fuse(a2["devel_test"],[Z4["devel_test"],Z5["devel_test"]],b),y["devel_test"],tf_)["uar"])
        best=None
        for bb in BETA_GRID:
            tr,ut=sweep_tau(fuse(a2["train_threshold"],[Z4["train_threshold"],Z5["train_threshold"]],bb),y["train_threshold"])
            dt_=evaluate_at_tau(fuse(a2["devel_test"],[Z4["devel_test"],Z5["devel_test"]],bb),y["devel_test"],tr)["uar"]
            if best is None or ut>best[0]: best=(ut,dt_)
        resel.append(best[1])
    return _ms(alone),_ms(fixed),_ms(resel)

print(f"\n{'grouping':<14} {'A2.5-alone':>16} {'fusion@lockedB':>16} {'fusion@reselect':>16}")
rows={}
for g,tsv in GROUPINGS.items():
    if not tsv.exists(): print(f"  {g}: missing"); continue
    a,f,r=eval_grouping(load_pseudo_speakers(tsv))
    rows[g]={"a2_alone":a,"fusion_locked_beta":f,"fusion_reselected":r,
             "fusion_gain_locked":f["mean"]-a["mean"]}
    print(f"  {g:<12} {a['mean']:.4f}+/-{a['std']:.4f}  {f['mean']:.4f}+/-{f['std']:.4f}  {r['mean']:.4f}+/-{r['std']:.4f}")

# pooled across 5 split seeds -> is 0.628 stable or one unlucky partition?
print("\n[pooled across split seeds] fusion@lockedB:")
pooled=load_pseudo_speakers(GROUPINGS["pooled_k420"]); pooled_seeds={}
for ss in [42,1,2,3,5]:
    _,f,_=eval_grouping(pooled,split_seed=ss); pooled_seeds[ss]=f["mean"]; print(f"  split_seed {ss}: {f['mean']:.4f}")
pv=list(pooled_seeds.values())
print(f"  pooled 5-split-seed: {st.mean(pv):.4f} +/- {(st.stdev(pv) if len(pv)>1 else 0):.4f}")

out={"rung_id":"audit_grouping_decomposition","locked_betas":LOCKED_BETA,
     "per_grouping":rows,"pooled_across_split_seeds":pooled_seeds,
     "reading":"If A2.5-alone is stable across groupings but fusion@lockedB drops on "
        "honest groupings, the FUSION GAIN is grouping-specific. If A2.5-alone swings, "
        "it's devel_test partition difficulty. Hidden test was 0.62."}
(ROOT/"results"/"audit_grouping_decomposition.json").write_text(json.dumps(out,indent=2))
print("\n[wrote] results/audit_grouping_decomposition.json")
