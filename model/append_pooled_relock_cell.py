"""Append A5f: retrain the WavLM A2.5 heads under the pooled (speaker-honest)
grouping and re-lock K=2. The honest headline. Idempotent; does NOT run.
From model/:  python append_pooled_relock_cell.py
"""
import ast, json, uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5f_pooled_relock"

MARKDOWN = "## A5f §6.1 — honest-grouping re-lock (retrain A2.5 + K=2 under pooled speakers)\n\n" \
    "audit_grouping_decomposition.py showed the shipped-head K=2 reads ~0.64 under the\n" \
    "speaker-honest pooled_k420 grouping vs 0.704 under k210, matching the hidden-test\n" \
    "0.62. This removes the last confound (the shipped heads were early-stopped on k210\n" \
    "devel_val): it RETRAINS the 5 A2.5 heads with train_fit/devel_val defined by\n" \
    "pooled_k420 (same recipe: honesty-prior init, T=50, 25 ep, patience 6), saves them\n" \
    "to a POOLED ckpt path (k210 heads untouched), then re-locks K=2. This becomes the\n" \
    "honest headline. You run this cell; I do not.  Cost: ~50 min on GPU."

CODE = r'''# A5f: honest-grouping re-lock. Retrain A2.5 (WavLM-Large) under pooled_k420
# speakers, then K=2 (A2.5 + G4_gi + G5_mod) beta lock. Same recipe as the k210
# A2.5 heads (T=50 honesty prior, proj 128, dropout 0.5, AdamW 1e-3 + cosine,
# 25 ep patience 6, best on devel_val). Heads saved to a POOLED path so the
# committed k210 heads are untouched. Output: results/A5f_pooled_relock.json
# + cache/microsoft_wavlm-large/head_A2grouped_honestprior_POOLED_seed{seed}.pt

import json, statistics as st, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from features import LayerWeightedPooledHead, extract_g4, extract_g5
from features.train import _pooled_collate, predict_probs, evaluate, make_balanced_sampler
from honesty import (cold_probe, speaker_probe, fit_cold_probe, predict_logit,
                     fit_zscore, fuse, sweep_tau, evaluate_at_tau)
from speakers.cluster import load_pseudo_speakers

DATA_DIR="../dataset/ComParE2017_Cold_4students"; WAV_DIR=f"{DATA_DIR}/wav"
CACHE_ROOT="../cache"; BACKBONE="microsoft_wavlm-large"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED=42; ALL_SEEDS=[42,123,7,999,31337]
PROBE_TSV=Path(f"{CACHE_ROOT}/pseudo_speakers/pooled_k420_seed42.tsv")   # HONEST grouping
CKPT_FMT=f"{CACHE_ROOT}/{BACKBONE}/head_A2grouped_honestprior_POOLED_seed{{seed}}.pt"
AUDIT_OUT="../results/A5d_pooled_layer_honesty.csv"
OUT_JSON="../results/A5f_pooled_relock.json"
T_INV=50.0; N_LAYERS=25; STAT_DIM=4096
BETA_GRID=[0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0]
REF_K210_K2=0.7037   # shipped-grouping headline, for the delta

t0=time.time(); print(f"[device] {DEVICE}  [grouping] pooled_k420 (speaker-honest)")
assert PROBE_TSV.exists(), f"missing {PROBE_TSV}; run build_pooled_pseudo_speakers.py first"

# ---- splits under the pooled grouping ----
full_train=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="train")
full_devel=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="devel")
labels_map=load_labels(DATA_DIR); pseudo=load_pseudo_speakers(PROBE_TSV)
tf,tt=stratified_grouped_split(full_train.files,labels_map,pseudo,val_frac=0.10,seed=SPLIT_SEED)
dv,dtst=stratified_grouped_split(full_devel.files,labels_map,pseudo,val_frac=0.50,seed=SPLIT_SEED)
def _stems(fs): return [f[:-4] if f.endswith(".wav") else f for f in fs]
SPL={"train_fit":(tf,_stems(tf)),"train_threshold":(tt,_stems(tt)),
     "devel_val":(dv,_stems(dv)),"devel_test":(dtst,_stems(dtst))}
y={n:np.array([labels_map[f] for f in fs],dtype=np.int64) for n,(fs,_) in SPL.items()}
spk={n:np.array([pseudo[s] for s in sm],dtype=np.int64) for n,(_,sm) in SPL.items()}
print(f"[splits] " + "  ".join(f"{n}={len(fs)}" for n,(fs,_) in SPL.items()))

# ---- materialise train_fit + devel_val (fp16) for audit + training ----
def _mat(files):
    ds=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,file_list=files)
    return torch.stack([ds[i]["pooled"] for i in range(len(ds))],0).to(torch.float16)
print("[load] materialising train_fit + devel_val pooled ...")
X_tf=_mat(tf); X_dv=_mat(dv)
print(f"  train_fit {tuple(X_tf.shape)}  devel_val {tuple(X_dv.shape)}")

# ---- STEP 1: per-layer honesty audit -> sub@1 ----
print("\n=== STEP 1: per-layer honesty audit (pooled grouping) ===")
rows=[]
for L in range(N_LAYERS):
    Xa=X_tf[:,L,:].float().numpy(); Xe=X_dv[:,L,:].float().numpy()
    cr=cold_probe(Xa,y["train_fit"],Xe,y["devel_val"],seed=SPLIT_SEED)
    sr=speaker_probe(Xa,spk["train_fit"],Xe,spk["devel_val"],seed=SPLIT_SEED)
    lg=cr.uar-0.5; sg=sr.top1-1.0/max(sr.n_classes,1); s1=lg-sg
    rows.append({"layer":L,"cold_uar":float(cr.uar),"speaker_top1":float(sr.top1),
                 "n_pseudo":int(sr.n_classes),"label_gain":float(lg),"speaker_gain":float(sg),"sub_at_1":float(s1)})
    print(f"  L{L:02d} cold={cr.uar:.4f} spk={sr.top1:.4f} sub@1={s1:+.4f}")
sub1=np.array([r["sub_at_1"] for r in rows],dtype=np.float64)
import csv
Path(AUDIT_OUT).parent.mkdir(parents=True,exist_ok=True)
with open(AUDIT_OUT,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(f"  [wrote] {AUDIT_OUT}")

# ---- STEP 2: train 5 A2.5 heads under pooled grouping ----
print("\n=== STEP 2: retrain A2.5 heads (5 seeds, pooled) ===")
class _Mem(Dataset):
    def __init__(s,X,files,lab): s.X,s.files,s.lab=X,files,lab
    def __len__(s): return len(s.files)
    def __getitem__(s,i): return {"pooled":s.X[i].float(),"label":torch.tensor(int(s.lab[i])),"file_name":s.files[i]}
    def get_labels(s): return [int(v) for v in s.lab]
ds_tf=_Mem(X_tf,tf,y["train_fit"]); ds_dv=_Mem(X_dv,dv,y["devel_val"])
head_results=[]
for seed in ALL_SEEDS:
    print(f"\n  --- seed {seed} ---"); torch.manual_seed(seed); np.random.seed(seed)
    head=LayerWeightedPooledHead(n_layers=N_LAYERS,stat_dim=STAT_DIM,proj_dim=128,n_classes=2,dropout=0.5).to(DEVICE)
    with torch.no_grad(): head.layer_weights.copy_(torch.from_numpy(T_INV*sub1).float())
    head.scaler.to(DEVICE); head.scaler.fit(DataLoader(ds_tf,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),verbose=False)
    tl=DataLoader(ds_tf,batch_size=64,sampler=make_balanced_sampler(ds_tf,seed=seed),num_workers=0,collate_fn=_pooled_collate)
    vl=DataLoader(ds_dv,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate)
    lossf=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(head.param_groups(base_lr=1e-3),weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25)
    bvu,bep,bst,pat=-1.0,-1,None,0
    for ep in range(1,26):
        head.train()
        for b in tl:
            lo,_=head(b["pooled"].to(DEVICE)); loss=lossf(lo,b["label"].to(DEVICE))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.0); opt.step()
        sch.step(); vu,_,_,_,_=evaluate(head,vl,DEVICE)
        if vu>bvu: bvu,bep=vu,ep; bst={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=6: break
    head.load_state_dict(bst); print(f"    best_val_UAR={bvu:.4f} at ep{bep}")
    cp=CKPT_FMT.format(seed=seed); Path(cp).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":bst,"val_uar":bvu,"epoch":bep,"n_layers":N_LAYERS,"stat_dim":STAT_DIM,
                "proj_dim":128,"honesty_prior_T":T_INV,"sub_at_1":sub1.tolist(),"grouping":"pooled_k420_seed42"},cp)
    head_results.append({"seed":seed,"best_val_uar":float(bvu),"best_epoch":int(bep)})
    print(f"    [wrote ckpt] {cp}")

# ---- STEP 3: A2.5 logits on all 4 splits (disk-backed) + K=2 relock ----
print("\n=== STEP 3: K=2 re-lock under pooled ===")
def _a2_disk(head,files):
    ds=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,file_list=files)
    p,_=predict_probs(head,DataLoader(ds,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),DEVICE)
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def _load(seed):
    h=LayerWeightedPooledHead(n_layers=N_LAYERS,stat_dim=STAT_DIM,proj_dim=128,n_classes=2,dropout=0.5).to(DEVICE)
    h.load_state_dict(torch.load(CKPT_FMT.format(seed=seed),weights_only=True,map_location=DEVICE)["state_dict"]); h.eval(); return h
Xg4={n:extract_g4(sm,CACHE_ROOT,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
Xg5={n:extract_g5(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0,"n":len(xs)}
alone_uars=[]; k2_runs=[]
for seed in ALL_SEEDS:
    h=_load(seed); a2={n:_a2_disk(h,fs) for n,(fs,_) in SPL.items()}; del h
    ta,_=sweep_tau(a2["train_threshold"],y["train_threshold"]); alone_uars.append(evaluate_at_tau(a2["devel_test"],y["devel_test"],ta)["uar"])
    c4,s4=fit_cold_probe(Xg4["train_fit"],y["train_fit"],seed=seed); l4={n:predict_logit(c4,s4,Xg4[n]) for n in SPL}; z4=fit_zscore(l4["train_fit"]); Z4={n:z4.apply(l4[n]) for n in SPL}
    c5,s5=fit_cold_probe(Xg5["train_fit"],y["train_fit"],seed=seed); l5={n:predict_logit(c5,s5,Xg5[n]) for n in SPL}; z5=fit_zscore(l5["train_fit"]); Z5={n:z5.apply(l5[n]) for n in SPL}
    rows_b=[]
    for beta in BETA_GRID:
        tau,uthr=sweep_tau(fuse(a2["train_threshold"],[Z4["train_threshold"],Z5["train_threshold"]],beta),y["train_threshold"])
        dt=evaluate_at_tau(fuse(a2["devel_test"],[Z4["devel_test"],Z5["devel_test"]],beta),y["devel_test"],tau)
        rows_b.append({"beta":float(beta),"tau":float(tau),"uar_train_threshold":float(uthr),"devel_test":dt})
    locked=max(rows_b,key=lambda r:r["uar_train_threshold"])
    k2_runs.append({"seed":seed,"locked":locked,"sweep":rows_b})
    print(f"  seed {seed}: A2.5-alone={alone_uars[-1]:.4f}  K2 locked beta*={locked['beta']:.2f} devel_test={locked['devel_test']['uar']:.4f}")
alone=_ms(alone_uars); k2=_ms([r["locked"]["devel_test"]["uar"] for r in k2_runs]); betas=[r["locked"]["beta"] for r in k2_runs]
print(f"\n=== HONEST HEADLINE (pooled grouping) ===")
print(f"  A2.5-alone 5-seed: {alone['mean']:.4f} +/- {alone['std']:.4f}")
print(f"  K=2 5-seed:        {k2['mean']:.4f} +/- {k2['std']:.4f}   betas={betas}")
print(f"  vs k210 headline 0.7037:  delta={k2['mean']-REF_K210_K2:+.4f}")
print(f"  vs hidden test 0.62")

out={"rung_id":"A5f_pooled_relock",
     "description":"Honest-grouping re-lock: A2.5 heads retrained under pooled_k420 "
        "(speaker-honest) + K=2 relock. Same recipe as the k210 A2.5 heads; heads saved "
        "to a POOLED ckpt path. Removes the fixed-head early-stopping confound in "
        "audit_grouping_decomposition. This is the honest headline vs the k210 0.7037.",
     "grouping":"pooled_k420_seed42","split_seed":SPLIT_SEED,"all_seeds":ALL_SEEDS,
     "split_sizes":{n:len(SPL[n][0]) for n in SPL},"honesty_prior_T":T_INV,"beta_grid":BETA_GRID,
     "layer_audit":rows,"head_training":head_results,
     "a2_alone_5seed":alone,"k2_5seed":k2,"k2_locked_betas":betas,"k2_runs":k2_runs,
     "ref_k210_k2":REF_K210_K2,"delta_vs_k210":k2["mean"]-REF_K210_K2,
     "elapsed_minutes":(time.time()-t0)/60.0}
Path(OUT_JSON).write_text(json.dumps(out,indent=2))
print(f"\n[wrote] {OUT_JSON}  [done] {(time.time()-t0)/60:.1f} min")
'''

def main():
    nb=json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"]=="code" and '"A5f_pooled_relock"' in "".join(c["source"]):
            print("[skip] A5f already present."); return
    ast.parse(CODE)
    def mk(t,s):
        c={"cell_type":t,"metadata":{},"source":s.splitlines(keepends=True),"id":uuid.uuid4().hex[:8]}
        if t=="code": c["execution_count"]=None; c["outputs"]=[]
        return c
    nb["cells"].append(mk("markdown",MARKDOWN)); nb["cells"].append(mk("code",CODE))
    NB.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
    print(f"[appended] {RUNG}; notebook now has {len(nb['cells'])} cells.")

if __name__=="__main__": main()
