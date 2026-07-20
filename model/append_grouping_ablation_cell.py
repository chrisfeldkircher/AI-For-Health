"""Append A5j: decompose the k210 -> pooled_k420 change into its TWO bundled
variables, {train-only, pooled} x {k=210, k=420}, and measure the WavLM K=2
honest number under each -- plus handcrafted-only under each (does backbone-null
survive every protocol?) and an evaluation-independence check (pooled refit
WITHOUT devel_test). Motivated by the reviewer point that the honest re-lock
changed pooling AND k at once and attributed the whole -0.10 to pooling.
Idempotent; does NOT run. From model/:  python append_grouping_ablation_cell.py
"""
import ast, json, uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5j_grouping_ablation"

MARKDOWN = "## A5j §6.2 — grouping ablation: is the honest deflation from POOLING or from k?\n\n" \
    "The honest re-lock (A5f) changed two variables at once: train-only->pooled clustering AND\n" \
    "k=210->k=420. The dossier attributed the whole -0.10 (0.7037->0.5992) to pooling. This\n" \
    "separates them: it rebuilds all four `{train-only, pooled} x {210, 420}` pseudo-speaker\n" \
    "labelings with an IDENTICAL recipe (L2 + KMeans n_init=10 seed=42), and reruns the exact\n" \
    "A5f WavLM K=2 system (A2.5 head + G4 + G5, heads early-stopped on that grouping's devel_val)\n" \
    "under each. It also reports handcrafted-only (G4+G5) under each grouping — so we see whether\n" \
    "'the backbone adds nothing' holds across ALL protocols, not just the one we picked. Finally\n" \
    "an evaluation-independence check: refit the pooled k=420 clustering WITHOUT devel_test\n" \
    "(assign it by nearest centroid) and confirm the K=2 number does not move — i.e. the pooled\n" \
    "fit isn't 'peeking' at the eval set. Reports a 2x2 table + the pooling/k decomposition.\n\n" \
    "First run builds 4 KMeans labelings (~15 min CPU) then trains WavLM heads under each\n" \
    "(~30 min GPU); re-runs skip clustering. You run this cell; I do not."

CODE = r'''# A5j: {train-only, pooled} x {k=210, k=420} ablation of the WavLM K=2 honest number.
# Identical clustering recipe across all 4 (L2 + KMeans n_init=10 seed=42), identical
# A5f head recipe (T=50 honesty prior, proj128, dropout .5, AdamW 1e-3 cosine, 25ep pat6).
# Output: results/A5j_grouping_ablation.json + cache/pseudo_speakers/ablation_*.tsv

import json, statistics as st, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from features import LayerWeightedPooledHead, extract_g4, extract_g5
from features.train import _pooled_collate, predict_probs, evaluate, make_balanced_sampler
from honesty import (cold_probe, speaker_probe, fit_cold_probe, predict_logit,
                     fit_zscore, fuse, sweep_tau, evaluate_at_tau)

DATA_DIR="../dataset/ComParE2017_Cold_4students"; WAV_DIR=f"{DATA_DIR}/wav"
CACHE_ROOT="../cache"; BACKBONE="microsoft_wavlm-large"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED=42; ALL_SEEDS=[42,123,7,999,31337]; SEED=42
NPZ=Path(f"{CACHE_ROOT}/ecapa-voxceleb/ecapa_embeddings.npz")
PS_DIR=Path(f"{CACHE_ROOT}/pseudo_speakers")
T_INV=50.0; N_LAYERS=25; STAT_DIM=4096
BETA_GRID=[0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0]
OUT_JSON="../results/A5j_grouping_ablation.json"
REF_K210_K2=0.7037

t0=time.time(); print(f"[device] {DEVICE}  ablation: {{train-only,pooled}} x {{210,420}}")
assert NPZ.exists(), f"missing {NPZ}"
labels_map=load_labels(DATA_DIR)

# ---- ECAPA embeddings for clustering ----
d=np.load(NPZ,allow_pickle=True); stems=d["stems"].astype(str); split=d["split"].astype(str); emb=d["embeddings"].astype(np.float32)
Xall=normalize(emb,axis=1); idx={s:i for i,s in enumerate(stems)}
tr_mask=split=="train"; dv_mask=split=="devel"
Xtr=Xall[tr_mask]; Xdv=Xall[dv_mask]; tr_stems=list(stems[tr_mask]); dv_stems=list(stems[dv_mask])

def _write_tsv(path,pseudo):
    with open(path,"w",encoding="utf-8",newline="\n") as f:
        f.write("file_stem\tsplit\tcluster\n")
        for sp in ("train","devel"):
            for s in stems[split==sp]: f.write(f"{s}\t{sp}\t{pseudo[s]}\n")

def _read_tsv(path):
    ps={}
    for ln in Path(path).read_text().splitlines()[1:]:
        s,_,c=ln.split("\t"); ps[s]=int(c)
    return ps

def build_grouping(mode,k,exclude=None):
    """mode in {train_only,pooled}. exclude: set of devel stems held OUT of the fit."""
    if mode=="train_only":
        km=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit(Xtr)
        ps={s:int(km.labels_[i]) for i,s in enumerate(tr_stems)}
        lab=km.predict(Xdv)
        for i,s in enumerate(dv_stems): ps[s]=int(lab[i])
    else:
        if exclude:
            keep=[s for s in dv_stems if s not in exclude]
            Xfit=np.vstack([Xtr]+[Xall[idx[s]][None] for s in keep]); fit_stems=tr_stems+keep
            km=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit(Xfit)
            ps={s:int(km.labels_[i]) for i,s in enumerate(fit_stems)}
            held=[s for s in dv_stems if s in exclude]
            lab=km.predict(normalize(np.vstack([emb[idx[s]] for s in held]),axis=1))
            for i,s in enumerate(held): ps[s]=int(lab[i])
        else:
            Xp=np.vstack([Xtr,Xdv]); ps_st=tr_stems+dv_stems
            km=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit(Xp)
            ps={s:int(km.labels_[i]) for i,s in enumerate(ps_st)}
    return ps

def get_grouping(tag,mode,k,exclude=None):
    path=PS_DIR/f"ablation_{tag}_seed42.tsv"
    if path.exists(): return _read_tsv(path)
    print(f"  [cluster] building {tag} ({mode}, k={k}{', excl devel_test' if exclude else ''}) ...")
    ps=build_grouping(mode,k,exclude); _write_tsv(path,ps); return ps

# ---- devel split-leakage (V8) diagnostic under a grouping (fast) ----
def devel_leak(pseudo):
    files=sorted(f for f in labels_map if f.startswith("devel_"))
    a,b=stratified_grouped_split(files,labels_map,pseudo,val_frac=0.50,seed=SPLIT_SEED)
    a_st=set(f[:-4] for f in a); order=[f[:-4] for f in files]
    X=normalize(np.vstack([emb[idx[s]] for s in order]),axis=1)
    side=np.array([0 if s in a_st else 1 for s in order],np.int8); n=X.shape[0]; nn_=np.empty(n,np.int8); B=2048
    for i0 in range(0,n,B):
        sim=X[i0:i0+B]@X.T
        for r in range(sim.shape[0]): sim[r,i0+r]=-2.0
        nn_[i0:i0+B]=side[sim.argmax(1)]
    return float((nn_!=side).mean())

# ---- the A5f WavLM K=2 system under a given grouping ----
class _Mem(Dataset):
    def __init__(s,X,files,lab): s.X,s.files,s.lab=X,files,lab
    def __len__(s): return len(s.files)
    def __getitem__(s,i): return {"pooled":s.X[i].float(),"label":torch.tensor(int(s.lab[i])),"file_name":s.files[i]}
    def get_labels(s): return [int(v) for v in s.lab]

def _mat(files):
    ds=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,file_list=files)
    return torch.stack([ds[i]["pooled"] for i in range(len(ds))],0).to(torch.float16)

def run_wavlm_k2(pseudo,tag):
    full_train=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="train")
    full_devel=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="devel")
    tf,tt=stratified_grouped_split(full_train.files,labels_map,pseudo,val_frac=0.10,seed=SPLIT_SEED)
    dv,dtst=stratified_grouped_split(full_devel.files,labels_map,pseudo,val_frac=0.50,seed=SPLIT_SEED)
    def _stems(fs): return [f[:-4] if f.endswith(".wav") else f for f in fs]
    SPL={"train_fit":(tf,_stems(tf)),"train_threshold":(tt,_stems(tt)),
         "devel_val":(dv,_stems(dv)),"devel_test":(dtst,_stems(dtst))}
    y={n:np.array([labels_map[f] for f in fs],dtype=np.int64) for n,(fs,_) in SPL.items()}
    spk={n:np.array([pseudo[s] for s in sm],dtype=np.int64) for n,(_,sm) in SPL.items()}
    X_tf=_mat(tf); X_dv=_mat(dv)
    # audit -> sub1
    sub1=np.empty(N_LAYERS,dtype=np.float64)
    for L in range(N_LAYERS):
        cr=cold_probe(X_tf[:,L,:].float().numpy(),y["train_fit"],X_dv[:,L,:].float().numpy(),y["devel_val"],seed=SPLIT_SEED)
        sr=speaker_probe(X_tf[:,L,:].float().numpy(),spk["train_fit"],X_dv[:,L,:].float().numpy(),spk["devel_val"],seed=SPLIT_SEED)
        sub1[L]=(cr.uar-0.5)-(sr.top1-1.0/max(sr.n_classes,1))
    ds_tf=_Mem(X_tf,tf,y["train_fit"]); ds_dv=_Mem(X_dv,dv,y["devel_val"])
    heads=[]
    for seed in ALL_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        head=LayerWeightedPooledHead(n_layers=N_LAYERS,stat_dim=STAT_DIM,proj_dim=128,n_classes=2,dropout=0.5).to(DEVICE)
        with torch.no_grad(): head.layer_weights.copy_(torch.from_numpy(T_INV*sub1).float())
        head.scaler.to(DEVICE); head.scaler.fit(DataLoader(ds_tf,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),verbose=False)
        tl=DataLoader(ds_tf,batch_size=64,sampler=make_balanced_sampler(ds_tf,seed=seed),num_workers=0,collate_fn=_pooled_collate)
        vl=DataLoader(ds_dv,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate)
        lossf=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(head.param_groups(base_lr=1e-3),weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25); bvu,bst,pat=-1.0,None,0
        for ep in range(1,26):
            head.train()
            for b in tl:
                lo,_=head(b["pooled"].to(DEVICE)); loss=lossf(lo,b["label"].to(DEVICE))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.0); opt.step()
            sch.step(); vu,_,_,_,_=evaluate(head,vl,DEVICE)
            if vu>bvu: bvu=vu; bst={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; pat=0
            else:
                pat+=1
                if pat>=6: break
        head.load_state_dict(bst); head.eval(); heads.append(head)
    # logits + fusion
    def _a2(head,files):
        ds=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,file_list=files)
        p,_=predict_probs(head,DataLoader(ds,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),DEVICE)
        p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
    Xg4={n:extract_g4(sm,CACHE_ROOT,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
    Xg5={n:extract_g5(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
    a2_alone=[]; k2=[]; hc=[]; betas=[]
    for seed,head in zip(ALL_SEEDS,heads):
        a2={n:_a2(head,fs) for n,(fs,_) in SPL.items()}
        ta,_=sweep_tau(a2["train_threshold"],y["train_threshold"]); a2_alone.append(evaluate_at_tau(a2["devel_test"],y["devel_test"],ta)["uar"])
        c4,s4=fit_cold_probe(Xg4["train_fit"],y["train_fit"],seed=seed); l4={n:predict_logit(c4,s4,Xg4[n]) for n in SPL}; z4=fit_zscore(l4["train_fit"]); Z4={n:z4.apply(l4[n]) for n in SPL}
        c5,s5=fit_cold_probe(Xg5["train_fit"],y["train_fit"],seed=seed); l5={n:predict_logit(c5,s5,Xg5[n]) for n in SPL}; z5=fit_zscore(l5["train_fit"]); Z5={n:z5.apply(l5[n]) for n in SPL}
        # WavLM K=2 (A5f): fuse(a2,[Z4,Z5])
        rb=[]
        for beta in BETA_GRID:
            tau,ut=sweep_tau(fuse(a2["train_threshold"],[Z4["train_threshold"],Z5["train_threshold"]],beta),y["train_threshold"])
            dt=evaluate_at_tau(fuse(a2["devel_test"],[Z4["devel_test"],Z5["devel_test"]],beta),y["devel_test"],tau)
            rb.append((ut,dt["uar"],beta))
        lk=max(rb,key=lambda r:r[0]); k2.append(lk[1]); betas.append(lk[2])
        # handcrafted-only G4+G5: fuse(l5,[Z4])
        rh=[]
        for beta in BETA_GRID:
            tau,ut=sweep_tau(fuse(l5["train_threshold"],[Z4["train_threshold"]],beta),y["train_threshold"])
            dt=evaluate_at_tau(fuse(l5["devel_test"],[Z4["devel_test"]],beta),y["devel_test"],tau)
            rh.append((ut,dt["uar"]))
        hc.append(max(rh,key=lambda r:r[0])[1])
    def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0}
    for h in heads: del h
    if DEVICE=="cuda": torch.cuda.empty_cache()
    return {"a2_alone":_ms(a2_alone),"k2":_ms(k2),"handcrafted_only":_ms(hc),"betas":betas,
            "split_sizes":{n:len(SPL[n][0]) for n in SPL},"devel_leak":devel_leak(pseudo)}

# ---- run the 2x2 ----
GROUPS=[("train_only_k210","train_only",210),("train_only_k420","train_only",420),
        ("pooled_k210","pooled",210),("pooled_k420","pooled",420)]
res={}
for tag,mode,k in GROUPS:
    print(f"\n=== grouping {tag} ===")
    ps=get_grouping(tag,mode,k)
    r=run_wavlm_k2(ps,tag); res[tag]=r
    print(f"  devel_leak={r['devel_leak']:.3f}  A2.5-alone={r['a2_alone']['mean']:.4f}  "
          f"WavLM K=2={r['k2']['mean']:.4f}+/-{r['k2']['std']:.4f}  handcrafted-only={r['handcrafted_only']['mean']:.4f}")

# ---- evaluation-independence: pooled k420 refit WITHOUT devel_test ----
print(f"\n=== evaluation-independence: pooled k420 refit WITHOUT devel_test ===")
full_devel=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="devel")
files=sorted(f for f in labels_map if f.startswith("devel_"))
_,dtst_full=stratified_grouped_split(files,labels_map,get_grouping("pooled_k420","pooled",420),val_frac=0.50,seed=SPLIT_SEED)
excl=set(f[:-4] for f in dtst_full)
ps_indep=get_grouping("pooled_k420_no_develtest","pooled",420,exclude=excl)
r_indep=run_wavlm_k2(ps_indep,"indep"); res["pooled_k420_no_develtest"]=r_indep
print(f"  WavLM K=2 (no-devel_test-in-fit)={r_indep['k2']['mean']:.4f}  vs full pooled_k420={res['pooled_k420']['k2']['mean']:.4f}  "
      f"delta={r_indep['k2']['mean']-res['pooled_k420']['k2']['mean']:+.4f}")

# ---- decomposition ----
def K2(tag): return res[tag]["k2"]["mean"]
pooling_effect=0.5*((K2("pooled_k210")-K2("train_only_k210"))+(K2("pooled_k420")-K2("train_only_k420")))
k_effect=0.5*((K2("train_only_k420")-K2("train_only_k210"))+(K2("pooled_k420")-K2("pooled_k210")))
print(f"\n=== 2x2 WavLM K=2 (devel_test UAR) ===")
print(f"                    k=210     k=420")
print(f"  train-only    {K2('train_only_k210'):.4f}   {K2('train_only_k420'):.4f}")
print(f"  pooled        {K2('pooled_k210'):.4f}   {K2('pooled_k420'):.4f}")
print(f"  main effect of POOLING (train->pooled): {pooling_effect:+.4f}")
print(f"  main effect of k (210->420):            {k_effect:+.4f}")
print(f"  shipped k210 headline ref: {REF_K210_K2}")
print(f"\n=== does backbone-null survive every protocol? (K=2 vs handcrafted-only) ===")
for tag,_,_ in GROUPS:
    d=res[tag]["k2"]["mean"]-res[tag]["handcrafted_only"]["mean"]
    print(f"  {tag:<18} K2={res[tag]['k2']['mean']:.4f}  hc-only={res[tag]['handcrafted_only']['mean']:.4f}  backbone adds {d:+.4f}")

out={"rung_id":"A5j_grouping_ablation",
     "description":"Decompose the k210->pooled_k420 honest re-lock into {train-only,pooled} x {210,420}. "
        "Identical clustering recipe; A5f WavLM K=2 (A2.5+G4+G5) under each grouping + handcrafted-only "
        "under each + evaluation-independence (pooled k420 refit without devel_test).",
     "all_seeds":ALL_SEEDS,"beta_grid":BETA_GRID,"ref_k210_k2":REF_K210_K2,
     "results":res,"pooling_effect":pooling_effect,"k_effect":k_effect,
     "independence_delta":r_indep["k2"]["mean"]-res["pooled_k420"]["k2"]["mean"],
     "elapsed_minutes":(time.time()-t0)/60.0}
Path(OUT_JSON).parent.mkdir(parents=True,exist_ok=True); Path(OUT_JSON).write_text(json.dumps(out,indent=2))
print(f"\n[wrote] {OUT_JSON}  [done] {(time.time()-t0)/60:.1f} min")
'''

def main():
    nb=json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"]=="code" and '"A5j_grouping_ablation"' in "".join(c["source"]):
            print("[skip] A5j already present."); return
    ast.parse(CODE)
    def mk(t,s):
        c={"cell_type":t,"metadata":{},"source":s.splitlines(keepends=True),"id":uuid.uuid4().hex[:8]}
        if t=="code": c["execution_count"]=None; c["outputs"]=[]
        return c
    nb["cells"].append(mk("markdown",MARKDOWN)); nb["cells"].append(mk("code",CODE))
    NB.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
    print(f"[appended] {RUNG}; notebook now has {len(nb['cells'])} cells.")

if __name__=="__main__": main()
