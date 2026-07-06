"""Append A5h: HuBERT-large (ll60k) under the honest (pooled) grouping, tested as
a WavLM replacement + fused with the honest handcrafted G4+G9. Motivated by the
other team's HuBERT reaching 0.67 on the hidden test vs our 0.63. Idempotent;
does NOT run. From model/:  python append_hubertlarge_cell.py
"""
import ast, json, uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5h_hubertlarge_pooled"

MARKDOWN = "## A5h §7.2 — HuBERT-large (ll60k) under the honest grouping (replace WavLM?)\n\n" \
    "The other team's HuBERT hit 0.67 on the hidden test vs our 0.63 (same held-out set,\n" \
    "the axis we're weakest on). We tested HuBERT-BASE and it failed; HuBERT-LARGE is\n" \
    "untested and the likely explanation. This extracts HuBERT-large, trains an A2.5 head\n" \
    "under the pooled (speaker-honest) grouping, and evaluates it standalone AND fused with\n" \
    "our honest G4+G9, against the real bars: WavLM head 0.58, handcrafted G4+G9 0.67(split42)\n" \
    "/0.62(shadow), ECAPA speaker baseline 0.59. First run extracts HuBERT-large pooled for\n" \
    "train+devel (GPU, ~1 hr). You run this cell; I do not."

CODE = r'''# A5h: HuBERT-large (facebook/hubert-large-ll60k) A2.5 head under the pooled
# (speaker-honest) grouping, as a WavLM replacement + fused with honest G4+G9.
# Same recipe/protocol as A5f (honesty-prior init T=50, proj 128, dropout 0.5,
# AdamW 1e-3 cosine, 25 ep patience 6, best on devel_val). Reports standalone
# HuBERT-large-A2.5, K=2 (A2.5+G4+G9), and the speaker gate, vs the honest bars.
# Output: results/A5h_hubertlarge_pooled.json + results/A5d_hubertlarge_layer_honesty.csv
# Cost: first run extracts HuBERT-large pooled for train+devel (GPU, ~1 hr),
# then ~15 min. Re-runs skip extraction.

import json, statistics as st, time, csv
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.data import AudioDataset
from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from features import LayerWeightedPooledHead, extract_g4, extract_g9
from features.backbone import build_backbone
from features.extract import extract_pooled
from features.train import _pooled_collate, predict_probs, evaluate, make_balanced_sampler
from honesty import (cold_probe, speaker_probe, fit_cold_probe, predict_logit,
                     fit_zscore, fuse, sweep_tau, evaluate_at_tau)
from speakers.cluster import load_pseudo_speakers

DATA_DIR="../dataset/ComParE2017_Cold_4students"; WAV_DIR=f"{DATA_DIR}/wav"
CACHE_ROOT="../cache"
HL_BACKBONE="facebook_hubert-large-ll60k"; HL_BUILD="hubert-large"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED=42; ALL_SEEDS=[42,123,7,999,31337]; CLIP_SECS=8.0; BATCH_EXTRACT=4
PROBE_TSV=Path(f"{CACHE_ROOT}/pseudo_speakers/pooled_k420_seed42.tsv")   # HONEST grouping
HL_CACHE=Path(f"{CACHE_ROOT}/{HL_BACKBONE}/pooled")
CKPT_FMT=f"{CACHE_ROOT}/{HL_BACKBONE}/head_A25_honestprior_POOLED_seed{{seed}}.pt"
AUDIT_OUT="../results/A5d_hubertlarge_layer_honesty.csv"
OUT_JSON="../results/A5h_hubertlarge_pooled.json"
T_INV=50.0
BETA_GRID=[0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0,24.0,32.0,48.0,64.0]
REF_A2_SPK_TOP1=0.0501; SPK_CHANCE=1.0/420.0
# honest bars (pooled grouping / shadow):
BAR={"wavlm_head":0.579,"handcrafted_G4G9_split42":0.674,"handcrafted_G4G9_shadow":0.62,"ecapa_speaker":0.594}

t0=time.time(); print(f"[device] {DEVICE}  [backbone] {HL_BACKBONE}  [grouping] pooled_k420 (honest)")
assert PROBE_TSV.exists(), f"missing {PROBE_TSV}; run build_pooled_pseudo_speakers.py first"
print(f"[honest bars] WavLM head {BAR['wavlm_head']}  handcrafted G4+G9 {BAR['handcrafted_G4G9_shadow']}(shadow)/"
      f"{BAR['handcrafted_G4G9_split42']}(split42)  ECAPA-speaker {BAR['ecapa_speaker']}")

# ---- splits (pooled grouping) ----
full_devel_wavlm=PooledCacheDataset(DATA_DIR,CACHE_ROOT,"microsoft_wavlm-large",split="devel")  # for file lists only
labels_map=load_labels(DATA_DIR); pseudo=load_pseudo_speakers(PROBE_TSV)
train_files=sorted(f for f in labels_map if f.startswith("train_"))
devel_files=sorted(f for f in labels_map if f.startswith("devel_"))
tf,tt=stratified_grouped_split(train_files,labels_map,pseudo,val_frac=0.10,seed=SPLIT_SEED)
dv,dtst=stratified_grouped_split(devel_files,labels_map,pseudo,val_frac=0.50,seed=SPLIT_SEED)
def _stems(fs): return [f[:-4] if f.endswith(".wav") else f for f in fs]
SPL={"train_fit":(tf,_stems(tf)),"train_threshold":(tt,_stems(tt)),
     "devel_val":(dv,_stems(dv)),"devel_test":(dtst,_stems(dtst))}
y={n:np.array([labels_map[f] for f in fs],dtype=np.int64) for n,(fs,_) in SPL.items()}
spk={n:np.array([pseudo[s] for s in sm],dtype=np.int64) for n,(_,sm) in SPL.items()}
print(f"[splits] " + "  ".join(f"{n}={len(fs)}" for n,(fs,_) in SPL.items()))

# ---- STEP 0.5: extract HuBERT-large pooled for train+devel (idempotent) ----
need=sorted({s for _,(_,sm) in SPL.items() for s in sm})
have=sum(1 for s in need if (HL_CACHE/f"{s}.pt").exists())
print(f"\n[step0.5] HuBERT-large pooled cache: {have}/{len(need)} present")
if have<len(need):
    print(f"  extracting HuBERT-large pooled (pad={CLIP_SECS}s, batch={BATCH_EXTRACT}) -- GPU, slow ...")
    backbone=build_backbone(HL_BUILD,device=DEVICE)
    for sp_ in ("train","devel"):
        ds_a=AudioDataset(data_dir=DATA_DIR,split=sp_,use_mel=False,use_opensmile=False,pad_or_truncate_secs=CLIP_SECS)
        extract_pooled(backbone=backbone,dataset=ds_a,cache_root=CACHE_ROOT,batch_size=BATCH_EXTRACT,skip_existing=True)
    del backbone
    if DEVICE=="cuda": torch.cuda.empty_cache()
have=sum(1 for s in need if (HL_CACHE/f"{s}.pt").exists())
assert have==len(need), f"HuBERT-large cache incomplete: {have}/{len(need)}"
print(f"  cache ready ({(time.time()-t0)/60:.1f} min)")

# detect layer/stat dims from a cached tensor
_samp=torch.load(HL_CACHE/f"{need[0]}.pt",weights_only=True,map_location="cpu")
N_LAYERS,STAT_DIM=_samp.shape
print(f"  HuBERT-large pooled shape: n_layers={N_LAYERS} stat_dim={STAT_DIM}")

def _mat(files):
    return torch.stack([torch.load(HL_CACHE/f"{s}.pt",weights_only=True,map_location="cpu")
                        for s in _stems(files)],0).to(torch.float16)
print("[load] materialising train_fit + devel_val ...")
X_tf=_mat(tf); X_dv=_mat(dv)

# ---- STEP 1: per-layer honesty audit -> sub@1 ----
print("\n=== STEP 1: per-layer honesty audit (HuBERT-large, pooled) ===")
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
best_cold=max(rows,key=lambda r:r["cold_uar"])
print(f"  best cold layer L{best_cold['layer']}: cold_uar={best_cold['cold_uar']:.4f} spk={best_cold['speaker_top1']:.4f}")
Path(AUDIT_OUT).parent.mkdir(parents=True,exist_ok=True)
with open(AUDIT_OUT,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(f"  [wrote] {AUDIT_OUT}")

# ---- STEP 2: train 5 HuBERT-large-A2.5 heads (pooled) ----
print("\n=== STEP 2: HuBERT-large-A2.5 head training (5 seeds, pooled) ===")
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
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25); bvu,bep,bst,pat=-1.0,-1,None,0
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
                "proj_dim":128,"honesty_prior_T":T_INV,"grouping":"pooled_k420_seed42","backbone":HL_BACKBONE},cp)
    head_results.append({"seed":seed,"best_val_uar":float(bvu),"best_epoch":int(bep)})

# ---- STEP 3: standalone + fusion with G4+G9 (disk-backed logits) ----
print("\n=== STEP 3: standalone + K=2 fusion (A2.5_HL + G4 + G9), pooled ===")
def _hl_logit(head,files):
    ds=PooledCacheDataset(DATA_DIR,CACHE_ROOT,HL_BACKBONE,file_list=files)
    p,_=predict_probs(head,DataLoader(ds,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),DEVICE)
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def _load(seed):
    h=LayerWeightedPooledHead(n_layers=N_LAYERS,stat_dim=STAT_DIM,proj_dim=128,n_classes=2,dropout=0.5).to(DEVICE)
    h.load_state_dict(torch.load(CKPT_FMT.format(seed=seed),weights_only=True,map_location=DEVICE)["state_dict"]); h.eval(); return h
Xg4={n:extract_g4(sm,CACHE_ROOT,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
Xg9={n:extract_g9(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0,"n":len(xs)}
alone=[]; k2_runs=[]
for seed in ALL_SEEDS:
    h=_load(seed); a2={n:_hl_logit(h,fs) for n,(fs,_) in SPL.items()}; del h
    ta,_=sweep_tau(a2["train_threshold"],y["train_threshold"]); alone.append(evaluate_at_tau(a2["devel_test"],y["devel_test"],ta)["uar"])
    c4,s4=fit_cold_probe(Xg4["train_fit"],y["train_fit"],seed=seed); l4={n:predict_logit(c4,s4,Xg4[n]) for n in SPL}; z4=fit_zscore(l4["train_fit"]); Z4={n:z4.apply(l4[n]) for n in SPL}
    c9,s9=fit_cold_probe(Xg9["train_fit"],y["train_fit"],seed=seed); l9={n:predict_logit(c9,s9,Xg9[n]) for n in SPL}; z9=fit_zscore(l9["train_fit"]); Z9={n:z9.apply(l9[n]) for n in SPL}
    rows_b=[]
    for beta in BETA_GRID:
        tau,uthr=sweep_tau(fuse(a2["train_threshold"],[Z4["train_threshold"],Z9["train_threshold"]],beta),y["train_threshold"])
        dt=evaluate_at_tau(fuse(a2["devel_test"],[Z4["devel_test"],Z9["devel_test"]],beta),y["devel_test"],tau)
        rows_b.append({"beta":float(beta),"tau":float(tau),"uar_train_threshold":float(uthr),"devel_test":dt})
    locked=max(rows_b,key=lambda r:r["uar_train_threshold"]); k2_runs.append({"seed":seed,"locked":locked,"sweep":rows_b})
    print(f"  seed {seed}: HL-A2.5 alone={alone[-1]:.4f}  +G4+G9 beta*={locked['beta']:.1f} devel_test={locked['devel_test']['uar']:.4f}")
# G9 speaker gate under pooled
g9s=speaker_probe(Xg9["train_fit"],spk["train_fit"],Xg9["devel_val"],spk["devel_val"])
hl_alone=_ms(alone); hl_k2=_ms([r["locked"]["devel_test"]["uar"] for r in k2_runs]); betas=[r["locked"]["beta"] for r in k2_runs]

print(f"\n=== HuBERT-large VERDICT (pooled, honest) ===")
print(f"  HuBERT-large-A2.5 standalone: {hl_alone['mean']:.4f} +/- {hl_alone['std']:.4f}   (WavLM head bar {BAR['wavlm_head']}, ECAPA {BAR['ecapa_speaker']})")
print(f"  HuBERT-large-A2.5 + G4 + G9:  {hl_k2['mean']:.4f} +/- {hl_k2['std']:.4f}   betas={betas}")
print(f"  handcrafted-only G4+G9 bar:   {BAR['handcrafted_G4G9_shadow']}(shadow)/{BAR['handcrafted_G4G9_split42']}(split42)")
print(f"  best cold layer spk_top1={best_cold['speaker_top1']:.4f} (gate ref {REF_A2_SPK_TOP1})")
beats_wavlm = hl_alone["mean"] > BAR["wavlm_head"]
beats_handcrafted = hl_k2["mean"] > BAR["handcrafted_G4G9_split42"]
print(f"  HL head beats WavLM head: {beats_wavlm}   HL+G4+G9 beats handcrafted-only(split42): {beats_handcrafted}")

out={"rung_id":"A5h_hubertlarge_pooled",
     "description":"HuBERT-large (ll60k) A2.5 head under the pooled (speaker-honest) grouping, "
        "as a WavLM replacement + fused with honest G4+G9. Same recipe as A5f. Motivated by "
        "the other team's HuBERT hidden-test 0.67 vs our 0.63.",
     "backbone":HL_BACKBONE,"grouping":"pooled_k420_seed42","split_seed":SPLIT_SEED,"all_seeds":ALL_SEEDS,
     "n_layers":int(N_LAYERS),"stat_dim":int(STAT_DIM),"honesty_prior_T":T_INV,"beta_grid":BETA_GRID,
     "honest_bars":BAR,"layer_audit":rows,"best_cold_layer":best_cold,"head_training":head_results,
     "hubertlarge_a25_standalone":hl_alone,"hubertlarge_plus_g4_g9":hl_k2,"locked_betas":betas,"k2_runs":k2_runs,
     "g9_speaker_gate":{"top1":float(g9s.top1),"reference":REF_A2_SPK_TOP1,"pass":bool(g9s.top1<=REF_A2_SPK_TOP1)},
     "beats_wavlm_head":bool(beats_wavlm),"beats_handcrafted_only_split42":bool(beats_handcrafted),
     "elapsed_minutes":(time.time()-t0)/60.0}
Path(OUT_JSON).write_text(json.dumps(out,indent=2))
print(f"\n[wrote] {OUT_JSON}  [done] {(time.time()-t0)/60:.1f} min")
'''

def main():
    nb=json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"]=="code" and '"A5h_hubertlarge_pooled"' in "".join(c["source"]):
            print("[skip] A5h already present."); return
    ast.parse(CODE)
    def mk(t,s):
        c={"cell_type":t,"metadata":{},"source":s.splitlines(keepends=True),"id":uuid.uuid4().hex[:8]}
        if t=="code": c["execution_count"]=None; c["outputs"]=[]
        return c
    nb["cells"].append(mk("markdown",MARKDOWN)); nb["cells"].append(mk("code",CODE))
    NB.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
    print(f"[appended] {RUNG}; notebook now has {len(nb['cells'])} cells.")

if __name__=="__main__": main()
