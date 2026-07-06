"""Append A5b_k3_cqt_pooled: re-check the CQT (G9) candidate under the pooled
(speaker-honest) heads + grouping. Depends on A5f having trained the pooled
heads. Idempotent; does NOT run. From model/:  python append_cqt_pooled_cell.py
"""
import ast, json, uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5b_k3_cqt_pooled"

MARKDOWN = "## A5b §4.14 — CQT (G9) re-checked under the pooled (honest) grouping\n\n" \
    "CQT admitted at 0.7253 under k210, but the k210 grouping was ~0.06 optimistic\n" \
    "(A5f). This re-runs the exact CQT config A/B sweep + speaker gate using the\n" \
    "pooled-retrained A2.5 heads and the pooled grouping, so we see whether the CQT\n" \
    "gain survives an honest evaluation or was grouping luck. RUN A5f FIRST (needs the\n" \
    "pooled heads). CQT cache is grouping-independent, already extracted. You run this."

CODE = r'''# A5b_k3_cqt_pooled: CQT (G9) under pooled heads + pooled grouping. Same config
# A/B sweep + two-dim honesty gate as the k210 CQT cell, so the numbers are directly
# comparable. Needs the pooled heads from A5f (head_..._POOLED_seed*.pt) and the CQT
# cache (grouping-independent, already built). Output: results/A5b_k3_cqt_pooled.json

import json, statistics as st, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from features import LayerWeightedPooledHead, extract_g4, extract_g5, extract_g9
from features.train import _pooled_collate, predict_probs
from honesty import (cold_probe, speaker_probe, fit_cold_probe, predict_logit,
                     fit_zscore, fuse, sweep_tau, evaluate_at_tau)
from speakers.cluster import load_pseudo_speakers

DATA_DIR="../dataset/ComParE2017_Cold_4students"; WAV_DIR=f"{DATA_DIR}/wav"
CACHE_ROOT="../cache"; BACKBONE="microsoft_wavlm-large"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED=42; ALL_SEEDS=[42,123,7,999,31337]
PROBE_TSV=Path(f"{CACHE_ROOT}/pseudo_speakers/pooled_k420_seed42.tsv")
CKPT_FMT=f"{CACHE_ROOT}/{BACKBONE}/head_A2grouped_honestprior_POOLED_seed{{seed}}.pt"
OUT_JSON="../results/A5b_k3_cqt_pooled.json"
BETA_GRID=[0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0]
REF_A2_SPK_TOP1=0.0501; SPK_CHANCE=1.0/420.0   # pooled k=420 on train+devel

t0=time.time(); print(f"[device] {DEVICE}  [grouping] pooled_k420")
for seed in ALL_SEEDS:
    assert Path(CKPT_FMT.format(seed=seed)).exists(), \
        f"missing pooled head for seed {seed}; RUN A5f_pooled_relock first."

# pooled K=2 reference (admission bar) from A5f if present
A5F=Path("../results/A5f_pooled_relock.json")
REF_K2 = json.loads(A5F.read_text())["k2_5seed"]["mean"] if A5F.exists() else 0.64
K3_BAR = REF_K2 + 0.005
print(f"[pooled K=2 reference] {REF_K2:.4f}  -> K=3 admission > {K3_BAR:.4f}")

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

Xg4={n:extract_g4(sm,CACHE_ROOT,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
Xg5={n:extract_g5(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
Xg9={n:extract_g9(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
print(f"[dims] G4={Xg4['train_fit'].shape[1]} G5={Xg5['train_fit'].shape[1]} G9={Xg9['train_fit'].shape[1]}")

# G9 standalone honesty audit under the pooled grouping
g9c=cold_probe(Xg9["train_fit"],y["train_fit"],Xg9["devel_val"],y["devel_val"])
g9s=speaker_probe(Xg9["train_fit"],spk["train_fit"],Xg9["devel_val"],spk["devel_val"])
spk_gate=g9s.top1<=REF_A2_SPK_TOP1
print(f"[G9 audit] cold UAR={g9c.uar:.4f}  speaker top-1={g9s.top1:.4f} (chance {SPK_CHANCE:.4f})  gate={'PASS' if spk_gate else 'FAIL'}")

def _a2(head,files):
    ds=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,file_list=files)
    p,_=predict_probs(head,DataLoader(ds,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),DEVICE)
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def _load(seed):
    h=LayerWeightedPooledHead(n_layers=full_train[0]["pooled"].shape[0],stat_dim=full_train[0]["pooled"].shape[1],proj_dim=128,n_classes=2,dropout=0.5).to(DEVICE)
    h.load_state_dict(torch.load(CKPT_FMT.format(seed=seed),weights_only=True,map_location=DEVICE)["state_dict"]); h.eval(); return h
def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0,"n":len(xs)}
per={}
for seed in ALL_SEEDS:
    h=_load(seed); a2={n:_a2(h,fs) for n,(fs,_) in SPL.items()}; del h
    def zg(X):
        c,s=fit_cold_probe(X["train_fit"],y["train_fit"],seed=seed); lg={n:predict_logit(c,s,X[n]) for n in SPL}
        z=fit_zscore(lg["train_fit"]); return {n:z.apply(lg[n]) for n in SPL}
    per[seed]={"a2":a2,"z4":zg(Xg4),"z5":zg(Xg5),"z9":zg(Xg9)}
    print(f"  seed {seed} done")
def sweep(keys):
    runs=[]
    for seed in ALL_SEEDS:
        c=per[seed]; rows=[]
        for beta in BETA_GRID:
            zt=[c[k]["train_threshold"] for k in keys]; zd=[c[k]["devel_test"] for k in keys]
            tau,uthr=sweep_tau(fuse(c["a2"]["train_threshold"],zt,beta),y["train_threshold"])
            dt=evaluate_at_tau(fuse(c["a2"]["devel_test"],zd,beta),y["devel_test"],tau)
            rows.append({"beta":float(beta),"tau":float(tau),"uar_train_threshold":float(uthr),"devel_test":dt})
        runs.append({"seed":seed,"locked":max(rows,key=lambda r:r["uar_train_threshold"]),"sweep":rows})
    return runs
A=sweep(["z4","z9"]); B=sweep(["z4","z5","z9"])
Aa=_ms([r["locked"]["devel_test"]["uar"] for r in A]); Ba=_ms([r["locked"]["devel_test"]["uar"] for r in B])
Ab=[r["locked"]["beta"] for r in A]; Bb=[r["locked"]["beta"] for r in B]
print(f"\n=== CQT under pooled grouping ===")
print(f"  pooled K=2 ref: {REF_K2:.4f}  (k210 CQT was: config A 0.7253, config B 0.7232)")
print(f"  Config A (K=2 A2+G4+G9): {Aa['mean']:.4f} +/- {Aa['std']:.4f}  betas={Ab}")
print(f"  Config B (K=3 A2+G4+G5+G9): {Ba['mean']:.4f} +/- {Ba['std']:.4f}  betas={Bb}")
admit=Ba["mean"]>K3_BAR and spk_gate
decision=("cqt_survives_honest_grouping" if admit else
          ("cqt_uar_ok_gate_fail" if Ba["mean"]>K3_BAR and not spk_gate else "cqt_gain_was_grouping_specific"))
print(f"\n=== VERDICT: {decision} (gate {'PASS' if spk_gate else 'FAIL'}, admit>{K3_BAR:.4f}: {'YES' if Ba['mean']>K3_BAR else 'NO'}) ===")

out={"rung_id":"A5b_k3_cqt_pooled",
     "description":"CQT (G9) K=3 candidate re-checked under the pooled (speaker-honest) "
        "A2.5 heads + grouping. Same config A/B sweep + speaker gate as the k210 CQT cell. "
        "Tests whether the k210 CQT gain (config A 0.7253) survives an honest evaluation.",
     "grouping":"pooled_k420_seed42","all_seeds":ALL_SEEDS,"beta_grid":BETA_GRID,
     "pooled_k2_reference":REF_K2,"k3_admission_threshold":K3_BAR,
     "k210_cqt_reference":{"config_a":0.7253,"config_b":0.7232},
     "g9_standalone_honesty":{"cold_uar_devel_val":float(g9c.uar),"speaker_top1_devel_val":float(g9s.top1),
        "speaker_nmi_devel_val":float(g9s.nmi),"speaker_gate_pass":bool(spk_gate)},
     "config_a_k2_replacement":{"runs":A,"locked_betas":Ab,"uar_devel_test":Aa},
     "config_b_k3_addition":{"runs":B,"locked_betas":Bb,"uar_devel_test":Ba,"admit":bool(admit)},
     "decision":decision,"elapsed_minutes":(time.time()-t0)/60.0}
Path(OUT_JSON).write_text(json.dumps(out,indent=2))
print(f"\n[wrote] {OUT_JSON}  [done] {(time.time()-t0)/60:.1f} min")
'''

def main():
    nb=json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"]=="code" and '"A5b_k3_cqt_pooled"' in "".join(c["source"]):
            print("[skip] cqt_pooled already present."); return
    ast.parse(CODE)
    def mk(t,s):
        c={"cell_type":t,"metadata":{},"source":s.splitlines(keepends=True),"id":uuid.uuid4().hex[:8]}
        if t=="code": c["execution_count"]=None; c["outputs"]=[]
        return c
    nb["cells"].append(mk("markdown",MARKDOWN)); nb["cells"].append(mk("code",CODE))
    NB.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
    print(f"[appended] {RUNG}; notebook now has {len(nb['cells'])} cells.")

if __name__=="__main__": main()
