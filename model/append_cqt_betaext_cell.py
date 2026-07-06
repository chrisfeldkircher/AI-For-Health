"""Append A5g: pin down the CQT (G9) result under the honest grouping. Extends
the beta grid (config A pinned at edge 16), adds a handcrafted-only baseline
(is A2.5 still contributing?), and a 5-split-seed shadow. Needs A5f's pooled
heads. Idempotent; does NOT run. From model/:  python append_cqt_betaext_cell.py
"""
import ast, json, uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5g_cqt_beta_extension"

MARKDOWN = "## A5g §4.14.1 — pin down CQT: extended beta, A2.5-contribution, shadow\n\n" \
    "Under the honest grouping, config A (A2.5+G4+G9) hit 0.6718 but with all betas at\n" \
    "the grid edge (16), so the optimum is beyond the grid. This: (1) extends beta to\n" \
    "find the real peak, (2) compares against a HANDCRAFTED-ONLY baseline (G4+G9, no\n" \
    "A2.5) to check whether the frozen WavLM head still contributes or the win is purely\n" \
    "the handcrafted features, (3) shadows config A over 5 split seeds to confirm the\n" \
    "gain is stable, not one lucky partition. Needs A5f pooled heads. You run this."

CODE = r'''# A5g: extended-beta + handcrafted-only + shadow verification of CQT (G9) under
# the honest pooled grouping. Output: results/A5g_cqt_beta_extension.json

import json, statistics as st, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from features import LayerWeightedPooledHead, extract_g4, extract_g9
from features.train import _pooled_collate, predict_probs
from honesty import fit_cold_probe, predict_logit, fit_zscore, fuse, sweep_tau, evaluate_at_tau
from speakers.cluster import load_pseudo_speakers

DATA_DIR="../dataset/ComParE2017_Cold_4students"; WAV_DIR=f"{DATA_DIR}/wav"
CACHE_ROOT="../cache"; BACKBONE="microsoft_wavlm-large"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
ALL_SEEDS=[42,123,7,999,31337]
PROBE_TSV=Path(f"{CACHE_ROOT}/pseudo_speakers/pooled_k420_seed42.tsv")
CKPT_FMT=f"{CACHE_ROOT}/{BACKBONE}/head_A2grouped_honestprior_POOLED_seed{{seed}}.pt"
OUT_JSON="../results/A5g_cqt_beta_extension.json"
EXT_BETA=[0.0,0.5,1.0,2.0,4.0,8.0,12.0,16.0,24.0,32.0,48.0,64.0,96.0,128.0]
SHADOW_SPLIT_SEEDS=[42,1,2,3,5]

t0=time.time(); print(f"[device] {DEVICE}")
for seed in ALL_SEEDS:
    assert Path(CKPT_FMT.format(seed=seed)).exists(), f"missing pooled head seed {seed}; run A5f first."
labels_map=load_labels(DATA_DIR); pseudo=load_pseudo_speakers(PROBE_TSV)
full_train=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="train")
full_devel=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="devel")
NL,SD=full_train[0]["pooled"].shape
def _stems(fs): return [f[:-4] if f.endswith(".wav") else f for f in fs]
def _a2(head,files):
    ds=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,file_list=files)
    p,_=predict_probs(head,DataLoader(ds,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),DEVICE)
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def _load(seed):
    h=LayerWeightedPooledHead(n_layers=NL,stat_dim=SD,proj_dim=128,n_classes=2,dropout=0.5).to(DEVICE)
    h.load_state_dict(torch.load(CKPT_FMT.format(seed=seed),weights_only=True,map_location=DEVICE)["state_dict"]); h.eval(); return h
def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0,"n":len(xs)}

# precompute A2.5 logits per seed over all train+devel (split-independent)
print("[precompute] A2.5 logits per seed (all train+devel) ...")
A2={}
for seed in ALL_SEEDS:
    h=_load(seed); lt=_a2(h,full_train.files); ld=_a2(h,full_devel.files); del h
    A2[seed]={**{s:lt[i] for i,s in enumerate(_stems(full_train.files))},
              **{s:ld[i] for i,s in enumerate(_stems(full_devel.files))}}
    print(f"  seed {seed}")

def build(split_seed):
    tf,tt=stratified_grouped_split(full_train.files,labels_map,pseudo,val_frac=0.10,seed=split_seed)
    dv,dt=stratified_grouped_split(full_devel.files,labels_map,pseudo,val_frac=0.50,seed=split_seed)
    SPL={"train_threshold":(tt,_stems(tt)),"devel_test":(dt,_stems(dt)),"train_fit":(tf,_stems(tf))}
    y={n:np.array([labels_map[f] for f in fs]) for n,(fs,_) in SPL.items()}
    Xg4={n:extract_g4(sm,CACHE_ROOT,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
    Xg9={n:extract_g9(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
    return SPL,y,Xg4,Xg9

# ===== main analysis on split seed 42 =====
SPL,y,Xg4,Xg9=build(42)
peak_betas=[]; peak_uars=[]; hc_only=[]; a2_alone=[]; a2_contrib=[]
sweeps={}
for seed in ALL_SEEDS:
    a2={n:np.array([A2[seed][s] for s in sm]) for n,(_,sm) in SPL.items()}
    c4,s4=fit_cold_probe(Xg4["train_fit"],y["train_fit"],seed=seed); l4={n:predict_logit(c4,s4,Xg4[n]) for n in SPL}; z4v=fit_zscore(l4["train_fit"]); Z4={n:z4v.apply(l4[n]) for n in SPL}
    c9,s9=fit_cold_probe(Xg9["train_fit"],y["train_fit"],seed=seed); l9={n:predict_logit(c9,s9,Xg9[n]) for n in SPL}; z9v=fit_zscore(l9["train_fit"]); Z9={n:z9v.apply(l9[n]) for n in SPL}
    # extended beta sweep (config A = A2.5 + G4 + G9)
    rows=[]
    for b in EXT_BETA:
        tau,ut=sweep_tau(fuse(a2["train_threshold"],[Z4["train_threshold"],Z9["train_threshold"]],b),y["train_threshold"])
        dt=evaluate_at_tau(fuse(a2["devel_test"],[Z4["devel_test"],Z9["devel_test"]],b),y["devel_test"],tau)["uar"]
        rows.append((b,float(ut),float(dt)))
    lk=max(rows,key=lambda r:r[1]); peak_betas.append(lk[0]); peak_uars.append(lk[2]); sweeps[seed]=rows
    # handcrafted-only: mean(z4,z9), no A2.5
    hc={n:0.5*(Z4[n]+Z9[n]) for n in SPL}
    th,_=sweep_tau(hc["train_threshold"],y["train_threshold"]); hc_only.append(evaluate_at_tau(hc["devel_test"],y["devel_test"],th)["uar"])
    ta,_=sweep_tau(a2["train_threshold"],y["train_threshold"]); a2_alone.append(evaluate_at_tau(a2["devel_test"],y["devel_test"],ta)["uar"])
    a2_contrib.append(lk[2]-hc_only[-1])
    print(f"  seed {seed}: peak beta*={lk[0]:.0f} uar={lk[2]:.4f} | hc-only(G4+G9)={hc_only[-1]:.4f} | A2.5-alone={a2_alone[-1]:.4f} | A2.5 adds {a2_contrib[-1]:+.4f}")

pk=_ms(peak_uars); hc=_ms(hc_only); aa=_ms(a2_alone); contrib=_ms(a2_contrib)
print(f"\n=== config A (A2.5+G4+G9), extended beta ===")
print(f"  peak UAR:        {pk['mean']:.4f} +/- {pk['std']:.4f}  peak betas={peak_betas}")
print(f"  handcrafted-only:{hc['mean']:.4f} +/- {hc['std']:.4f}  (G4+G9, no A2.5)")
print(f"  A2.5-alone:      {aa['mean']:.4f} +/- {aa['std']:.4f}")
print(f"  A2.5 contribution over handcrafted-only: {contrib['mean']:+.4f} +/- {contrib['std']:.4f}")
still_edge=any(b>=EXT_BETA[-1] for b in peak_betas)
a2_dead = contrib['mean'] < 0.005
print(f"  peak still at grid edge? {still_edge}   A2.5 effectively dead? {a2_dead}")

# ===== shadow: config A extended-beta lock across split seeds =====
print(f"\n=== shadow (config A extended-beta) across split seeds {SHADOW_SPLIT_SEEDS} ===")
shadow={}
for ss in SHADOW_SPLIT_SEEDS:
    SPLs,ys,Xg4s,Xg9s=build(ss); us=[]
    for seed in ALL_SEEDS:
        a2={n:np.array([A2[seed][s] for s in sm]) for n,(_,sm) in SPLs.items()}
        c4,s4=fit_cold_probe(Xg4s["train_fit"],ys["train_fit"],seed=seed); l4={n:predict_logit(c4,s4,Xg4s[n]) for n in SPLs}; z4v=fit_zscore(l4["train_fit"]); Z4={n:z4v.apply(l4[n]) for n in SPLs}
        c9,s9=fit_cold_probe(Xg9s["train_fit"],ys["train_fit"],seed=seed); l9={n:predict_logit(c9,s9,Xg9s[n]) for n in SPLs}; z9v=fit_zscore(l9["train_fit"]); Z9={n:z9v.apply(l9[n]) for n in SPLs}
        best=None
        for b in EXT_BETA:
            tau,ut=sweep_tau(fuse(a2["train_threshold"],[Z4["train_threshold"],Z9["train_threshold"]],b),ys["train_threshold"])
            dt=evaluate_at_tau(fuse(a2["devel_test"],[Z4["devel_test"],Z9["devel_test"]],b),ys["devel_test"],tau)["uar"]
            if best is None or ut>best[0]: best=(ut,dt)
        us.append(best[1])
    shadow[ss]=_ms(us); print(f"  split_seed {ss}: {shadow[ss]['mean']:.4f} +/- {shadow[ss]['std']:.4f}")
sv=[shadow[ss]["mean"] for ss in SHADOW_SPLIT_SEEDS]
print(f"  shadow-mean over split seeds: {st.mean(sv):.4f} +/- {(st.stdev(sv) if len(sv)>1 else 0):.4f}")

out={"rung_id":"A5g_cqt_beta_extension",
     "description":"Pins down the honest-grouping CQT result: extended beta grid to find "
        "the real config-A peak (was pinned at edge 16), handcrafted-only (G4+G9, no A2.5) "
        "baseline to test whether the frozen WavLM head still contributes, and a 5-split-"
        "seed shadow for stability.",
     "grouping":"pooled_k420_seed42","all_seeds":ALL_SEEDS,"extended_beta":EXT_BETA,
     "config_a_peak":pk,"peak_betas":peak_betas,"handcrafted_only":hc,"a2_alone":aa,
     "a2_contribution_over_handcrafted":contrib,"peak_still_at_grid_edge":bool(still_edge),
     "a2_effectively_dead":bool(a2_dead),
     "shadow_split_seeds":{str(k):v for k,v in shadow.items()},
     "shadow_mean":float(st.mean(sv)),"shadow_std":float(st.stdev(sv) if len(sv)>1 else 0),
     "per_seed_sweeps":{str(s):sweeps[s] for s in ALL_SEEDS},
     "elapsed_minutes":(time.time()-t0)/60.0}
Path(OUT_JSON).write_text(json.dumps(out,indent=2))
print(f"\n[wrote] {OUT_JSON}  [done] {(time.time()-t0)/60:.1f} min")
'''

def main():
    nb=json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"]=="code" and '"A5g_cqt_beta_extension"' in "".join(c["source"]):
            print("[skip] A5g already present."); return
    ast.parse(CODE)
    def mk(t,s):
        c={"cell_type":t,"metadata":{},"source":s.splitlines(keepends=True),"id":uuid.uuid4().hex[:8]}
        if t=="code": c["execution_count"]=None; c["outputs"]=[]
        return c
    nb["cells"].append(mk("markdown",MARKDOWN)); nb["cells"].append(mk("code",CODE))
    NB.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
    print(f"[appended] {RUNG}; notebook now has {len(nb['cells'])} cells.")

if __name__=="__main__": main()
