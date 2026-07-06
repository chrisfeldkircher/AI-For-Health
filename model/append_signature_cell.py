"""Append the A5b K=3 log-signature (G10) candidate cell to run.ipynb.
Idempotent; does NOT run. From model/:  python append_signature_cell.py
"""
import ast, json, uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5b_k3_signature_5seed"

MARKDOWN = "## A5b §4.13 — K=3 with G10 (depth-2 log-signature of a raw-acoustic path)\n\n" \
    "The rough-path branch from the work plan: a depth-2 log-signature (increments +\n" \
    "Levy areas) of a 4-channel raw-acoustic path (log-RMS, log-F0, spectral centroid,\n" \
    "ZCR), pure-numpy so no signature library is needed. Tests whether order-sensitive\n" \
    "cross-channel temporal structure that pooling discards adds cold signal, and whether\n" \
    "it passes the speaker gate. Applied to RAW acoustics (not WavLM), per the plan's\n" \
    "question of whether the embedding step suppresses fine temporal signal. You run this."

CODE = r'''# A5b K=3 candidate: G10 = depth-2 log-signature of a 4-channel raw-acoustic path
# (Tier-2 4.13). Mirrors the CQT cell: (a) STEP 1 extracts the signature cache for
# train+devel if missing (parallel, CPU; pYIN-based, so ~15-30 min first run),
# (b) STEP 2 standalone honesty audit (cold UAR + speaker top-1). Config A = K=2 with
# G10 replacing G5; Config B = K=3 = A2 + G4_gi + G5_mod + G10. 5 seeds.
# Output: results/A5b_k3_signature_5seed.json

import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v,"1")

import json, statistics as st, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

from data.cached_dataset import PooledCacheDataset, stratified_grouped_split, load_labels
from data.data import _load_audio
from features import (LayerWeightedPooledHead, extract_g4, extract_g5,
                      extract_g10, signature_features)
from features.train import _pooled_collate, predict_probs
from honesty import (cold_probe, speaker_probe, fit_cold_probe, predict_logit,
                     fit_zscore, fuse, sweep_tau, evaluate_at_tau)
from speakers.cluster import load_pseudo_speakers

DATA_DIR="../dataset/ComParE2017_Cold_4students"; WAV_DIR=f"{DATA_DIR}/wav"
CACHE_ROOT="../cache"; BACKBONE="microsoft_wavlm-large"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED=42; ALL_SEEDS=[42,123,7,999,31337]; CLIP_SECS=8.0; N_JOBS=-1
PROBE_TSV=Path(f"{CACHE_ROOT}/pseudo_speakers/k210_seed42.tsv")
SIG_DIR=Path(f"{CACHE_ROOT}/handcrafted/signature")
OUT_JSON="../results/A5b_k3_signature_5seed.json"
BETA_GRID=[0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0]
REF_K2_MEAN=0.7037; REF_K2_STD=0.0060; K3_BAR=REF_K2_MEAN+0.005
REF_A2_SPK_TOP1=0.0501; SPK_CHANCE=1.0/210.0

t0=time.time(); print(f"[device] {DEVICE}")
print(f"[K3 admission] mean devel_test UAR > {K3_BAR:.4f} AND G10 speaker top-1 <= {REF_A2_SPK_TOP1:.4f}")

full_train = PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="train")
full_devel = PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,split="devel")
labels_map = load_labels(DATA_DIR); pseudo = load_pseudo_speakers(PROBE_TSV)
tf,tt = stratified_grouped_split(full_train.files, labels_map, pseudo, val_frac=0.10, seed=SPLIT_SEED)
dv,dtst = stratified_grouped_split(full_devel.files, labels_map, pseudo, val_frac=0.50, seed=SPLIT_SEED)
def _stems(fs): return [f[:-4] if f.endswith(".wav") else f for f in fs]
SPL={"train_fit":(tf,_stems(tf)),"train_threshold":(tt,_stems(tt)),
     "devel_val":(dv,_stems(dv)),"devel_test":(dtst,_stems(dtst))}
y={n:np.array([labels_map[f] for f in fs],dtype=np.int64) for n,(fs,_) in SPL.items()}
spk={n:np.array([pseudo[s] for s in sm],dtype=np.int64) for n,(_,sm) in SPL.items()}

# STEP 1: ensure signature cache (parallel; pYIN inside signature_features is slow)
print(f"\n=== STEP 1: signature cache ({SIG_DIR}) ===")
SIG_DIR.mkdir(parents=True,exist_ok=True)
need=sorted({s for _,(_,sm) in SPL.items() for s in sm})
todo=[s for s in need if not (SIG_DIR/f"{s}.npy").exists()]
print(f"  {len(need)} needed; {len(need)-len(todo)} cached; {len(todo)} to compute")
def _one(stem):
    tgt=SIG_DIR/f"{stem}.npy"
    if tgt.exists(): return 0
    audio,sr=_load_audio(str(Path(WAV_DIR)/f"{stem}.wav"))
    np.save(tgt, signature_features(audio[:int(CLIP_SECS*sr)], sr=sr).astype(np.float32)); return 1
if todo:
    try:
        from joblib import Parallel, delayed
        Parallel(n_jobs=N_JOBS, backend="loky", verbose=5, batch_size=8)(delayed(_one)(s) for s in todo)
    except ImportError:
        for s in todo: _one(s)
assert all((SIG_DIR/f"{s}.npy").exists() for s in need), "signature cache incomplete"
print(f"  ready ({(time.time()-t0)/60:.1f} min)")
Xg4={n:extract_g4(sm,CACHE_ROOT,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
Xg5={n:extract_g5(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
Xg10={n:extract_g10(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
print(f"  dims: G4={Xg4['train_fit'].shape[1]} G5={Xg5['train_fit'].shape[1]} G10={Xg10['train_fit'].shape[1]}")

# STEP 2: standalone honesty audit of G10
print(f"\n=== STEP 2: G10 standalone honesty audit (train_fit -> devel_val) ===")
g10c=cold_probe(Xg10["train_fit"],y["train_fit"],Xg10["devel_val"],y["devel_val"])
g10s=speaker_probe(Xg10["train_fit"],spk["train_fit"],Xg10["devel_val"],spk["devel_val"])
print(f"  G10 cold UAR={g10c.uar:.4f}  speaker top-1={g10s.top1:.4f} (chance {SPK_CHANCE:.4f})")
spk_gate = g10s.top1 <= REF_A2_SPK_TOP1
print(f"  speaker gate: {'PASS' if spk_gate else 'FAIL'}")

def _a2(head,files):
    ds=PooledCacheDataset(DATA_DIR,CACHE_ROOT,BACKBONE,file_list=files)
    p,_=predict_probs(head,DataLoader(ds,batch_size=256,shuffle=False,num_workers=0,collate_fn=_pooled_collate),DEVICE)
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def _head(seed):
    s=full_train[0]["pooled"]; nl,sd=s.shape
    h=LayerWeightedPooledHead(n_layers=nl,stat_dim=sd,proj_dim=128,n_classes=2,dropout=0.5).to(DEVICE)
    st_=torch.load(f"{CACHE_ROOT}/{BACKBONE}/head_A2grouped_honestprior_seed{seed}.pt",weights_only=True,map_location=DEVICE)
    h.load_state_dict(st_["state_dict"]); h.eval(); return h
def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0,"n":len(xs)}

print(f"\n=== STEP 3: per-seed A2.5 logits + z-logits ===")
per={}
for seed in ALL_SEEDS:
    h=_head(seed); a2={n:_a2(h,fs) for n,(fs,_) in SPL.items()}
    def zg(X):
        c,s=fit_cold_probe(X["train_fit"],y["train_fit"],seed=seed); lg={n:predict_logit(c,s,X[n]) for n in SPL}
        z=fit_zscore(lg["train_fit"]); return {n:z.apply(lg[n]) for n in SPL}
    per[seed]={"a2":a2,"z4":zg(Xg4),"z5":zg(Xg5),"z10":zg(Xg10)}; del h
    print(f"  seed {seed} done")

def sweep(keys):
    runs=[]
    for seed in ALL_SEEDS:
        c=per[seed]; best=None; rows=[]
        for beta in BETA_GRID:
            zt=[c[k]["train_threshold"] for k in keys]; zd=[c[k]["devel_test"] for k in keys]
            tau,uthr=sweep_tau(fuse(c["a2"]["train_threshold"],zt,beta),y["train_threshold"])
            dt=evaluate_at_tau(fuse(c["a2"]["devel_test"],zd,beta),y["devel_test"],tau)
            rows.append({"beta":float(beta),"tau":float(tau),"uar_train_threshold":float(uthr),"devel_test":dt})
        locked=max(rows,key=lambda r:r["uar_train_threshold"]); runs.append({"seed":seed,"sweep":rows,"locked":locked})
    return runs

print(f"\n=== STEP 4: config A (A2+G4+G10) and config B (A2+G4+G5+G10) ===")
A=sweep(["z4","z10"]); B=sweep(["z4","z5","z10"])
Aa=_ms([r["locked"]["devel_test"]["uar"] for r in A]); Ba=_ms([r["locked"]["devel_test"]["uar"] for r in B])
Ab=[r["locked"]["beta"] for r in A]; Bb=[r["locked"]["beta"] for r in B]
print(f"  Config A (K=2 A2+G4+G10): {Aa['mean']:.4f} +/- {Aa['std']:.4f}  betas={Ab}")
print(f"  Config B (K=3 A2+G4+G5+G10): {Ba['mean']:.4f} +/- {Ba['std']:.4f}  betas={Bb}")
admit = Ba["mean"]>K3_BAR and spk_gate
decision=("k3_signature_admitted" if admit else
          ("k3_signature_uar_ok_but_speaker_gate_fail" if Ba["mean"]>K3_BAR and not spk_gate else
           ("k2_with_signature_admitted" if Aa["mean"]>REF_K2_MEAN+0.005 and spk_gate else "k2_g4_g5_stays_canonical")))
print(f"\n=== VERDICT: {decision} (speaker gate {'PASS' if spk_gate else 'FAIL'}) ===")

out={"rung_id":"A5b_k3_signature_5seed",
     "description":"A5b K=3 candidate G10 = depth-2 log-signature (10-d: increments + "
        "Levy areas of a 4-channel raw-acoustic path log-RMS/log-F0/centroid/ZCR, per-"
        "channel z-scored). Config A = K=2 with G10 replacing G5; Config B = K=3. 5 seeds, "
        "beta locked on train_threshold, reported devel_test. Two-dim gate: mean UAR > "
        "0.7087 AND G10 speaker top-1 <= 0.0501.",
     "split_seed":SPLIT_SEED,"all_seeds":ALL_SEEDS,"beta_grid":BETA_GRID,
     "feature_dims":{"g4":int(Xg4['train_fit'].shape[1]),"g5":int(Xg5['train_fit'].shape[1]),"g10":int(Xg10['train_fit'].shape[1])},
     "k3_admission_threshold":K3_BAR,
     "g10_standalone_honesty":{"cold_uar_devel_val":float(g10c.uar),
        "speaker_top1_devel_val":float(g10s.top1),"speaker_nmi_devel_val":float(g10s.nmi),
        "speaker_chance":SPK_CHANCE,"reference_a2_speaker_top1":REF_A2_SPK_TOP1,"speaker_gate_pass":bool(spk_gate)},
     "config_a_k2_replacement":{"runs":A,"locked_betas":Ab,"uar_devel_test":Aa,"delta_vs_k2_locked_mean":Aa["mean"]-REF_K2_MEAN},
     "config_b_k3_addition":{"runs":B,"locked_betas":Bb,"uar_devel_test":Ba,"delta_vs_k2_locked_mean":Ba["mean"]-REF_K2_MEAN,"admit_k3":bool(admit)},
     "decision":decision,"elapsed_minutes":(time.time()-t0)/60.0}
Path(OUT_JSON).parent.mkdir(parents=True,exist_ok=True); Path(OUT_JSON).write_text(json.dumps(out,indent=2))
print(f"\n[wrote] {OUT_JSON}  [done] {(time.time()-t0)/60:.1f} min")
'''

def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"]=="code" and '"A5b_k3_signature_5seed"' in "".join(c["source"]):
            print("[skip] signature cell already present."); return
    ast.parse(CODE)
    def mk(t,s):
        c={"cell_type":t,"metadata":{},"source":s.splitlines(keepends=True),"id":uuid.uuid4().hex[:8]}
        if t=="code": c["execution_count"]=None; c["outputs"]=[]
        return c
    nb["cells"].append(mk("markdown",MARKDOWN)); nb["cells"].append(mk("code",CODE))
    NB.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
    print(f"[appended] {RUNG}; notebook now has {len(nb['cells'])} cells.")

if __name__=="__main__": main()
