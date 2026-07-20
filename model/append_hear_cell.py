"""Append A5i: HeAR (Google Health Acoustic Representations) under the honest
(pooled) grouping. HeAR is a health-acoustic foundation model (coughs/breathing),
NOT a speaker/ASR model like WavLM/HuBERT -- so it is the one frozen backbone that
could carry real cold pathology instead of the speaker-identity confound every
speech model collapses onto. Tests HeAR standalone AND fused with the honest
handcrafted G4+G9, vs the real bars. Idempotent; does NOT run.
From model/:  python append_hear_cell.py
"""
import ast, json, uuid
from pathlib import Path

NB = Path("run.ipynb")
RUNG = "A5i_hear_pooled"

MARKDOWN = "## A5i §7.3 — HeAR (Google Health Acoustic Representations) under the honest grouping\n\n" \
    "Every frozen *speech* backbone we tried (WavLM 0.579, HuBERT-large 0.592) lands on the\n" \
    "ECAPA speaker baseline (0.594) and adds ~0.000 over the handcrafted G4+G9 — they encode\n" \
    "*who is talking*, not *cold*. HeAR is different: a ViT trained by masked auto-encoding on\n" \
    "300M+ health-acoustic clips (cough/breath/throat), so it is the one candidate that could\n" \
    "carry genuine respiratory-pathology signal. We extract HeAR's 512-d embedding per URTIC\n" \
    "chunk (2s windows, mean-pooled), then run the SAME honest protocol as A5f/A5h: speaker\n" \
    "gate, standalone LR probe, and late fusion with G4+G9, against the bars WavLM 0.579,\n" \
    "HuBERT-large 0.592, ECAPA-speaker 0.594, handcrafted G4+G9 0.674(split42)/0.62(shadow).\n\n" \
    "Preprocessing is Google's exact mel-PCEN (vendored, verified numerically identical to\n" \
    "upstream, 0.0 diff). The pooler is reconstructed from the checkpoint's Linear(1024→512)\n" \
    "because the env's transformers 4.48 would otherwise build the wrong pooler. First run\n" \
    "extracts HeAR for train+devel (GPU, ~7 min); then seconds. You run this cell; I do not.\n\n" \
    "Two clean outcomes: HeAR also adds nothing → conclusion nailed shut (no frozen backbone\n" \
    "beats handcrafted CQT); HeAR adds signal → a genuine finding + the likely piece the other\n" \
    "team's HeAR+WavLM stack had. (If standalone HeAR is strong, HeAR+WavLM early fusion is the\n" \
    "planned follow-up.)"

CODE = r'''# A5i: HeAR (google/hear-pytorch) 512-d health-acoustic embedding under the pooled
# (speaker-honest) grouping. Standalone LR cold probe + late fusion with the honest
# handcrafted G4+G9, vs the real bars. HeAR slots in exactly where a handcrafted
# probe group does (a fixed 512-d vector per chunk) -- no head training.
# Output: results/A5i_hear_pooled.json
# Cost: first run extracts HeAR pooled for train+devel (GPU, ~7 min); re-runs skip it.

import json, statistics as st, time
from pathlib import Path
import numpy as np, torch
import soundfile as sf
from scipy import signal as _sg

from data.cached_dataset import stratified_grouped_split, load_labels
from features import extract_g4, extract_g9
from features.hear_model import load_hear
from honesty import (cold_probe, speaker_probe, fit_cold_probe, predict_logit,
                     fit_zscore, fuse, sweep_tau, evaluate_at_tau)
from speakers.cluster import load_pseudo_speakers

DATA_DIR="../dataset/ComParE2017_Cold_4students"; WAV_DIR=f"{DATA_DIR}/wav"
CACHE_ROOT="../cache"
HEAR_BACKBONE="google_hear-pytorch"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED=42; ALL_SEEDS=[42,123,7,999,31337]
PROBE_TSV=Path(f"{CACHE_ROOT}/pseudo_speakers/pooled_k420_seed42.tsv")   # HONEST grouping
HEAR_CACHE=Path(f"{CACHE_ROOT}/{HEAR_BACKBONE}/pooled")
OUT_JSON="../results/A5i_hear_pooled.json"
BETA_GRID=[0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0,24.0,32.0,48.0,64.0]
REF_A2_SPK_TOP1=0.0501; SPK_CHANCE=1.0/420.0
# honest bars (pooled grouping / shadow):
BAR={"wavlm_head":0.579,"hubertlarge_head":0.592,"ecapa_speaker":0.594,
     "handcrafted_G4G9_split42":0.674,"handcrafted_G4G9_shadow":0.62}

t0=time.time(); print(f"[device] {DEVICE}  [backbone] google/hear-pytorch  [grouping] pooled_k420 (honest)")
assert PROBE_TSV.exists(), f"missing {PROBE_TSV}; run build_pooled_pseudo_speakers.py first"
print(f"[honest bars] WavLM {BAR['wavlm_head']}  HuBERT-large {BAR['hubertlarge_head']}  "
      f"ECAPA-speaker {BAR['ecapa_speaker']}  handcrafted G4+G9 {BAR['handcrafted_G4G9_shadow']}(shadow)/"
      f"{BAR['handcrafted_G4G9_split42']}(split42)")

# ---- splits (pooled grouping), identical to A5f/A5h ----
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

# ---- STEP 0.5: extract HeAR 512-d for train+devel (idempotent) ----
need=sorted({s for _,(_,sm) in SPL.items() for s in sm})
have=sum(1 for s in need if (HEAR_CACHE/f"{s}.pt").exists())
print(f"\n[step0.5] HeAR pooled cache: {have}/{len(need)} present")
if have<len(need):
    print(f"  extracting HeAR 512-d (2s windows, mean-pool) -- GPU ViT + CPU mel-PCEN ...")
    HEAR_CACHE.mkdir(parents=True,exist_ok=True)
    emb_model=load_hear(device=DEVICE)
    done=0; miss=[s for s in need if not (HEAR_CACHE/f"{s}.pt").exists()]
    for s in miss:
        wav,sr=sf.read(f"{WAV_DIR}/{s}.wav",dtype="float32")
        if wav.ndim>1: wav=wav.mean(axis=1)
        if sr!=16000: wav=_sg.resample(wav,int(round(len(wav)*16000/sr))).astype("float32")
        ev=emb_model.embed_waveform(torch.from_numpy(np.ascontiguousarray(wav)))
        torch.save(ev.clone(),HEAR_CACHE/f"{s}.pt")
        done+=1
        if done%1000==0: print(f"    {done}/{len(miss)} extracted ({(time.time()-t0)/60:.1f} min)")
    del emb_model
    if DEVICE=="cuda": torch.cuda.empty_cache()
have=sum(1 for s in need if (HEAR_CACHE/f"{s}.pt").exists())
assert have==len(need), f"HeAR cache incomplete: {have}/{len(need)}"
print(f"  cache ready ({(time.time()-t0)/60:.1f} min)")

def _hear_mat(stems):
    return np.stack([torch.load(HEAR_CACHE/f"{s}.pt",weights_only=True,map_location="cpu").numpy()
                     for s in stems],0).astype(np.float32)
X_hear={n:_hear_mat(sm) for n,(_,sm) in SPL.items()}
EMB_DIM=X_hear["train_fit"].shape[1]
print(f"  HeAR embedding dim = {EMB_DIM}")

# ---- STEP 1: honesty gate + does the embedding carry cold signal at all? ----
print("\n=== STEP 1: HeAR honesty gate + embedding cold-probe ===")
sr_gate=speaker_probe(X_hear["train_fit"],spk["train_fit"],X_hear["devel_val"],spk["devel_val"],seed=SPLIT_SEED)
cr_emb=cold_probe(X_hear["train_fit"],y["train_fit"],X_hear["devel_val"],y["devel_val"],seed=SPLIT_SEED)
gate_lg=cr_emb.uar-0.5; gate_sg=sr_gate.top1-1.0/max(sr_gate.n_classes,1); gate_s1=gate_lg-gate_sg
gate_pass=bool(sr_gate.top1<=REF_A2_SPK_TOP1)
print(f"  speaker-probe top1={sr_gate.top1:.4f} (gate ref {REF_A2_SPK_TOP1}, chance {SPK_CHANCE:.4f}) -> gate {'PASS' if gate_pass else 'FAIL'}")
print(f"  cold-probe on embedding devel_val UAR={cr_emb.uar:.4f}  sub@1={gate_s1:+.4f}")

# ---- STEP 2: standalone HeAR cold probe (5 seeds) ----
print("\n=== STEP 2: HeAR standalone cold probe (5 seeds, pooled) ===")
def _ms(xs): return {"mean":float(st.mean(xs)),"std":float(st.stdev(xs)) if len(xs)>1 else 0.0,"n":len(xs)}
alone=[]
for seed in ALL_SEEDS:
    c,s=fit_cold_probe(X_hear["train_fit"],y["train_fit"],seed=seed)
    lg={n:predict_logit(c,s,X_hear[n]) for n in SPL}
    ta,_=sweep_tau(lg["train_threshold"],y["train_threshold"])
    u=evaluate_at_tau(lg["devel_test"],y["devel_test"],ta)["uar"]; alone.append(u)
    print(f"  seed {seed}: HeAR alone devel_test={u:.4f}")
hear_alone=_ms(alone)

# ---- STEP 3: fusion (5 seeds), pooled ----
# fuse(base,[z...],beta) = base + beta*MEAN(z...). Because it averages the extras,
# a naive fuse(l9,[Z4,Zh]) would HALVE G4's weight vs fuse(l9,[Z4]) -- so removing
# HeAR would NOT recover the handcrafted system, and a dilution artifact could fake
# "HeAR hurts". We therefore give HeAR its OWN independent weight via a nested
# 2-stage lock, so gamma=0 recovers the handcrafted system EXACTLY:
#   hc_only      : lock beta over fuse(l9,[Z4])           -> Hstar = l9 + beta*Z4       (G4+G9)
#   hc_plus_hear : lock gamma over Hstar + gamma*Zh       -> gamma=0 => Hstar (nested!) (G4+G9 + HeAR)
#   hear_base    : lock beta over fuse(lh,[Z4,Z9])        -> HeAR base + z(G4,G9)        (parallels A5h)
# hear_marginal = hc_plus_hear - hc_only is now a TRUE nested marginal; locked gamma=0
# for all seeds means the sweep chose to exclude HeAR entirely.
print("\n=== STEP 3: fusion (hc-only; +HeAR nested marginal; HeAR-base), pooled ===")
Xg4={n:extract_g4(sm,CACHE_ROOT,WAV_DIR)[:,4:] for n,(_,sm) in SPL.items()}
Xg9={n:extract_g9(sm,CACHE_ROOT) for n,(_,sm) in SPL.items()}
def _lock(logit_fn,grid,pname):
    rows=[]
    for p in grid:
        lg=logit_fn(p)
        tau,uthr=sweep_tau(lg["train_threshold"],y["train_threshold"])
        dt=evaluate_at_tau(lg["devel_test"],y["devel_test"],tau)
        rows.append({pname:float(p),"tau":float(tau),"uar_train_threshold":float(uthr),"devel_test":dt})
    return max(rows,key=lambda r:r["uar_train_threshold"]),rows
hearbase_k=[]; hc_only=[]; hc_hear=[]
for seed in ALL_SEEDS:
    ch,sh=fit_cold_probe(X_hear["train_fit"],y["train_fit"],seed=seed); lh={n:predict_logit(ch,sh,X_hear[n]) for n in SPL}; zh=fit_zscore(lh["train_fit"]); Zh={n:zh.apply(lh[n]) for n in SPL}
    c4,s4=fit_cold_probe(Xg4["train_fit"],y["train_fit"],seed=seed); l4={n:predict_logit(c4,s4,Xg4[n]) for n in SPL}; z4=fit_zscore(l4["train_fit"]); Z4={n:z4.apply(l4[n]) for n in SPL}
    c9,s9=fit_cold_probe(Xg9["train_fit"],y["train_fit"],seed=seed); l9={n:predict_logit(c9,s9,Xg9[n]) for n in SPL}; z9=fit_zscore(l9["train_fit"]); Z9={n:z9.apply(l9[n]) for n in SPL}
    # stage 1: lock handcrafted G4+G9
    lk_hc,_=_lock(lambda b:{n:fuse(l9[n],[Z4[n]],b) for n in SPL},BETA_GRID,"beta")
    Hstar={n:fuse(l9[n],[Z4[n]],lk_hc["beta"]) for n in SPL}   # locked handcrafted logit
    # stage 2: add HeAR with independent weight gamma (BETA_GRID starts at 0.0 -> nested)
    lk_g,_=_lock(lambda g:{n:Hstar[n]+g*Zh[n] for n in SPL},BETA_GRID,"gamma")
    # A5h-parallel: HeAR as the base, z-scored G4,G9 fused on top
    lk_hb,_=_lock(lambda b:{n:fuse(lh[n],[Z4[n],Z9[n]],b) for n in SPL},BETA_GRID,"beta")
    hc_only.append({"seed":seed,"locked":lk_hc})
    hc_hear.append({"seed":seed,"locked":lk_g,"beta_hc":lk_hc["beta"]})
    hearbase_k.append({"seed":seed,"locked":lk_hb})
    print(f"  seed {seed}: hc-only={lk_hc['devel_test']['uar']:.4f}  +HeAR(g*={lk_g['gamma']:.2f})={lk_g['devel_test']['uar']:.4f} "
          f"  HeAR-base+G4+G9(b*={lk_hb['beta']:.1f})={lk_hb['devel_test']['uar']:.4f}")
hc_g4g9=_ms([r["locked"]["devel_test"]["uar"] for r in hc_only])
hc_hear_g4g9=_ms([r["locked"]["devel_test"]["uar"] for r in hc_hear])
hear_g4g9=_ms([r["locked"]["devel_test"]["uar"] for r in hearbase_k])
gammas=[r["locked"]["gamma"] for r in hc_hear]      # HeAR's independent weight (0 => excluded)
hb_betas=[r["locked"]["beta"] for r in hearbase_k]
hear_marginal=hc_hear_g4g9["mean"]-hc_g4g9["mean"]
# G9 speaker gate under pooled (context)
g9s=speaker_probe(Xg9["train_fit"],spk["train_fit"],Xg9["devel_val"],spk["devel_val"])

# ---- VERDICT ----
print(f"\n=== HeAR VERDICT (pooled, honest) ===")
print(f"  HeAR standalone:            {hear_alone['mean']:.4f} +/- {hear_alone['std']:.4f}   "
      f"(WavLM {BAR['wavlm_head']}, HuBERT-large {BAR['hubertlarge_head']}, ECAPA {BAR['ecapa_speaker']})")
print(f"  handcrafted-only G4+G9:     {hc_g4g9['mean']:.4f} +/- {hc_g4g9['std']:.4f}   (in-run; bar {BAR['handcrafted_G4G9_split42']} split42 / {BAR['handcrafted_G4G9_shadow']} shadow)")
print(f"  handcrafted G4+G9 + HeAR:   {hc_hear_g4g9['mean']:.4f} +/- {hc_hear_g4g9['std']:.4f}   gammas={gammas}  (gamma=0 => HeAR excluded)")
print(f"  --> HeAR NESTED marginal over handcrafted: {hear_marginal:+.4f}")
print(f"  HeAR-base + G4 + G9 (A5h-style): {hear_g4g9['mean']:.4f} +/- {hear_g4g9['std']:.4f}   betas={hb_betas}")
print(f"  HeAR speaker gate top1={sr_gate.top1:.4f} (ref {REF_A2_SPK_TOP1}) -> {'PASS' if gate_pass else 'FAIL'}")
beats_speech_heads = hear_alone["mean"] > max(BAR["wavlm_head"],BAR["hubertlarge_head"],BAR["ecapa_speaker"])
adds_over_handcrafted = hear_marginal > 0.005
print(f"  HeAR standalone beats all speech heads (WavLM/HuBERT/ECAPA): {beats_speech_heads}")
print(f"  HeAR adds >0.005 nested marginal over handcrafted-only: {adds_over_handcrafted}")

out={"rung_id":"A5i_hear_pooled",
     "description":"HeAR (google/hear-pytorch) 512-d health-acoustic embedding under the pooled "
        "(speaker-honest) grouping. Standalone LR cold probe + late fusion with honest G4+G9 "
        "(NESTED marginal: handcrafted G4+G9 lock, then HeAR added at independent weight gamma; "
        "gamma=0 recovers handcrafted exactly), vs WavLM/HuBERT/ECAPA/handcrafted bars. "
        "Preprocessing = Google's exact mel-PCEN (vendored, verified 0.0 vs upstream); pooler "
        "reconstructed Linear(1024->512) from checkpoint.",
     "backbone":"google/hear-pytorch","embedding_dim":int(EMB_DIM),"pooling":"2s windows, mean-pooled",
     "grouping":"pooled_k420_seed42","split_seed":SPLIT_SEED,"all_seeds":ALL_SEEDS,"beta_grid":BETA_GRID,
     "honest_bars":BAR,
     "speaker_gate":{"top1":float(sr_gate.top1),"n_classes":int(sr_gate.n_classes),
                     "reference":REF_A2_SPK_TOP1,"chance":SPK_CHANCE,"pass":gate_pass},
     "embedding_cold_probe_devel_val":{"uar":float(cr_emb.uar),"sub_at_1":float(gate_s1)},
     "hear_standalone":hear_alone,
     "handcrafted_only_g4_g9_inrun":hc_g4g9,"handcrafted_g4_g9_plus_hear":hc_hear_g4g9,
     "hear_nested_marginal_over_handcrafted":float(hear_marginal),"hear_locked_gammas":gammas,
     "hear_base_plus_g4_g9":hear_g4g9,"hear_base_locked_betas":hb_betas,
     "hc_only_runs":hc_only,"hc_plus_hear_runs":hc_hear,"hear_base_runs":hearbase_k,
     "g9_speaker_gate":{"top1":float(g9s.top1),"reference":REF_A2_SPK_TOP1,"pass":bool(g9s.top1<=REF_A2_SPK_TOP1)},
     "beats_speech_heads":bool(beats_speech_heads),"adds_over_handcrafted":bool(adds_over_handcrafted),
     "elapsed_minutes":(time.time()-t0)/60.0}
Path(OUT_JSON).parent.mkdir(parents=True,exist_ok=True)
Path(OUT_JSON).write_text(json.dumps(out,indent=2))
print(f"\n[wrote] {OUT_JSON}  [done] {(time.time()-t0)/60:.1f} min")
'''

def main():
    nb=json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"]=="code" and '"A5i_hear_pooled"' in "".join(c["source"]):
            print("[skip] A5i already present."); return
    ast.parse(CODE)
    def mk(t,s):
        c={"cell_type":t,"metadata":{},"source":s.splitlines(keepends=True),"id":uuid.uuid4().hex[:8]}
        if t=="code": c["execution_count"]=None; c["outputs"]=[]
        return c
    nb["cells"].append(mk("markdown",MARKDOWN)); nb["cells"].append(mk("code",CODE))
    NB.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8")
    print(f"[appended] {RUNG}; notebook now has {len(nb['cells'])} cells.")

if __name__=="__main__": main()
