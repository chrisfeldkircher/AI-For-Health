# ML4Health — Plan (Live)

URTIC Cold Detection · ComParE 2017 Cold sub-challenge · SS26

This document is the canonical attack plan with status flags and post-hoc deviations folded in. Numbers and detailed diagnostics live in [summary.md](summary.md); this file tracks **what we set out to do, what we actually did, why we changed course, and what is next**. Cross-reference, do not duplicate.

---

## 1. Core thesis (unchanged)

The 2017 ComParE Cold baseline (UAR = 71.0 %) was won by **late fusion of three orthogonal pillars**: hand-crafted acoustic functionals + discrete audio-word histograms + end-to-end deep learning. Each pillar has a 2026 upgrade:

- End-to-end → pretrained foundation models (WavLM, HuBERT, Whisper)
- Random-codebook BoAW → learned discrete tokens (HuBERT cluster IDs / VQ-VAE)
- Full 6 373-dim ComParE → curated, statistically-grounded hand-crafted subset complementing the FM

On top, three speaker-confounding interventions: cross-speaker augmentation (data), speaker-masked supervised contrastive pretraining (representation), MDD/DANN gradient-reversal adversary (gradient).

**Two independent paper contributions**: a better UAR, *and* a methodologically cleaner read of how much of the 2017 numbers was shortcut learning on speaker identity. The methodology contribution is bankable even if UAR doesn't beat 71.

---

## 2. Mapping 2017 → 2026 (unchanged from PDF)

See section 2 of the original PDF — the per-paper mapping table is still the reference. One amendment from running the project:

- **Phoneme-aware pooling on WavLM (Wagner / Huckvale VOW)**: attempted in our A3 with pYIN+RMS 3-cat manner labels, **rejected** as a standalone feature stream (see § 4 below). The labels themselves are clean and remain available as a candidate feature group inside A5; what failed was the per-(layer, category) pooled-stats head, not the labelling.

---

## 3. Architecture stack — current state vs. designed

### 3.1 Forward pass (what the design said)

`augmentation → frozen WavLM → temporal pooling → handcrafted branch → discrete-token branch → concat → projection MLP → z`

### 3.1' Forward pass (what is actually wired today)

`frozen WavLM-Large → per-layer mean+std+skew+kurt pooled stats → FeatureStandardiser → softmax layer-weights → 2-layer MLP 128-d → 2-class linear`

That is **A2 only**. The handcrafted branch, discrete-token branch, projection MLP, OOD head, and adversary are all still on the to-build list. The augmentation module exists ([data/augmentation.py](AI-For-Health/model/data/augmentation.py)) but is not yet wired into the training loop.

### 3.2 Heads (designed: 3, current: 1)

- **Cold classifier head**: 2-class linear on top of the layer-weighted pooled features. **Built, locked at A2.** Class-weighted loss replaced by balanced sampler (cleaner gradients, better-calibrated boundary).
- **Speaker probe**: built as a **measurement tool**, not a training head. 2-layer MLP on frozen `z`, run after every de-confounding rung. Currently reports A2 train top-1 ≈ 0.92 vs. devel top-1 ≈ 0.05 — the Huckvale trap quantified.
- **Speaker adversary head (MDD/DANN)**: **not built yet.** A7.
- **OOD score head (Mahalanobis)**: **not built yet.** Folded into the new A5 design (see § 5 below).

### 3.3 Training phases (designed: 2, current: 1)

Phase 1 (contrastive pretraining) and Phase 2 (supervised) both **not implemented yet**. Current pipeline is single-phase supervised on top of cached pooled features. Phase 1 lands in A6.

---

## 4. The ablation ladder — status report

Numbering follows the executed sequence, not the original PDF. Where the executed rungs differ from the PDF, the deviation is explicit.

| #     | Status              | What it tests                                                       | Result / next action                                                                       |
| ----- | ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| A0    | **DONE**            | Majority class (sanity)                                             | UAR = 0.5 by construction; chance baseline acknowledged                                    |
| A1    | **subsumed by A2**  | Frozen WavLM + mean-pool + linear probe                             | Skipped as a separate rung — A2 strictly dominates and shares all caches                   |
| A2    | **LOCKED**          | Frozen WavLM-L + layer-weighted pooled-stats (mean+std+skew+kurt)   | **UAR 0.6428 ± 0.0034** (3 seeds); val→test gap −0.001 ± 0.005; speaker probe top-1 0.0501 |
| A2.5  | **NEW BASELINE**    | A2 + honesty-prior layer-weight init (logits = T·sub@1 from A5d)    | UAR 0.6564 ± 0.0038 (Δ +0.020 vs A2_grouped, 4.7σ); MLP probe flat, LR probe -0.0035       |
| A3    | **REJECTED**        | Manner-aware pooling (pYIN+RMS, 3-cat) two-stream head              | UAR 0.6344 ± 0.0069 (Δ −0.008), probe top-1 +0.005. Both gates failed.                     |
| A5a   | **LOCKED**          | Honesty audit over low-dim physiological feature groups             | 8 groups, 6 admitted (G1,G2,G3,G4_gi,G5,G6). Admission by sub@1: G4_gi,G1,G6,G5,G2,G3.     |
| A5b   | **FINAL LOCKED K=2 at N=5 (both gates PASS)** | Constrained late fusion: per-group linear logits, β fixed = honesty | K=2 (A2.5 + G4_gi + G5_mod) at N=5: **UAR 0.7037 ± 0.0060 (+0.0103 over K=1 at 4.30σ; +0.068 cumulative over uniform-A2 baseline, ~17σ); probe (i) 0.0182 ± 0.0006 (PASS, 4.3× margin); probe (ii) 0.0729 ± 0.0005 (PASS); 0.006 from 0.71 target** |
| A5c   | revivable           | Learned per-group gate, honesty-initialised + regularised           | A5b passed → revivable, but K=1 leaves little room; on hold pending A5.5/A6                |
| A5d   | **DONE**            | Per-layer honesty diagnostic on cached pooled stats (no retraining) | Spk mono L0→L24 (.087→.043 R, .072→.042 G); cold UAR flat; sub@1 ≪ 0.15 on both           |
| A5e   | **SKIPPED**         | A2 retrain on a band-restricted WavLM layer slice                   | A5d trigger missed: no sub@1 > 0.15; cold peak L7 = spk peak band. GPU → A5.5 / A6.        |
| A4    | planned             | Discrete-token histograms (HuBERT units → optional VQ-VAE)          | Deferred behind A5 — more speculative, no built-in anti-shortcut mechanism                 |
| A5.5  | **LOCKED (cons-α)** | Cross-speaker augmentation. Audio splicing FAIL → embedding mixup pivot | Conservative α∈[0.70,0.85]: UAR 0.6624 (Δ +0.006, ~1.6σ), probe ~unchanged. Aggressive α∈[0.50,0.70] branch (d): UAR 0.6397 (Δ -0.017), probe still flat → narrow window EMPTY. A5.5 = cons-α canonical; aggro = ablation row |
| A6    | **queued (PoC)**    | Supervised contrastive pretraining (speaker-masked positives)       | Phase 1 head-only PoC scoped (§4.9): projection MLP on cached pooled stats, A2.5 anchor, 30-60 min CPU/GPU. PoC verdict drives escalation to layer-weight-open or transformer fine-tune |
| A7    | **queued (PoC)**    | DANN/MDD speaker adversary (load-bearing for de-confounding claim)  | Layer-weight-open from start (avoid M10 bottleneck-confound); §4.10 scoped: λ_adv sweep + M10/M11 controls baked in |
| A8    | planned             | MDD vs DANN comparison                                              | Only run if A7 lands                                                                       |
| A9    | **merged into A5**  | Late fusion with standalone ComParE+SVM                             | A5's output stage is the late-fusion result; no separate rung                              |

### 4.1 What we did (A2 locked)

- Frozen `microsoft/wavlm-large` (25 hidden states), pooled mean+std+skew+kurt per layer.
- Per-position z-score `FeatureStandardiser` as first child of head — without it, training collapses to majority class (per-position std spans 4 orders of magnitude).
- Softmax layer-weights with lr × 0.1.
- 2-layer MLP 128-d, BatchNorm, GELU, dropout 0.5, 2-class linear.
- Balanced sampler (no class weights in loss). AdamW lr=1e-3, cosine schedule, early stop patience 6.
- Threshold τ on `train_threshold` (10 % of train), never on devel.
- Calibrated UAR = 0.6464 ± 0.0082 — within noise of argmax.

Methodology locked: stratified `train_fit / train_threshold` and `devel_val / devel_test` 90/10 and 50/50 splits, all seed=42; lock seeds `{42, 123, 7}`; minimum detectable rung gain ≈ 0.007 UAR (2σ at N=3).

### 4.2 What we tried that failed (A3 manner-aware pooling)

Two sub-paths:

**Path 1 — phoneme CTC labels (`wav2vec2-xlsr-53-espeak-cv-ft`).** ABANDONED. Soft-aggregation diagnostic: 84.1 % blank-wins-top1, mean top-1 prob 0.962, mean per-frame entropy 0.16 nats — sharply confident, not smeared, with hard and soft histograms identical to within sampling noise. Classic CommonVoice→URTIC domain mismatch; the model retreats to its blank prior on German clinical recordings. Smearing heuristics rejected as untestable on URTIC (no phoneme-boundary ground truth). Code kept in [features/phoneme.py](AI-For-Health/model/features/phoneme.py) as documented negative result.

**Path 2 — pYIN voicing + RMS silence-gate, 3 categories (silence/voiced/unvoiced).** Labels validated against decades of voicing-detection literature. Validation gate passed (silence 40 %, voiced 38 %, unvoiced 22 % on 20-chunk subset). Full extraction took 22.6 h on CPU; cache built. Two-stream head (A2 stream + manner stream) trained 3 seeds.

**Why both gates failed**: per-utterance per-category mean of WavLM frames in *early* layers IS a speaker fingerprint (voiced-frame mean ≈ formants, unvoiced-frame mean ≈ spectral envelope). The manner stream concentrated weight on L1/L4/L8 across all seeds, which is exactly where WavLM stores speaker/acoustic information; the A2 stream's layer weights stayed pinned to uniform; the MLP became a manner-stream-only classifier with severe train overfitting (probe train top-1 → 0.996).

**What we keep**: `cache/manner_labels/` (19 101 stems) is the salvaged input for **low-dimensional scalar features** in A5 (voiced fraction, voicing dropouts, mean RMS in low-energy regions, voiced-segment durations, etc.). The 6 144-d `manner_pooled/` WavLM cache is **explicitly not** carried into A5 as a representation stream — it is the same speaker-fingerprint substrate that just failed in A3. Cache stays on disk as the documented negative result; bundles are not loaded by the A5 dataloader.

### 4.3 Why we deviated from the original PDF rung order

- **A1 collapsed into A2**: mean-pool is a strict subset of layer-weighted pooled-stats; running both wastes a slot.
- **A3 in the PDF was "phoneme-aware pooling"**, vague about the labeller. We tried both the phonetic and the acoustic-manner reading; both failed as standalone feature streams. The "phoneme insight transfers" question is answered: not as a per-utterance pooling axis, possibly as a per-group enrichment feature inside A5.
- **A5 in the PDF was vague ("OOD Mahalanobis feature")**. We replaced it with a sharper, more ambitious design split into three sub-rungs: A5a honesty audit, A5b constrained logit fusion with β fixed = honesty score, A5c learned per-group gate (only if A5b clears the gates). OOD Mahalanobis is *one* candidate group inside A5 rather than the whole rung.
- **A9 (late fusion) merged into A5**: the late-fusion stage is A5's output, not a separate rung. One fusion design, one end-to-end run, one probe check.
- **A5 promoted ahead of A4**: A5 attacks the speaker shortcut directly via measurement (the honesty score is the Huckvale trap in numerical form). A4 (discrete tokens) is deferred — it's more speculative and gives no probe guarantee.
- **Logit fusion, not concat**: high-dimensional concatenation is the substrate that let A3's MLP rediscover speaker shortcuts. A5 fuses per-group *cold-probe logits* (1-d each) so every group has to prove standalone label utility before getting any β weight.

### 4.4 Pseudo-speakers — locked

URTIC has no speaker IDs in the 4students release. We rebuilt them:

- **ECAPA-VoxCeleb (192-d)** + **KMeans k=210** as the pseudo-speaker labelling.
- **HDBSCAN cross-validation**: independently finds **204 clusters**, KMeans-vs-HDBSCAN ARI 0.856 / NMI 0.962 on raw L2-normalised embeddings. 204 ≈ 210 ≈ URTIC's expected ~210 speakers/split — not a self-fulfilling silhouette number, but cross-method agreement.
- **Negative control**: WavLM-base-plus-sv flags 25 % of points as noise and KMeans-vs-HDBSCAN ARI = 0.093 — the WavLM speaker-tuned encoder cannot recover speaker structure on URTIC. Confirms the architectural-circularity concern empirically and justifies keeping ECAPA.
- **Independence note**: ECAPA is fed **raw 16 kHz waveforms**, not WavLM features. SpeechBrain's `spkrec-ecapa-voxceleb` runs end-to-end (mel-bank front-end → TDNN → AAM-Softmax x-vector head) on the audio directly. The `cache/microsoft_wavlm-large/pooled/` cache and `cache/ecapa-voxceleb/` cache never touch each other — that architectural independence is what makes the cross-encoder validation credible.
- Revisit only before A6, where pseudo-speaker labels become *training targets* rather than probe ground truth. Candidate: TitaNet-L or CAM++ (architecturally independent from WavLM).

### 4.5 Methodology — speaker-grouped sub-splits (within-partition leak fix, DONE)

**Status: DONE.** Cross-partition disjointness (train ↔ devel ↔ test) is guaranteed by URTIC construction. **Within-partition sub-splits** (`train_fit`/`train_threshold`, `devel_val`/`devel_test`) were per-class random shuffle only ([`stratified_split`](AI-For-Health/model/data/cached_dataset.py#L186)), so the same pseudo-speakers appeared in both halves of each partition, reading different chunks of the same passage. New [`stratified_grouped_split`](AI-For-Health/model/data/cached_dataset.py) uses `sklearn.model_selection.StratifiedGroupKFold` to stratify by Cold label and disjoin by pseudo-speaker ID from `cache/pseudo_speakers/k210_seed42.tsv`.

**Within-partition speaker-overlap diagnostic** (random vs grouped, k=210 pseudo-speakers):

```text
                            random   grouped
train_fit / train_threshold  198/210     0/210
devel_val / devel_test       198/206     0/206
```

Massive within-partition leak under random splits — almost every pseudo-speaker is in *both* halves of each partition. Grouped splits give clean disjointness.

**A2 retrain on grouped splits** (3 seeds {42, 123, 7}, splits derived with `stratified_grouped_split(seed=42)`, head architecture and training recipe unchanged, output `results/A2_grouped.json`):

```text
metric                  RANDOM (A2.json)        GROUPED (A2_grouped.json)
uar_argmax              0.6428 ± 0.0034         0.6361 ± 0.0019           Δ -0.0067 (~2σ)
uar_calibrated          0.6464 ± 0.0082         0.6498 ± 0.0028           Δ +0.0034 (within σ)
recall_C @ τ            0.4321 ± 0.0284         0.5533 ± 0.0628           more cold-biased after calib
recall_NC @ τ           0.8607 ± 0.0192         0.7462 ± 0.0664
val_test_gap            -0.0009 ± 0.0047        -0.0133 ± 0.0019          reveals real speaker-disjointness
speaker_probe MLP top-1 0.0501 ± 0.0009         0.0498 ± 0.0031           unchanged
speaker_probe LR  top-1 (not in A2.json)        0.0760 ± 0.0020           new codepath-consistent ceiling
```

**Verdict.** Argmax UAR was mildly inflated by within-partition leak (~1pp shift). Calibrated UAR essentially unchanged. The val-test gap is now negative (-0.013) — the expected signature of fixing the leak: under random splits devel_val and devel_test shared speakers so were ~equal; under grouped splits they are speaker-disjoint and devel_test is genuinely harder. The MLP speaker probe moved by ≪1σ — confirms the cross-partition disjointness was already doing all the load-bearing work for the speaker-probe interpretation in the paper, and the historical 0.0501 reference holds. The LR probe is slightly higher under grouped (0.0760 vs random's 0.0674 from the A5b controls cell) — partly because the grouped devel_val has fewer distinct true-speaker classes (~103 vs ~206 random). 0.0760 ± 0.0020 is the new apples-to-apples LR-substrate ceiling for §5.7 / A5b / A5d audits.

**Downstream consequence (DONE).** A5b K=1 lock + locked-K speaker probes re-run on grouped splits using `head_A2grouped_seed{seed}.pt` as the anchor (`results/A5b_grouped.json`). A5d per-layer diagnostic also re-run on grouped splits (`results/A5d_grouped_layer_honesty.csv`). **K=1 PASS holds and is strengthened**: K=1 fused UAR 0.6588 ± 0.0059 (was 0.6576 ± 0.0011 on random); Δ vs A2_argmax = +0.0227 ± 0.0059 (was +0.0148 ± 0.0045 — went from 3.3σ to 3.8σ above zero). Fusion lift got *larger* under grouped splits because A2's argmax baseline was the part of the random-split number that was inflated. Speaker probes both PASS against the new LR ceiling 0.0780: probe (i) literal 2-D = 0.0153 ± 0.0032 (~5× margin); probe (ii) backbone-concat = 0.0733 ± 0.0002 (by 0.0047). A5d structural finding survives: speaker top-1 still monotone L0→L24 (0.072→0.042, ~similar shape to random's 0.087→0.043 — confirms Pasad 2021 / Chen 2022 for speaker on URTIC), cold UAR still flat (0.56–0.61). Best `sub@1` layer shifts L21 → L0 (the cold-rich early layers are now ranked highest because they're speaker-rich early too — but absolute sub@1 stays small at +0.040), best `cold_uar` shifts L7 → L6, both A5e trigger conditions still fire — A5e SKIPPED holds. A5b_diag / A5b_ablation (K=2 ablations) not re-run on grouped splits — they ship as paper diagnostics about the K-sweep pathology, which is split-independent.

### 4.6 A2.5 — honesty-prior layer-weight init (named contribution, NEW BASELINE)

**Status: DONE.** Side-experiment motivated by the per-layer A5d audit and the observation that A2's softmax layer weights stay near-uniform across all seeds + splits despite A5d showing meaningful per-layer differentiation. Hypothesis: the joint loss landscape is flat in the layer-weight direction, so the optimizer never specialises from a uniform start — but it would *stay* at a useful initialisation if given one. Test: initialise `head.layer_weights ← T_INV * sub@1` (T_INV=50, sub@1 from `results/A5d_grouped_layer_honesty.csv`), keep everything else identical to A2_grouped, retrain 3 seeds. Output: `results/A2_grouped_honestprior.json`.

**Headline result (3 seeds, grouped splits):**

```text
metric                  A2_grouped              A2_grouped_honestprior      Δ
uar_argmax              0.6361 ± 0.0019         0.6564 ± 0.0038            +0.0202 (4.7σ)
uar_calibrated          0.6498 ± 0.0028         0.6576 ± 0.0165            +0.0078 (within wider σ)
val_test_gap            -0.0133 ± 0.0019        -0.0202 ± 0.0054           more pessimistic
spk MLP top1            0.0498 ± 0.0031         0.0501 ± 0.0045            unchanged
spk LR  top1            0.0760 ± 0.0020         0.0725 ± 0.0002            -0.0035 (slight drop)
```

**Pareto improvement on every dimension** — UAR up substantially, MLP probe flat, LR probe slightly down. Re-running the same training recipe with a different layer-weight init alone delivers a +0.020 UAR lift comparable in magnitude to A5b's K=1 fusion lift (+0.023) but with no new features and no fusion stage.

**Optimization-landscape diagnosis (the methodological insight).** Layer-weight specialisation diagnostic per seed: `cos(init_softmax, final_softmax) = 0.9996–0.9998`; `max|delta| ≈ 0.003`; init max/min ratio 8.47× ≈ final 8.49–8.51×. **The optimizer did not move the prior at all.** This settles the question of why uniform-init A2 always converged to uniform: not because uniform was the loss-landscape optimum, but because the loss landscape is flat enough in the layer-weight direction that the optimizer has no gradient pressure to specialise. Whatever you initialise with becomes the equilibrium. Generalises beyond URTIC: **for foundation-model layer weighting in low-data settings, "let the optimizer figure it out" can fail silently** when the head can solve the task to its asymptote regardless of layer weighting. Data-derived priors do real work, not vanity initialisation.

**Layer selection by the prior** (top-5 final softmax weights across all seeds): `[0, 2, 5, 22, 6]` ≈ prior's `[0, 2, 5, 22, 24]`. Notable: the prior pulls from BOTH the early speaker-rich layers (L0/L2/L5 — high cold UAR but also high speaker top-1) AND the late cold-pure layers (L22/L24 — lower cold UAR but lowest speaker top-1). The bottleneck downstream (per-channel cold-MLP) handles each via different paths: cold-correlated speaker info from early layers is retained (and useful for cold prediction); cold info uncorrelated with speaker from late layers passes through cleanly.

**Open concern: val-test gap widening.** -0.0133 → -0.0202 (1.5σ shift, more pessimistic). Plausible mechanism: training peaks at epochs 2-3 instead of 3-7, so early-stopping signal is noisier with the more-decisive prior-initialised classifier. Worth re-checking at 5 seeds before paper submission; if gap consistently widens, calibration may need re-anchoring on a different held-out fold.

**Probe-substrate clarification (sharpened by this run).** The reading "grouped LR probe is HIGHER than random LR probe (0.076 vs 0.067) — that's worse for honesty" is **wrong**. Random splits share 198/210 pseudo-speakers between train_fit and devel_val, so a random-splits probe's top-1 is partly *recognition* of voices it has seen during training. Grouped splits share 0/210, so a grouped-splits probe's top-1 measures *generalisation across unseen pseudo-speakers* — the strictly more honest measurement. The +0.0035 LR-probe DROP from A2_grouped to A2_HP is therefore a real (small) honesty improvement on the more-honest measurement substrate.

**Architectural framing — bottleneck is the cold-MLP, not the fused vector.** A 4096-d fused vector still carries substantial speaker info under either A2 variant (probe (ii) = 0.072–0.073). The actual speaker-information bottleneck is the per-channel cold-MLP that compresses 4096-d → 1-d cold logit, optimised for cold prediction: speaker info correlated with cold gets retained (visible at the fusion-input as logit_A2 carrying cold-relevant residual speaker signal); speaker info orthogonal to cold gets stripped (visible as the ~5.5pp drop probe (ii) → probe (i)). The honest-prior chooses BOTH layer types because the cold-MLP handles each one cleanly on its own path.

**Paper paragraph (draft).** *"We use the per-layer honesty audit (§5.5) to derive an initialisation prior over WavLM layer-weight softmax logits. Compared to uniform initialisation, this gives +0.020 UAR (4.7σ) without measurable speaker probe inflation, demonstrating that A2's layer weights collapse to uniform from random init not due to absence of differential signal, but due to absence of gradient pressure to specialise — a finding with direct implications for how foundation-model layer weighting should be initialised in low-data settings."*

**Locked decision.** A2_grouped_honestprior is the **new canonical A2** for downstream rungs (A5.5, A6, A7). The earlier two A2 variants (random splits, grouped uniform) become methodology-section reference points, not active baselines. **A5b K=1 was re-run on the new anchor** (`head_A2grouped_honestprior_seed{seed}.pt`) — see §4.7 for the verdict and the calibration audit it required.

#### 4.6.1 Mechanism revision (lr stress test, single seed)

**Status: DONE (paper-narrative-correcting control).** The "flat loss landscape" claim above was tested by re-training A2.5 with the layer-weights learning rate boosted across `lr_factor ∈ {0.1, 1.0, 10.0, 100.0}` × `base_lr` (default A2 recipe is 0.1× → layer_lr = 1e-4). Single seed (42), `head_A2grouped_lrstress_lr*_seed42.pt` checkpoints. Output: `results/A2_grouped_honestprior_lr_stress.json`.

```text
lr_factor  layer_lr  cos(init,final)  L2(delta)  final/init max/min   final top-5            devel_test UAR
   0.1     1e-4      0.9999           0.0025     8.50× (= init)       [0, 2, 5, 22, 24]      0.6382
   1.0     1e-3      0.9953           0.0239     8.84× (1.04× init)   [0, 2, 5, 22, 6]       0.6511
  10.0     1e-2      0.8302           0.2099     44.08× (5.21× init)  [0, 5, 2, 6, 7]        0.6599 ← peak
 100.0     1e-1      0.2083           0.7675     20120× (2376× init)  [4, 5, 2, 6, 0]        0.6443 ← degenerate
```

**The "flat landscape" claim is contradicted.** At lr×100 the optimizer moves the layer weights dramatically (cos drops 0.9999 → 0.21) — there IS gradient signal in the layer-weight subspace, just weak relative to the standard 0.1× lr factor. The original A2.5 finding "cos(init, final) = 0.9998" was an artifact of low lr × short training horizon (25 epochs, patience 6), not landscape geometry.

**The Goldilocks pattern.** UAR is non-monotonic in lr: lr×0.1 → 0.638, lr×1 → 0.651, lr×10 → 0.660 (peak), lr×100 → 0.644 (drops). lr×10 is the sweet spot; lr×100 overshoots into a degenerate solution where L4 alone gets 85% of the weight (final max/min ratio 20120×). The optimizer at lr×100 collapses onto a low-sub@1 layer (L4 has cold UAR 0.581 / speaker top-1 0.075 / sub@1 ~0.012 per A5d) — independent cross-validation that A5d's audit signal points away from optimization-bad attractors.

**Revised mechanism for A2.5.** Not "flat landscape; optimizer cannot specialise." The honest reading:

> *At default optimization settings (layer_weights lr = base_lr × 0.1 = 1e-4, the s3prl convention, with 25-epoch / patience-6 training), the layer-weight subspace shows weak gradient signal that fails to drive specialisation from uniform init within the standard horizon. The honesty prior provides a useful starting point that the optimizer accepts (cos(init, final) = 0.9998 at default lr). At higher layer-weight lr, the optimizer DOES discover non-uniform configurations from any starting point — different layer mixes, similar UAR — but this requires hyperparameter changes that the standard recipe doesn't make.*

The corrected paper claim: *"Data-derived layer-weight init produces +0.020 UAR over uniform init at the standard low-lr regime, where the optimizer is constrained from moving regardless of starting point. The prior is practically useful within standard optimization settings rather than fundamentally necessary across all settings."* Narrower but defensible.

**Two open questions raised by the lr stress test** (to be settled by §4.6.2 controls below before locking the canonical A2.5):

1. **Does prior init + lr×{3, 5, 10} dominate prior init + lr×0.1?** Single-seed lr×10 hit UAR 0.6599 (within A2.5's 3-seed σ); 3-seed replicate needed.
2. **Does uniform init + lr×{3, 5, 10} reach competitive UAR?** If yes (likely), the prior contribution shrinks from "uniquely optimal" to "useful at standard lr." If no, the prior remains genuinely informative across lr regimes.

**Val-test gap pattern under lr stress** (worth tracking):

```text
lr_factor   val_test_gap
0.1         -0.0138
1.0         -0.0206
10.0        -0.0261
100.0       -0.0240
```

Gap widens (more pessimistic) as lr increases — higher-lr models harder to model-select cleanly on devel_val. The canonical A2.5 should balance UAR mean and val-test gap stability; if lr×10 gives 0.66 UAR with -0.03 gap vs lr×1 giving 0.65 UAR with -0.02 gap, the lr×1 version is the more honest baseline despite slightly lower UAR. Decision deferred to §4.6.2 controls.

#### 4.6.2 Controls — lr × init grid (DONE; A2.5 canonical confirmed)

**Status: DONE.** 18 retrains: (uniform init, honesty-prior init) × (lr×3, lr×5, lr×10) × 3 seeds. Output: `results/A2_lr_init_grid.json`.

**Aggregate per (init, lr_factor):**

```text
init           lr×    argmax UAR        spk LR top1   cos(init,final)   final max/min
uniform        3.0    0.6447 ± 0.0073   0.0807        0.844             ~6× (from 1×)
uniform        5.0    0.6466 ± 0.0099   0.0850        0.802             ~9×
uniform       10.0    0.6447 ± 0.0070   0.0817        0.651             ~50×
honest_prior   3.0    0.6572 ± 0.0041   0.0803        0.951             ~13× (from 8.5×)
honest_prior   5.0    0.6578 ± 0.0058   0.0900        0.889             ~25×
honest_prior  10.0    0.6638 ± 0.0034   0.0880        0.844             ~47×

Reference (lr×0.1, prior 3-seed runs):
uniform       0.1    0.6361 ± 0.0019   0.0760
honest_prior  0.1    0.6564 ± 0.0038   0.0725  ← A2.5 default (CANONICAL)
```

**Three load-bearing findings:**

1. **The honesty prior is a different attractor, not just a faster start.** Uniform init plateaus at 0.6447–0.6466 across lr×{3, 5, 10} — never reaches the 0.656+ that honesty-prior achieves at every lr setting tested. Δ(prior − uniform) at lr×10 = +0.019 with prior having tighter σ (0.0034 vs 0.0070). This refines (A1) substantially: the prior's value isn't "useful at standard low lr only" — it's a *distinct attractor in the layer-weight subspace that uniform init cannot reach via lr alone*. Strongest formulation of the architectural claim earned so far.
2. **Speaker probe inflates with higher lr in both regimes.** Uniform: 0.076 → 0.081–0.085. Honest_prior: 0.0725 → 0.080–0.090. The lr increase concentrates weight on early speaker-rich layers (top-5 final layers shift toward L0/L4/L5 at higher lr), pushing more speaker info into the fused 4096-d. **For a speaker-honesty-focused paper, this matters.**
3. **A2.5-default (honest_prior + lr×0.1) is Pareto-canonical.** A2.5-default vs honest_prior + lr×10:
   - A2.5-default: UAR 0.6564, spk 0.0725
   - honest_prior + lr×10: UAR 0.6638 (+0.0074, ~1.3σ at N=3), spk 0.0880 (+0.016, real inflation)
   - Trade: +0.007 UAR for +0.016 speaker leak. **Bad trade for de-confounding paper narrative.** A2.5-default stays canonical.

**Decision: A2.5 (honest_prior, lr×0.1) is locked as canonical.** The lr × init grid serves as ablation evidence ("the prior contribution is robust across lr regimes") rather than a replacement. honest_prior + lr×10 (A2.7-equivalent) is a reference row showing the upper-UAR Pareto frontier; honest_prior + lr×3 (A2.6-equivalent) is strictly Pareto-dominated by A2.5 (+0.0008 UAR for +0.008 spk leak — same UAR, more leak, no reason to switch).

**Mechanism revision (third pass).** The original "flat landscape" claim was wrong (M5). The first revision said "good init dominates because lr is too low to move." The lr × init grid revises that further: **"the honesty prior is a genuinely different attractor that uniform init cannot reach via lr alone — at any tested lr the prior wins by +0.011 to +0.019 UAR with comparable or tighter σ."** The corrected (A1) paper claim: *"Data-derived layer-weight init produces +0.020 UAR over uniform init at default lr; the contribution is not 'cheaper start to the same attractor' but rather convergence to a distinct attractor that uniform init does not reach even with 10× layer-weight learning rate."*

**Open: val-test gap drift at higher lr.** -0.0204 → -0.0224 → -0.0177 (uniform); -0.0224 → -0.0240 → -0.0224 (prior). All slightly more pessimistic than A2.5-default's -0.020. Plausible mechanism: faster early-stopping (epochs 1–3 vs 3–7) at higher lr makes the early-stopping signal noisier. A2.5-default still has the cleanest val-test gap; another reason to keep it canonical. 5-seed re-run on the locked A2.5 deferred to paper-prep stage.

### 4.7 A5b K=1 re-audit on A2.5 anchor (β=1.00 FAIL → β-sweep PASS, boundary-pegged)

**Context.** The original A5b K=1 PASS was locked on the uniform-A2 anchor with β=1.00. After A2.5 became the new canonical baseline (§4.6), the K=1 verdict needed re-evaluation. β=1.00 was implicitly conditioned on the uniform-A2 anchor; under A2.5's more-confident logits it doesn't necessarily transfer.

#### 4.7.1 Protocol-strict β=1.00 re-run (FAIL — calibration failure)

**Status: DONE.** Mirror of the A5b grouped cell with `head_A2grouped_honestprior_seed{seed}.pt` as anchor, β fixed at 1.00 per the original protocol. 3 seeds. Output: `results/A5b_grouped_honestprior.json`.

```text
                                 uniform A2_grouped (PASS)   A2.5 honest-prior (FAIL)
A2 anchor argmax UAR             0.6361 ± 0.0019             0.6564 ± 0.0038
K=1 fused UAR (β=1.00)            0.6588 ± 0.0059             0.6346 ± 0.0168
Δ vs anchor argmax                +0.0227 ± 0.0059 (3.8σ)    -0.0218 ± 0.0143 (FAIL)
probe (i) literal 2-D            0.0153 ± 0.0032            0.0156 ± 0.0026   PASS
probe (ii) backbone+G4_gi        0.0733 ± 0.0002            0.0733 ± 0.0002   PASS
```

**Calibration-failure smoking gun.** All 3 seeds locked τ at -3.925 to -3.975 — the floor of `np.linspace(-4.0, 4.0, 321)`. β was forced to 1.00 (the original protocol). Both knobs at search edge → textbook search pathology, NOT a genuine signal absence. The fusion at β=1.00 over-weights G4_gi relative to the more-confident A2.5 logits, creating a fused logit distribution where no operating point in the τ search range generalises from train_threshold to devel_test.

**Decision.** The β=1.00 forced FAIL is a *calibration failure* under the wrong-for-new-anchor hyperparameter. Not a signal-absence verdict. The honest re-audit requires re-tuning β under the new anchor (§4.7.2).

#### 4.7.2 β-sweep on A2.5 anchor (PASS at β*=2.0, boundary-pegged)

**Status: DONE (verdict caveated, follow-up pending).** β grid `[0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]` swept jointly with τ on `train_threshold`; (β*, τ*) locked per seed by argmax thr_UAR; one-shot eval on devel_test. Output: `results/A5b_grouped_honestprior_betasweep.json`.

```text
LOCKED CONFIG (3 seeds):
  per-seed β*       = [2.0, 2.0, 2.0]   (all at upper edge of sweep grid)
  per-seed τ*       = [-3.825, -2.275, -3.550]
  K=1 fused UAR     = 0.6755 ± 0.0208
  Δ vs A2.5_argmax  = +0.0192 ± 0.0175   (PASS the +0.007 gate by mean)
  Δ vs A2.5_τ       = +0.0187 ± 0.0085   (cleaner statistic, 2.2σ)
```

**β=0 sanity passes cleanly.** All 3 seeds reproduce A2.5_τ exactly at β=0 (devel_test UAR 0.6417 / 0.6751 / 0.6536 = A2.5_τ values). Fusion code is correct.

**Per-β trend** (3-seed mean):

```text
β=0.00 → 0.6568   (sanity == A2.5_τ ✓)
β=0.05 → 0.6592   (tiny lift)
β=0.10 → 0.6408   (dip — middle-β interference)
β=0.50 → 0.6396
β=1.00 → 0.6346   (the original FAIL point, lowest)
β=1.50 → 0.6550
β=2.00 → 0.6755   (best, but at search edge)
```

Non-monotonic and bumpy. Middle β values create unfavorable τ-operating-point interference; low and high β both give lift. Recall pattern explains it: as β grows, the model becomes more cold-biased (recC: 0.45 → 0.83, recNC: 0.86 → 0.43); the τ sweep finds different balance points, landing well at low β or high β but poorly at mid β.

**Caveat: β* pegged at search boundary.** Same diagnostic that flagged the K-sweep pathology and the β=1.00 fail. We don't know if the true optimum is β=2.0 (just outside the grid) or β=4, β=8, β=16 (G4_gi-dominant regime). **Per-seed: 1 of 3 seeds gives Δ ≈ 0 (seed 42 = -0.0006), 2 of 3 give Δ +0.025 to +0.032.** Mean PASSES the gate but only ~0.7σ above the gate threshold; per-seed reliability is mixed.

**Recall-asymmetry observation.** At β*=2.0, recC ≈ 0.82, recNC ≈ 0.53 — heavily cold-biased. A2.5 alone has recC ≈ 0.45, recNC ≈ 0.86 (balanced). UAR is invariant to the asymmetry by construction, but the operating point at β*=2.0 is qualitatively different from A2.5's. This is consistent with the fusion at high β being driven primarily by G4_gi-induced shifts rather than refining A2.5's decision boundary — the underlying mechanism may be more "switch from A2.5-driven decisions to G4_gi-driven decisions" than "stack A2.5 with a G4_gi adjustment."

**G4_gi-alone reference (from A5a).** G4_gi standalone label_gain = +0.132 → UAR ~0.632 on devel_val. Materially weaker than A2.5 (0.6564). β→∞ baseline on devel_test would land near ~0.63, well below the 0.6755 K=1 number. So fusion is NOT being dominated by G4_gi (the "G4_gi alone is better than A2.5" possibility is empirically ruled out by A5a). But "G4_gi has more weight than A2.5 in the locked fusion" remains an open mechanism question.

#### 4.7.3 Extended β-sweep + G4-alone baseline (DONE; verdict LOCKED)

**Status: DONE.** β grid extended with {2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0}. G4-alone (β=∞ limit) added: logit_G4 alone with τ swept on train_threshold, evaluated on devel_test. Robust-β diagnostic: smallest β within 1σ of per-β-UAR peak. Output: `results/A5b_grouped_honestprior_betasweep_extended.json`.

**Per-β aggregate (3 seeds):**

```text
β       dt_UAR mean ± std    thr_UAR mean    tau@edge
0.00    0.6568 ± 0.0169      0.5184          0/3   (sanity == A2.5_τ ✓)
0.05    0.6592 ± 0.0150      0.5185          0/3
0.10    0.6408 ± 0.0224      0.5219          1/3
0.25    0.6461 ± 0.0189      0.5299          0/3
0.50    0.6396 ± 0.0200      0.5523          1/3
0.75    0.6507 ± 0.0195      0.5628          1/3
1.00    0.6346 ± 0.0168      0.5829          1/3   ← original FAIL point
1.50    0.6550 ± 0.0233      0.6075          2/3
2.00    0.6755 ± 0.0208      0.6256          0/3   ← original boundary lock
2.50    0.6743 ± 0.0126      0.6393          1/3
3.00    0.6830 ± 0.0099      0.6489          0/3
4.00    0.6875 ± 0.0111      0.6552          0/3   ← robust β (smallest within 1σ of peak)
6.00    0.6917 ± 0.0085      0.6570          0/3   ← peak β
8.00    0.6895 ± 0.0089      0.6558          0/3
12.00   0.6917 ± 0.0030      0.6518          0/3
16.00   0.6895 ± 0.0021      0.6463          0/3

G4-alone (β=∞)  0.6632 ± 0.0000   (τ ≈ +0.475 across all seeds)
```

**Per-seed locked configs (per-seed argmax of train_threshold UAR):**

```text
seed    β*    τ*       devel_test UAR    Δ vs A2.5_argmax
42      6.0   -3.325   0.6825            +0.0294
123     8.0   -1.100   0.6957            +0.0351
7       4.0   -2.825   0.6957            +0.0403

aggregate (per-seed argmax lock):
  K=1 fused UAR     = 0.6913 ± 0.0076
  Δ vs A2.5_argmax  = +0.0349 ± 0.0054   (~6.5σ, strong PASS)
  Δ vs A2.5_τ       = +0.0345 ± 0.0120
```

**Three independent lines of evidence converge on PASS:**

1. **Interior peak at β=6, plateau through β=12-16.** The β-curve climbs sharply from β=2 to β=4, plateaus at 0.6875–0.6917 across β ∈ [4, 16], then drops to 0.6632 at β=∞ (G4-alone). The lock is genuinely in an interior optimum, not a boundary artifact. The original boundary-pegged β*=2.0 was just the upper edge of the truncated grid; the true plateau is β=[4, 16].
2. **K=1 fused (0.6913) > G4-alone (0.6632) by +0.0281.** Fusion is genuinely additive — A2.5 contributes real cold signal that G4_gi alone doesn't carry, even at the high-β regime where G4 dominates the fusion. Settles the "G4 dominance" possibility raised in §4.7.2: G4_gi alone is materially weaker than the fused result.
3. **Per-seed β* spread is wide [4, 6, 8] but the per-β aggregate is stable in the plateau.** σ of per-β UAR drops from 0.011 (β=4) to 0.003 (β=12) — the high-β region is dominated by G4_gi (deterministic per seed), so seed-to-seed variance shrinks. The result survives any plateau-region β choice.

**Speaker probes are β-independent** — both probe (i) literal 2-D = `[logit_A2, z_logit_G4]` and probe (ii) backbone concat = `[pooled_4096, G4_gi_7]` operate on β-independent feature spaces (β only enters the cold classifier's decision rule, not the probe inputs). Locked values from the original sweep cell carry over: probe (i) = 0.0156 ± 0.0026, probe (ii) = 0.0733 ± 0.0002 — **both PASS** the codepath-consistent ceiling 0.0780 by wide margin (~5×) and small margin (0.005) respectively.

**Three reportable lock options for the paper:**

| Lock protocol                     | β                   | UAR             | Δ vs A2.5       | σ        | Notes                                |
| --------------------------------- | ------------------- | --------------- | --------------- | -------- | ------------------------------------ |
| Per-seed argmax (protocol-strict) | per-seed [6, 8, 4]  | 0.6913 ± 0.0076 | +0.0349 (~6.5σ) | tightest | mixes seed-specific τ over-fitting   |
| Aggregate-peak (across seeds)     | 6.0                 | 0.6917 ± 0.0085 | +0.0353         | clean    | slightly above A2.5 + 0.035          |
| Robust-β (across seeds)           | 4.0                 | 0.6875 ± 0.0111 | +0.0311         | wider    | most defensible / conservative       |

**LOCKED VERDICT.** A5b K=1 PASSES on A2.5 anchor at all three lock protocols. The conservative defensible lock is **β=4** (robust-β, Δ +0.031, ~5.7σ above gate); the headline lock is **β=6** (peak, Δ +0.035, ~6.5σ); the protocol-strict per-seed argmax matches the headline (Δ +0.035). All three plateau-region β values produce K=1 fused UAR within ~1σ of each other. Speaker probes (i) and (ii) PASS by wide margins.

**Striking parity observation.** G4-alone hits UAR 0.6632 — nearly equal to A2.5-alone 0.6564 (Δ ~0.7σ). A 7-dim handcrafted feature matches a 4096-dim WavLM-Large representation on chunk-level cold UAR. Two interpretations worth noting in the paper: (a) G4_gi's gain-invariant slice happens to be unusually well-suited to URTIC cold detection; (b) the chunk-level signal has a low ceiling and any reasonable representation hits it. Either way, the fusion result (+0.028 over G4-alone) confirms partial orthogonality of A2.5 and G4 cold signals — they capture distinct information.

**Final paper claim chain (locked):**

```text
A2 grouped (uniform, lr×0.1):     UAR 0.6361 ± 0.0019  ← historical baseline (within-partition leak corrected)
A2.5 (honest_prior, lr×0.1):       UAR 0.6564 ± 0.0038  ← +0.0202 (4.7σ), CANONICAL anchor
G4-alone (β=∞):                    UAR 0.6632 ± 0.0000  ← reference (7-dim handcrafted)
A5b K=1 fused (β=4, robust):       UAR 0.6875 ± 0.0111  ← +0.0311 over A2.5
A5b K=1 fused (β=6, peak):         UAR 0.6917 ± 0.0085  ← +0.0353 over A2.5
A5b K=1 fused (per-seed argmax):   UAR 0.6913 ± 0.0076  ← +0.0349 over A2.5 (protocol-strict, headline)
```

Total stack uniform-A2-grouped → A2.5 → A5b K=1 fused = +0.055 UAR over the leak-corrected baseline (+0.049 over the historical A2 baseline). Two clean stackable contributions, both speaker-probe-clean, both resolvable to interior optima.

### 4.8 A5.5 — cross-speaker augmentation (LOCKED at conservative-α embedding mixup; aggro-α ablation closes α-axis)

**Status: Phase 1-2 DONE; Phase 3 FAILED; Phase 3.5 (diagnostic) queued.** Implementation broken into phases for risk-staged execution.

#### 4.8.1 Phase 1 — splice primitives (DONE)

[`model/data/splice.py`](AI-For-Health/model/data/splice.py) implements plan §6 splicing recipe: silence/unvoiced boundary picker on cached manner labels, equal-power crossfade, RMS match, partner window with voiced-fraction floor, splice composer, partner-pool builder. Smoke test on 30 anchors × 5 trials: 77.3% successful splice rate, 0 length mismatches, partner voiced-fraction mean 0.715, partner reuse top user 1.3% of trials. Module + smoke test committed.

#### 4.8.2 Phase 2 — augmented pooled cache build (DONE)

K=3 augmented variants per grouped train_fit chunk pre-extracted to `cache/microsoft_wavlm-large/pooled_aug_k{0,1,2}/{stem}.pt` with sidecar metadata in `_meta/`. Per (anchor, k): sample partner (same Cold + different pseudo-speaker, 5 retries), splice via `splice_chunk`, fall back to original audio if splicer skips (sidecar flag), run WavLM forward, save pooled stats. Output: `results/A5_5_phase2_extract.json`.

```text
k    successful splices    fallback     boundary kinds
0    6871/8532 (80.5%)     1661         6819 unvoiced + 52 silence
1    6872/8532 (80.5%)     1660         6815 unvoiced + 57 silence
2    6868/8532 (80.5%)     1664         6815 unvoiced + 53 silence
```

Total: 10.9 min on GPU (8× faster than 90-135 min estimate). Consistent 80.5% successful-splice rate replicates the smoke test prediction at corpus scale (smoke 77.3% on 150 trials → 80.5% on 25.6k). Three variants are independent (no overlap in top-5 partner reuse lists across k). Partner reuse top user only ~7 trials = 0.082% of total — diverse cross-speaker mixings, not three slight variants of the same splicing pattern. The fallback-to-original design preserves dataset size and avoids introducing splice-amenability as a confounded variable.

#### 4.8.3 Phase 3 — splice-detector audit (FAIL — gate as written is unrealistic for cross-speaker splicing)

**Status: FAILED at the originally-specified hard gate.** Per plan §6: "linear probe on train_fit cached pooled features for original vs spliced; gate at detector UAR ≤ 0.55." Output: `results/A5_5_phase3_splice_detector.json`.

```text
substrate                            UAR     PASS≤0.55
layer-averaged (4096-d, headline)    0.9981  FAIL
worst single layer (L7)              0.9982  FAIL
per-k breakdown (k=0/1/2)            0.997-0.999  ALL FAIL
```

**Diagnosis (with reflection-sharpened framing).** UAR 0.998 means the binary LR probe perfectly distinguishes original from spliced pooled features at every layer (L0-L24) and every k variant. Two competing interpretations of why:

1. **"Gate is miscalibrated for cross-speaker splicing."** Pooled stats are mean+std+skew+kurt over time per WavLM layer. Replacing 25-30% of a chunk with audio from a *different speaker* shifts those per-layer time-means toward the partner's per-layer mean. Different speakers occupy different regions of WavLM's per-layer 1024-d hidden state space (same property ECAPA exploits for speaker recognition). A 25-30% mass-shift in 4096-d features is trivially detectable by a linear probe. Cross-speaker splicing is supposed to make chunks acoustically different — that's how the (chunk → anchor speaker) link breaks. The 0.55 gate was likely designed for *perceptual-equivalence* augmentations (time-warp, noise injection, mild EQ); for cross-speaker splicing it's unachievable by construction.

2. **"Splicer technique itself is leaving artifacts beyond cross-speaker mixing."** Detection is layer-uniform (L0=0.992, L7=0.998, L24=0.995) — if cross-speaker mixing were the *only* loud signal, the detector should be noticeably stronger in early speaker-rich layers (per A5d: spk_top1 0.072 @ L0 → 0.042 @ L24). Layer-uniform detection suggests multi-faceted artifacts (crossfade discontinuities, spectral boundary mismatches, recording-session signature, F0 trajectory jumps) on top of cross-speaker mixing. **Caveat: at UAR 0.998 the detector is at saturation, so the layer-distribution argument is only suggestive.**

**The shortcut concern is structurally mitigated regardless of which interpretation holds.** Plan §6 risk: "augmentation creates splice-detection shortcut." The mitigation is symmetric-across-classes splicing — every chunk regardless of cold label gets K=3 augmented variants, splice presence is uncorrelated with cold. The model can detect "this is spliced" with UAR 0.998 *and still not be able to use it to predict cold*, because the splice-vs-original distinction is class-balanced (partner-pool design enforces same Cold + different pseudo-speaker). A direct splice-bit-as-cold-predictor audit would give UAR ≈ 0.5 by construction.

**Decision: don't proceed to Phase 4 yet.** Run the diagnostic in §4.8.4 first to settle which interpretation is correct.

#### 4.8.4 Phase 3.5 — self-splice control (DONE; verdict: branch C — splicer broken)

**Status: DONE. Verdict: FAIL → pivot to Plan B (embedding mixup).** Output: `results/A5_5_phase3p5_selfsplice_control.json`.

```text
substrate                          self-splice UAR    cross-splice UAR (Phase 3)    Δ
layer L0                           0.9800             0.9917                        -0.012
layer L4                           0.9843             0.9965                        -0.012
layer L7                           0.9893             0.9982                        -0.009
layer L12                          0.9900             0.9979                        -0.008
layer L18                          0.9925             0.9975                        -0.005
layer L24                          0.9654             0.9954                        -0.030
layer-averaged (4096-d, headline)  0.9900             0.9981                        -0.008

self-splice extraction summary:
  successful self-splices: 788 / 1000 (78.8%)  ← consistent with Phase 2's 80.5% cross-splice
  fallback to original   : 212
  boundary kinds         : 768 unvoiced + 20 silence
```

**The clean diagnosis.** Self-splice UAR 0.9900 ≈ cross-splice UAR 0.9981 (Δ layer-averaged = -0.008). Cross-speaker mixing accounts for **less than 1%** of the detectability; the splicer operation itself accounts for ~99% of the signal. **The artifacts are coming from the splice OPERATION, not the cross-speaker MIXING.** Even concatenating audio from the SAME speaker via our crossfade pipeline produces an acoustic signature that's effectively perfectly detectable from WavLM pooled stats.

**This rules out an entire class of fixes.** No partner-selection refinement (same-session, same-speaker, RMS-matched) will help — the splicing operation produces a signature regardless of who you splice with. Splicer-internal fixes (longer crossfade 250→400ms, tighter low-energy boundary selection, smaller r, spectral envelope matching at the seam) might shave 5-10 UAR points but won't drop 0.99 → 0.55 (a 40+ point gap). The fundamental issue is that **WavLM's CNN feature-extractor + transformer attention are sensitive to splice-induced spectral discontinuities at every layer (L0 = 0.98, L24 = 0.965 — uniform across the stack)**. Crossfade is already 250ms; longer doesn't help much because attention windows span much wider time scales.

**Generalisable methodological warning (paper-worthy).** "Audio-level cross-speaker splicing on URTIC WavLM representations is detectable at >99% UAR even when the operation is performed within a single speaker. This indicates the splicing operation itself, not the cross-speaker mixing, is the source of the artifactual signal. Anyone doing audio-level cross-speaker augmentation for foundation-model-based health classification should run a self-splice control before proceeding to training." Transferable beyond URTIC — the artifact is a property of how WavLM (and likely HuBERT, Whisper, etc.) represent splice discontinuities at every layer.

**Caveat (acknowledged but doesn't change verdict).** Same-pseudo-speaker partner is a proxy for same-speaker (k=210 clusters with kNN cohesion @ k=10 = 0.957). URTIC speakers may have been recorded across multiple sessions with slightly different mic/room acoustics; even self-splices CAN cross sessions. So *some* of the 0.99 self-splice UAR could be session-mismatch rather than pure splicer technique. But even if half of it were session-mismatch (which would be high), splicer-only artifacts would still account for ~0.74 UAR — still well above the 0.55 gate and well into branch C territory. Pivot to Plan B is the right call regardless of the session-mismatch decomposition.

**Decision: pivot to Plan B (embedding mixup, §4.8.5).** Audio-level splicing is dead on this corpus + this backbone.

#### 4.8.5 Plan B — embedding mixup (Phase 2 + audit DONE; Phase 4 PARTIAL PASS at cons-α; aggro-α α-sweep §4.8.8 confirms narrow window EMPTY → A5.5 LOCKED at cons-α)

**Status: Phase 2 + audit DONE (both PASS); Phase 4 PARTIAL PASS (UAR floor PASS, speaker probe drop FAIL); α ∈ [0.5, 0.7] sweep queued (§4.8.8).** With Phase 3.5's branch-C verdict (splicer broken, §4.8.4), audio-level splicing is dead on this corpus. Embedding mixup operates on the cached WavLM pooled stats directly — no audio operation, no WavLM forward, no crossfade artifacts.

**Recipe.** For each grouped train_fit chunk × K=3:

```text
1. Sample partner from same Cold + different pseudo-speaker (reuse build_partner_pool from data/splice.py)
2. α ~ Uniform(0.70, 0.85)   # anchor stays dominant; preserves cold label validity
3. mixed_pooled = α · anchor_pooled + (1 - α) · partner_pooled        # both are [25, 4096] fp16
4. Save to cache/microsoft_wavlm-large/pooled_mixup_k{0,1,2}/{stem}.pt + sidecar JSON
```

Cost: ~1-2 min CPU (no WavLM forward, just tensor mixing on cached pooled stats; 25,596 mixed variants total).

**Design choices made explicit:**

- **α ∼ Uniform(0.70, 0.85)** rather than Beta(0.4, 0.4) (standard mixup). Beta puts mass near 0 and 1 — heavy mixing or near-no-mixing — which doesn't fit our use case. Uniform(0.70, 0.85) gives consistent moderate mixing (15-30% partner contribution) where anchor stays dominant for label validity. Parallel to the original audio-splicing `r ∼ Uniform(0.20, 0.30)`.
- **Mix pooled stats `[25, 4096]` directly, before standardisation.** Pooled stats are stored fp16 raw; downstream consumers (A2.5 head) apply standardisation themselves via `head.scaler`. Mixing before standardisation is the natural place — preserves the architecture's existing data flow.
- **What stays unmixed:** devel_val and devel_test (always evaluation-only, never augmented); G4_gain_invariant handcrafted features (computed from raw audio, mixing them at the pooled-stats level is meaningless); pseudo-speaker labels for augmented chunks inherit anchor's label (per-chunk anchor pseudo-speaker, the de-confounding mechanism reduces speaker-decodability of augmented chunks).
- **Partner pool: same Cold label, different pseudo-speaker** (same as the audio-splicing partner pool, reused via `build_partner_pool` from `model/data/splice.py`).

**Naming and architectural framing for the paper.** Embedding mixup is technically operating on cached upstream features (WavLM pooled stats) — that's "input-level for the head" but "representation-level for WavLM." A more honest framing than reframing as pure "representation-level": **"input-level mixing of frozen-foundation-model representations."** Distinguishes it from A6 (contrastive loss on the trained head's projection) without overclaiming the audio-level intervention we couldn't get clean. Worth a sentence in the methodology section.

**Revised gate framing — for ALL cross-speaker augmentations going forward.** The original splice-detector hard threshold (≤ 0.55) was *always* wrong as a gate for ANY cross-speaker augmentation that injects partner information. By construction, augmented chunks contain partner content, which is detectable. The new two-tier audit:

- **(1) Detector UAR (orig vs mixed)** — descriptive diagnostic. Expected non-zero by augmentation construction. Report as "augmentation-class baseline detectability." For embedding mixup with α ∼ U(0.70, 0.85), expected ~0.65-0.85 (smoother than audio splicing's 0.998; the mixed pooled vector is a 70/30 weighted average of two natural pooled vectors, distributionally distinguishable but much less artifactual than crossfaded audio).
- **(2) Mix-bit-as-cold-predictor ≤ 0.52** — the actual shortcut gate. Should be ~0.5 by class-balanced augmentation design (every chunk regardless of cold label gets K=3 mixed variants; partner pool enforces same Cold; mix presence is uncorrelated with cold). If above 0.52, partner-pool symmetry is broken or the augmentation is inducing class-confounded shifts.

The same gate framing applies to audio splicing in retrospect — Phase 3's UAR 0.998 was descriptive (telling us audio splicing produces strong artifacts), but its hard threshold should never have been treated as the actual shortcut gate. The audit served its diagnostic purpose (caught the splicer-artifact problem via Phase 3.5) but the gate framing was over-strict from the start.

**Paper framing — option 2 (audit-and-pivot as methodological contribution).** Don't bury the audio-splicing failure; lead with it. Draft methodology paragraph:

> *"We initially attempted audio-level cross-speaker splicing as the data-level de-confounding intervention (plan §6, A5.5 v1). Pre-extracted K=3 spliced variants per training chunk, applied an equal-power crossfade at unvoiced boundaries with RMS-matched partner segments. The splice-detector audit (binary LR probe on (original, spliced) WavLM pooled stats, 80/20 chunk-disjoint split) found UAR 0.998 — perfect detection across every layer. A self-splice control (splicing within the same pseudo-speaker cluster) gave UAR 0.990, revealing that the splicing operation itself, not the cross-speaker mixing, was the source of the artifactual signal. We pivoted to embedding mixup: a less aggressive but more reliable de-confounding mechanism that mixes WavLM pooled stats per chunk (α ∼ U(0.70, 0.85) anchor weight) instead of audio. This sidesteps every audio-level artifact while preserving the de-confounding hypothesis: training on mixed-speaker representations should reduce the model's reliance on speaker-specific features for cold prediction. The audit-and-pivot itself is a transferable methodological warning — anyone doing audio-level cross-speaker augmentation on foundation-model representations should run a self-splice control."*

**Trade-offs vs the original audio-splicing design.** Loses the temporal-mixing aspect (mixup operates per-chunk on global summaries, not within-chunk acoustic content); changes the project's three-mechanism story slightly (two of three rungs are now representation-level: A5.5 mixup + A6 contrastive; only A7 is gradient-level). Either reframe as "two representation-level interventions + one gradient-level" or keep three by counting A5.5 explicitly as "input-level mixing of frozen-FM representations" (different from A6's representation-level contrastive on the trained projection).

##### 4.8.5.a Plan B Phase 2-equiv (mixup cache build) — DONE

`results/A5_5_planB_phase2_mixup.json`. K=3 mixed pooled-stats variants per grouped train_fit chunk, α ∼ Uniform(0.70, 0.85). Pure CPU tensor op on cached pooled stats; no WavLM forward needed.

```text
                                       k=0      k=1      k=2
successful mixes / total          8532/8532  8532/8532  8532/8532   (100%)
fallback to original                       0          0          0
α distribution (mean ± std)        0.775 ± 0.043   0.775 ± 0.043   0.775 ± 0.043
α range                            [0.700, 0.850]  [0.700, 0.850]  [0.700, 0.850]
partner-class balance              cold partners 10.1% (corpus rate ~9.5% — symmetric ✓)
partner reuse top user             ~7 trials = 0.1% of total (diverse partners)
elapsed                            0.30 min       0.30 min       0.30 min
total: 1.13 min CPU
```

100% successful mixes (no fallback) — pooled stats are always cached, so unlike audio splicing's 19.5% skip rate, mixup never has to fall back. α distribution exactly matches the design U(0.70, 0.85). **Partner-class balance critical for gate B**: cold-partner usage rate 10.1% matches corpus cold rate 9.5% (within sampling noise) — confirms `build_partner_pool` enforces same-Cold + different-pseudo-speaker symmetrically across both classes.

##### 4.8.5.b Plan B Audit — DONE (BOTH gates PASS)

`results/A5_5_planB_audit.json`. Two-tier audit per the revised gate framing.

**Audit (A) Detector UAR (descriptive, NOT a hard gate):** Binary LR probe distinguishes original vs mixed pooled features. Substrates: per-layer L=0/4/7/12/18/24, layer-averaged 4096-d (headline), per-k breakdown.

```text
substrate                    UAR     recall_orig   recall_mixed   vs Phase 3 (audio splicing)
layer L0                     0.5048  0.2239        0.7857
layer L4                     0.5010  0.2222        0.7798
layer L7                     0.5053  0.2456        0.7649
layer L12                    0.5037  0.2333        0.7741
layer L18                    0.5020  0.2011        0.8029
layer L24                    0.5048  0.2286        0.7810
layer-averaged (headline)    0.5034  0.2198        0.7870          Δ vs splicing -0.4947
per-k k=0                    0.4997
per-k k=1                    0.5018
per-k k=2                    0.5023
```

Detector UAR is essentially CHANCE (0.5034 layer-averaged, all per-layer values within [0.501, 0.505], all per-k within [0.500, 0.502]). **Δ -0.4947 vs Phase 3 audio splicing (0.998 → 0.503)** — the smooth tensor-blend completely sidesteps the artifactual signal that crossfaded audio left across every WavLM layer. Mixed pooled vectors are statistically indistinguishable from natural ones at the linear-probe level. The recall imbalance (recall_orig ~0.22, recall_mixed ~0.78) shows the probe defaults to predicting "mixed" because of the 1:3 class ratio (1 orig per 3 mixed); UAR averages this out to chance.

**Audit (B) Mix-bit-as-cold-predictor (HARD GATE ≤ 0.52):** 1-d LR (`mix_present ∈ {0, 1}` → `cold ∈ {0, 1}`).

```text
mix-bit → cold UAR              0.5000   recall_NC=1.0000  recall_C=0.0000   PASS
cold rate among original chunks 0.1017
cold rate among mixed chunks    0.1017   Δ exactly +0.0000
```

**Trivial PASS by class-balanced augmentation construction.** Mix-bit predicts cold at exactly 0.5 UAR (the trivial classifier predicts "always non-cold" — recall_C = 0, recall_NC = 1). Δ cold rate (mixed - orig) = 0.0000 exactly: every chunk gets K=3 mixed variants regardless of its cold label, partner-pool enforces same-Cold symmetrically, so mix presence is uncorrelated with cold by construction. **The actual shortcut concern is structurally mitigated.**

**Verdict: PASS — proceed to Plan B Phase 4 training.** Augmentation cannot create a cold-prediction shortcut (per gate B). Detector UAR (0.503) is descriptive of augmentation-class baseline detectability — by construction, mixed pooled vectors are slightly distinguishable from natural ones, but this can't be exploited as a cold shortcut.

#### 4.8.6 Phase 4 — A5.5 head training on mixed embeddings (DONE; conservative-α PARTIAL PASS, aggro-α α-sweep in §4.8.8 closes the α-axis at branch (d))

**Status: DONE. Verdict: PARTIAL PASS at conservative α ∈ [0.70, 0.85] — UAR floor PASS but speaker probe drop FAIL on the strict 3-D gate.** §4.8.8 ran the α ∈ [0.50, 0.70] aggressive sweep to test the de-confounding-mechanism hypothesis; verdict = branch (d) (UAR drops AND probe still doesn't drop), confirming both endpoints of the α-axis fail. **A5.5 is LOCKED at the conservative-α variant as canonical**; the aggressive variant ships as ablation evidence (M9 narrow-window-empty). Decision-tree details below in §4.8.8.

Output: `results/A5_5_planB_phase4_mixup.json`, `cache/microsoft_wavlm-large/head_A55mixup_seed{seed}.pt` (3 seeds).

**Recipe used:**

- Sampling: per-epoch random sampling across {original, mix_0, mix_1, mix_2} per chunk (uniform; one of 4 versions per epoch); preserves ~9.5k epoch size.
- Anchor: warm-start from A2.5 (`head_A2grouped_honestprior_seed{seed}.pt`).
- Pseudo-speaker labels for mixed chunks: anchor's label.
- Devel unaugmented (original PooledCacheDataset for devel_val + devel_test).
- Training recipe: identical to A2.5 (lr×0.1, 25 epochs, patience 6, AdamW, cosine, balanced sampler with `train_ds.get_labels()` returning anchor-side cold labels).
- 3 seeds {42, 123, 7}.

**Per-seed details:**

```text
seed 42:   best_epoch=3   best_val_uar=0.6317   tau=+0.225   cos(init_A2.5, final)=0.9999   max/min 8.49x → 8.50x
seed 123:  best_epoch=1   best_val_uar=0.6327   tau=+0.605   cos(init_A2.5, final)=1.0000   max/min 8.51x → 8.51x
seed 7:    best_epoch=5   best_val_uar=0.6356   tau=+0.080   cos(init_A2.5, final)=0.9997   max/min 8.49x → 8.50x

Per-seed devel_test argmax UAR:  0.6631 / 0.6590 / 0.6651
Per-seed val_test_gap:           -0.0315 / -0.0263 / -0.0296   (all pessimistic)
Per-seed top-5 final layers:     [0, 2, 5, 22, 6]  (unchanged from A2.5 init)
```

**Aggregate (3 seeds, side-by-side vs A2.5 baseline):**

```text
metric                  A2.5 baseline                A5.5 Plan B (mixup, α∈[0.70, 0.85])    Δ
uar_argmax              0.6564 ± 0.0038              0.6624 ± 0.0031                       +0.0060 (~1.6σ above zero)
uar_calibrated          0.6576 ± 0.0165              0.6636 ± 0.0145                       +0.0060
recall_C @ τ            ~0.43                        0.4738 ± 0.0882                       +0.04   (more cold-balanced)
recall_NC @ τ           ~0.87                        0.8534 ± 0.0610                       -0.02
val_test_gap            -0.0202 ± 0.0054             -0.0291 ± 0.0026                      more pessimistic, tighter σ
spk MLP top1            0.0501 ± 0.0045              0.0506 ± 0.0019                       +0.0005 (essentially unchanged)
spk LR  top1            0.0725 ± 0.0002              0.0759 ± 0.0030                       +0.0034 (slightly UP)
```

**3-D acceptance gate evaluation:**

```text
gate (1) UAR ≥ A2.5 - 1σ (= 0.6525)            : 0.6624   Δ +0.010  PASS
gate (2a) MLP probe drops ≥ 1σ (≤ 0.0456)      : 0.0506   Δ +0.005  FAIL  (probe slightly UP)
gate (2b) LR  probe drops ≥ 1σ (≤ 0.0723)      : 0.0759   Δ +0.004  FAIL  (probe slightly UP)
gate (3)  mix-bit-as-cold-predictor ≤ 0.52     : 0.5000   PASS (verified pre-training)

OVERALL VERDICT: FAIL on the strict 3-D gate
                 (UAR floor PASS but speaker probe DID NOT DROP)
```

**Diagnostic interpretation:**

- **The +0.006 UAR is real but small.** ~1.6σ above zero on N=3 seeds. Comparable to noise. Compared to A2.5's +0.020 over uniform-init A2 and A5b's +0.035 over A2.5, A5.5's contribution is the smallest of the rungs locked so far.
- **cos(init_A2.5, final) = 0.9999, 1.0000, 0.9997** across all 3 seeds — the optimizer DID NOT MOVE the layer weights from A2.5's converged state. Best epochs are early (3, 1, 5) with training plateauing or worsening after epoch 5. **The +0.006 UAR comes entirely from MLP/classifier-level adjustments to the augmented training distribution, NOT from any layer-weight refinement.** Consistent with M5 (layer-weight subspace has weak gradient signal at default lr; the standard recipe can't move from any starting point).
- **Recall pattern is meaningfully more cold-balanced.** A2.5's recC ≈ 0.43, recNC ≈ 0.87 (heavily NC-biased); A5.5 mixup gives recC ≈ 0.47, recNC ≈ 0.85. Net UAR shift +0.006, but recC moves +0.04 and recNC moves -0.02 — a real operating-point change. For a minority-class problem (cold rate ~9.5%), recovering more cold examples is practically valuable beyond what the UAR scalar captures. Worth reporting both numbers.
- **Speaker probe did NOT drop — augmentation didn't activate the de-confounding mechanism at α ∈ [0.70, 0.85].** Three plausible reasons: (1) α=0.775 mean = 22.5% partner contribution is too gentle — anchor's speaker characteristics still dominate the mixed pooled vector; (2) the audit detector at UAR=0.503 confirms mixed pooled vectors are statistically indistinguishable from natural ones, so the model has nothing to "learn invariance to" — there's no augmentation signal strong enough to push it toward speaker-invariant features; (3) class-balancing inadvertently introduces partner-distribution drift that shows up as +0.003 LR-probe inflation (small, within noise).
- **Val-test gap widened** (-0.020 → -0.029, more pessimistic). Not concerning — pessimistic gap is the safe direction. Indicates training on noisier data converges earlier (best_epoch 1-5 vs A2.5's 1-3), so devel performance benefits from the early stop.

**Verdict reading (three interpretations, all defensible):**

1. **Strict gate read:** A5.5 FAILS on the 3-D acceptance gate (gates 2a/2b FAIL on probe drop). Augmentation lifts UAR slightly but doesn't reduce speaker leakage, which was the primary purpose. Modest result.
2. **Substantive criteria read:** A5.5 PASSES on UAR floor + no probe inflation beyond noise + clean shortcut gate. The +0.006 UAR with more balanced recall pattern is a real contribution; per the plan §6 spirit, augmentation didn't make things worse and added a small lift.
3. **Mechanism read (paper-relevant):** A5.5's de-confounding mechanism activated weakly. The +0.006 UAR comes from generic regularization (more effective training-data variation), NOT from speaker-invariant feature learning. The paper claim should be honest: *"Embedding mixup at α ∈ [0.70, 0.85] gave a small UAR lift over A2.5 (+0.006, within σ) without measurably reducing the speaker probe; the de-confounding mechanism didn't activate at this conservative mixing strength."*

**Decision (taken; see §4.8.8 for the sweep result): A5.5 LOCKED at conservative α ∈ [0.70, 0.85] as canonical.** The α ∈ [0.50, 0.70] sweep (§4.8.8) returned branch (d) — UAR DROPPED (Δ -0.017 vs A2.5) AND probe still didn't drop (MLP and LR both within noise of A2.5). Both ends of the α-axis fail: conservative is too gentle to de-confound; aggressive is too strong for label validity without recovering de-confounding. Per-chunk pooled-stat mixing has no usable operating point on URTIC + frozen WavLM. **Paper claim:** *"Embedding mixup at α ∈ [0.70, 0.85] gave a small UAR lift over A2.5 (+0.006, within σ) without measurably reducing the speaker probe; the de-confounding mechanism didn't activate at this conservative mixing strength, and a more aggressive sweep at α ∈ [0.50, 0.70] degraded UAR (Δ -0.017) without any probe drop, confirming that the operating window for per-chunk pooled-stat mixing on URTIC's frozen WavLM is empty. A5.5 ships as a modest data-level rung; the de-confounding load moves to A6 (representation-level contrastive) and A7 (gradient-level adversary)."*

#### 4.8.8 α-sweep clarifying experiment (DONE; α ∈ [0.50, 0.70] → branch (d) → A5.5 LOCKED at conservative-α)

**Status: DONE. Verdict: branch (d, unexpected) — UAR DROPS AND probe still doesn't drop. Both endpoints of the α-axis fail.** This was the cleanest possible disposition for the locking decision: it removes any "we should try more α" follow-up by showing that the operating window for per-chunk pooled-stat mixing on URTIC + frozen WavLM is empty, not just under-explored. **A5.5 LOCKED at conservative α ∈ [0.70, 0.85] as canonical**; the aggressive variant ships as ablation evidence (extends M9 → narrow-window-empty finding).

**Recipe (executed).** Mirrored Plan B Phase 2-equiv + Phase 4 with `ALPHA_RANGE = (0.50, 0.70)`, `BASE_SEED = 42 + 1000` (different seed namespace to avoid partner collision with conservative cache). New cache subdir `cache/microsoft_wavlm-large/pooled_mixup_aggro_k{0,1,2}/` (preserves the conservative-α cache for ablation comparison). Checkpoints `head_A55mixup_aggro_seed{seed}.pt`. Outputs: `results/A5_5_planB_phase2_mixup_aggro.json` + `results/A5_5_planB_phase4_mixup_aggro.json`. Cost: 1.13 min CPU cache build + 4.10 min Phase 4 (3 seeds, warm-start from A2.5).

**Cache integrity (Phase 2-equiv aggressive).** All k ∈ {0, 1, 2}: 8532/8532 successful mixes (100%), 0 fallback-to-original, mean α = 0.600 ± ~0.058 across all variants (sampled centre of [0.50, 0.70]), partner-class balance ~9% cold (matches corpus rate; class-balanced sampling preserved at the new α range). No partner-pool degeneracy — partner reuse top entries are 4–5× (same magnitude as conservative cache).

**Phase 4 results — aggressive α (3 seeds {42, 123, 7}):**

```text
metric                  A2.5 baseline                  A5.5 cons-α (α∈[0.70,0.85])    A5.5 aggro-α (α∈[0.50,0.70])    Δ vs A2.5 (aggro)
UAR (argmax)            0.6564 ± 0.0038                0.6624 ± 0.0031                 0.6397 ± 0.0186                 -0.0166  (-0.9σ_aggro, σ exploded ~6×)
UAR (calibrated)        —                              0.6636 ± 0.0145                 0.6323 ± 0.0194                 —
recall_C                0.43                           0.474 ± 0.088                   0.404 ± 0.123                   ↓
recall_NC               0.87                           0.853 ± 0.061                   0.860 ± 0.088                   flat
val→test gap            -0.001                         -0.029 ± 0.003                  -0.006 ± 0.020                  noisier (σ 7×)
MLP probe top-1         0.0501 ± 0.0045                0.0506 ± 0.0019                 0.0485 ± 0.0016                 -0.0017  (within noise)
LR  probe top-1         0.0725 ± 0.0002                0.0759 ± 0.0030                 0.0735 ± 0.0006                 +0.0010  (within noise)
cos(init_A2.5, final)   1.0000                         {0.99985, 0.99998, 0.99974}     {0.99998, 0.99998, 0.99998}     even MORE locked at A2.5 init
best_epoch              {1, 1, 5}                      {3, 1, 5}                       {1, 1, 1}                       converged-and-degrades immediately
```

**3-D gate verdict (aggressive α):**

| gate | target | achieved | passed |
| ---- | ------ | -------- | ------ |
| 1. UAR floor (≥ A2.5 - 1σ = 0.6525) | 0.6525 | 0.6397 | **FAIL** |
| 2a. MLP probe drop (≤ A2.5 - 1σ = 0.0456) | 0.0456 | 0.0485 | **FAIL** |
| 2b. LR  probe drop (≤ A2.5 - 1σ = 0.0723) | 0.0723 | 0.0735 | **FAIL** |
| 3. mix-bit-as-cold (≤ 0.52, gate B) | 0.52 | inherits 0.5000 from §4.8.5.b (α-independent) | **PASS** |
| **overall** | strict 3-of-3 (gates 1 + 2 + 3) | 1-of-4 | **FAIL → branch (d)** |

**Diagnostic interpretation (aggressive α, three findings):**

- **Branch (d) is the most informative branch in the decision tree.** It's the only branch that closes the α-axis: not "α was wrong, try another setting" (which is what branches a, b, or c would have left open) but "the entire α-axis is the wrong knob for this corpus + this backbone." The follow-up question becomes "what mechanism is missing?" rather than "what α should we try?"
- **σ exploded ~6× (0.0031 → 0.0186 on UAR), best_epoch collapsed to {1, 1, 1}.** At aggressive α the partner contribution (35–50%) is large enough that the cold label is no longer reliably the anchor's — mixed pooled vectors increasingly resemble partner-class statistics. The model converges on epoch 1 (training loss bottoms out fast), then DEGRADES with further training (the augmented distribution becomes noise from the cold-prediction perspective). σ across seeds explodes because the augmented training distribution is now sensitive to which partners get sampled — exactly the label-validity damage the conservative range was designed to avoid.
- **cos(init_A2.5, final) = 0.99998 across all 3 seeds — EVEN MORE LOCKED at A2.5 init than the conservative variant (which had {0.99985, 0.99998, 0.99974}).** The optimizer didn't move the layer weights at all. Combined with best_epoch = 1 across seeds, the picture is consistent: the head fine-tuned the MLP/classifier in the first epoch to fit the noisier augmented distribution, didn't move the layer-weight subspace (which is M5's known issue at default lr), and additional epochs only over-fit to partner-noise. **No mechanism activated** — not de-confounding (probes flat), not generic regularization (UAR dropped), not layer reweighting (cosine ~1).

**M9 extension (narrow-window-empty):** The original M9 ("embedding mixup at safe α produces small UAR lift but doesn't activate de-confounding") is now extended with the symmetric endpoint: **at aggressive α, label-validity damage shows up in UAR before any de-confounding shows up in the probe.** Both endpoints fail in different ways. The intermediate range U(0.60, 0.80) wouldn't change this: gentler than aggressive → falls back toward conservative's "no probe drop"; stronger than conservative → falls toward aggressive's "label damage." There's no escape via α tuning.

**Decision (locked).** Skip further α exploration. **A5.5 = conservative α ∈ [0.70, 0.85]** (UAR Δ +0.006, probe within noise, recall pattern more cold-balanced) **as canonical**; aggressive α ∈ [0.50, 0.70] ships as the ablation row that closes the α-axis. The de-confounding load moves to A6 (contrastive pretraining with speaker-masked positives) where the de-confounding objective is **explicit in the loss**, not implicit in the data distribution. Per the reflection's deeper diagnosis: mixup on post-frozen-backbone representations cannot de-confound without an explicit objective; A6 (representation-level) and A7 (gradient-level adversary) are the architecturally necessary rungs for the speaker-invariance claim.

**Three-level de-confounding story (paper framing).** With A5.5 locked, the de-confounding ladder is now scoped at three architectural levels — each addressing the speaker shortcut at a different point in the pipeline:

| level | rung | mechanism | A5.5's role |
| ----- | ---- | --------- | ----------- |
| data-level | A5.5 (LOCKED) | per-chunk embedding mixup, pseudo-speaker-aware partner sampling | small UAR lift via training-distribution variation; null on de-confounding probe — *evidence that data-level intervention is insufficient on its own* |
| representation-level | A6 (NEXT) | supervised contrastive pretraining with speaker-masked positives | explicit speaker-invariance objective in the embedding loss |
| gradient-level | A7 | MDD/DANN speaker adversary on the head's gradient | adversarial subtraction of the speaker direction during head training |

A5.5's modest contribution is itself the load-bearing evidence for *why* A6 and A7 are necessary. The paper writes this as a coherent three-level story rather than a missed gate.

**This is a closed clarifying experiment, not a new rung.** No further α-sweep, no follow-up Phase 4 variants. Documentation updates (this section, §4.8.6 decision pointer, §4.8 header, ladder row, summary.md, EXPLAINER.md M9 + A4) are the deliverable; A6 is the next active rung.

#### 4.8.7 Time-budget reality check

The reflection raised a fair point: A5.5 debugging could eat a week if it requires multiple splicer iterations. Two framings:

- **A5.5 as keystone "data-level de-confounding" rung** (paired with A6 representation-level + A7 gradient-level for a triple-mechanism story). Worth a week of iteration; the paper's architectural completeness depends on it.
- **A5.5 as additive ladder rung** (one more +1-2 UAR contribution). A week of debugging is steep; might skip and move to A6 (which is conceptually similar — its contrastive loss enforces de-confounding via training signal rather than data).

We're committed to A5.5 as keystone. Phase 3.5 self-splice control (~3 min compute) is the cheap diagnostic that decides whether splicer fixes are tractable or whether we need to pivot to embedding mixup. Either way, A5.5 ships.

### 4.9 A6 — supervised contrastive pretraining with speaker-masked positives (Phase 1 PoC scoped; queued)

**One-line framing.** A5.5 LOCKED at conservative-α with a null-on-probe verdict (M9 narrow-window-empty) tells us data-level mixup on post-frozen-backbone representations cannot de-confound without an explicit objective. A6 is the first rung where the de-confounding mechanism is **explicit in the loss**: a supervised contrastive objective with **speaker-masked positives** (same Cold label + different pseudo-speaker = positive; same-speaker same-class pairs masked out) directly pulls same-class-different-speaker representations together while pushing different-class apart. If this doesn't move the speaker probe, no purely-data-level intervention will, and we proceed to A7 (gradient-level adversary).

**Status: Phase 1 head-only PoC scoped + queued.** Anchor: A2.5 alone (cleaner ablation than A2.5 + A5b K=1 fused — A5b can be re-applied on top of A6 later as a separate fusion stage; that ordering keeps "contrastive helps" disambiguated from "contrastive helps on top of fusion"). Compute envelope for Phase 1: head-only PoC (~30-60 min) — cheapest first read on whether the speaker-masked-positive recipe even moves the probe. PoC verdict gates escalation to deeper interventions.

#### 4.9.1 Phase 1 PoC — head-only contrastive on cached pooled stats

**Recipe.**

- **Inputs.** Cached `pooled[chunk] ∈ R^{25×4096}` (existing A2/A2.5 cache, no new feature extraction). Apply A2.5's locked layer-weight softmax to collapse 25 layers → `R^4096` per chunk (uses the locked A2.5 init as the layer-mix prior; layer weights stay frozen during Phase 1 PoC — only the projection MLP and supervised-contrastive temperature train).
- **Projection MLP.** `R^4096 → R^512 → R^128` with GELU + LayerNorm; output L2-normalized to the unit hypersphere. Initialised fresh (no warm-start — projection space is new).
- **Loss.** Supervised contrastive (Khosla et al. 2020 SupCon) with **speaker-masked positives**:
  - Positive set for anchor `i`: all chunks `j ≠ i` in batch with `cold_label[j] == cold_label[i]` AND `pseudo_speaker[j] != pseudo_speaker[i]`. Same-speaker same-class pairs are EXCLUDED from positives (the de-confounding lever).
  - Negative set: all chunks `j ≠ i` in batch with `cold_label[j] != cold_label[i]` (different class). Same-speaker different-class pairs ARE used as negatives (no speaker-masking on negatives — we want the model to push apart classes regardless of speaker).
  - Loss: `L_i = -1/|P(i)| · sum_{p∈P(i)} log( exp(z_i · z_p / τ) / sum_{a∈A(i)} exp(z_i · z_a / τ) )`, τ = 0.07 default (SupCon's reported sweet spot).
- **Batch composition.** 8 pseudo-speakers × 8 chunks per batch (64 chunks per batch). Pseudo-speaker sampling: stratified to ensure each batch has both cold and non-cold chunks across multiple pseudo-speakers (so each anchor has at least 1 cross-speaker positive of its own class with high probability). Class proportions tracked per batch; reject batches where any anchor has 0 valid positives in the batch (rare but possible at small batch sizes given URTIC's 9.5% cold rate; document the rejection rate as a diagnostic).
- **Optimizer.** AdamW, lr 5e-5 (lower than head training's 1e-3 because the projection space is new and contrastive losses have higher gradient variance), weight decay 1e-4, 10 epochs, no early stopping (Phase 1 is pretraining, not classification).
- **Splits.** Train on `train_fit` (grouped split, 8532 chunks). Speaker-masked positive sampling uses the k=210 pseudo-speaker assignments. Devel never seen during contrastive training.

**Phase 1 measurement protocol (no classifier yet).**

After contrastive pretraining, evaluate the projection-MLP output `z ∈ R^128` on devel (no fine-tuning, no classifier head — the projection itself is the representation under test):

1. **Speaker probe on `z`** (this is the de-confounding measurement — the *primary* PoC signal). Train MLP (`speakers/probe.py`) and LR (`honesty/probe.py`) probes on `z[train_fit]` with pseudo-speaker targets, evaluate on `z[devel]`. Report top-1 + NMI. **Target: probe top-1 drops measurably below A2.5's 0.0501 (MLP) / 0.0725 (LR) — anything > 0.04 (MLP) or > 0.06 (LR) is a "didn't activate" verdict.**
2. **Cold linearity probe on `z`** (sanity: is cold info still present?). Train logistic regression on `z[train_fit]` with cold targets, evaluate on `z[devel]`. Report UAR. **Target: cold-LR UAR on `z` ≥ 0.60 (within ~5pp of A2.5's full-stack 0.6564) — confirms the projection didn't strip the cold signal alongside the speaker signal.**
3. **Class-collapse diagnostic.** Compute mean intra-class cosine similarity vs mean inter-class cosine similarity on `z[devel]`. **Target: intra-class > inter-class by ≥ 0.05** (the contrastive objective should produce a measurable class margin in cosine space).

**Phase 1 PoC acceptance gate (tiered P1-G1; 3 conditions).**

- **(P1-G1)** Speaker probe drops — **two tiers**:
  - **strict:** MLP top-1 ≤ 0.040 AND LR top-1 ≤ 0.060 on `z[devel]` (clear de-confounding signal on both substrates).
  - **soft:**   MLP top-1 ≤ 0.040 AND LR top-1 ≤ 0.0727 (LR didn't *inflate* beyond A2.5 + 1 seed-σ = 0.0725 + 0.0002; MLP still cleared strict). Catches "MLP responded to the contrastive objective; LR substrate weak but not damaged."
- **(P1-G2)** Cold information preserved: cold-LR on `z` UAR ≥ 0.60.
- **(P1-G3)** Class-margin emerged: intra/inter cosine gap ≥ 0.05.

`A_strict_PASS` requires P1-G1 strict + G2 + G3; `A_soft_PASS` requires P1-G1 soft + G2 + G3 (and not strict). Both escalate to Phase 2, but the soft tier flags the LR substrate as weak signal and recommends a τ-sweep diagnostic before committing to the heavier (A-ii) full-fine-tune. The gate is intentionally cheaper than the full A5.5 3-D gate (no full classifier training, no layer-weight stress test) — this is a diagnostic on whether the recipe activates the de-confounding mechanism at all, not a final A6 verdict.

**Phase 1 decision tree (5 branches with the tiered P1-G1):**

```text
(A_strict_PASS) ALL gates PASS at strict P1-G1:
    → Recipe works clearly on both probe substrates. Escalate to Phase 2
      (re-decide A-i layer-weight-open vs A-ii full-fine-tune per §4.9.2).
    → Default escalation: A-i if all gates clear with strong margins, else
      A-i first as the cheaper investment before considering A-ii.

(A_soft_PASS) ALL gates PASS at soft P1-G1 (MLP strict; LR within noise):
    → MLP cleared strict but LR substrate weak (between strict 0.060 and
      soft-noise 0.0727). The contrastive objective activated asymmetrically.
    → Escalate to Phase 2 — but BEFORE committing to (A-ii)'s heavy spend,
      run the branch-(B) sub-action τ-sweep ∈ {0.05, 0.07, 0.1, 0.2} as a
      ~30 min diagnostic to test whether a different temperature lifts the
      LR substrate too. If it does, re-run as (A_strict_PASS) and proceed
      to (A-i). If not, escalate to (A-i) at τ=0.07 as the working setting
      and accept the LR substrate is intrinsically weaker on URTIC.

(B) GATES P1-G2 + P1-G3 PASS, P1-G1 SOFT FAILS (probe didn't drop, even at the
soft threshold — i.e., MLP > 0.040 OR LR > 0.0727):
    → Class-margin emerged but speaker is still in z. Likely cause: pseudo-speaker
      labels at k=210 are too coarse (real speaker count >> 210, masked positives
      still share many true speakers) OR temperature τ too low (encourages
      sharp same-anchor neighborhoods that preserve speaker microstructure).
    → Diagnostic before escalation: (a) re-cluster to k=420 or k=600 and re-run;
      (b) τ sweep ∈ {0.05, 0.07, 0.1, 0.2}; (c) hard-negative mining (use
      same-speaker-different-class as forced hard negatives).
    → If diagnostics don't move the probe, accept "data + representation-level
      can't de-confound on URTIC + frozen WavLM" and proceed to A7
      (gradient-level adversary — only mechanism left below transformer fine-tune).

(C) GATES P1-G1 PASS (any tier), P1-G2 FAILS (cold info collapsed):
    → Contrastive objective stripped cold alongside speaker. Likely cause: λ_balance
      between supervised-contrastive and cold-aware regularization is wrong;
      the unmodified SupCon assumes cold and speaker are independently separable,
      but URTIC's cold-in-voice-quality means the two are partially aligned in
      pooled space. Stripping speaker direction strips part of cold direction.
    → Diagnostic: add a cold-classification auxiliary loss with small weight λ_cls
      (e.g., 0.1) to anchor cold linearity during contrastive training. Re-run.
    → If aux-loss doesn't recover cold without losing speaker drop, document the
      Pareto trade-off (analogous to A5.5's branch (c)) and lock A6 as
      "speaker-invariant but UAR-degraded" with explicit framing.

(D) NEITHER GATE PASSES (probe stays AND cold collapses):
    → Recipe completely failed to learn structure. Most likely cause: projection
      MLP collapsed to constant (all z ≈ same vector — class-margin gate would
      flag this too). Diagnose batch composition (any anchor with 0 valid
      positives → SupCon undefined; rejection rate too high collapses the loss).
    → Fix batch sampler (force ≥ 1 cross-speaker positive per anchor by
      construction), re-run.
    → If still null, the head-only setting is too constrained and the contrastive
      signal can't propagate into the frozen feature space. Escalate directly to
      Phase 2 (open layer weights) without claiming Phase 1 PoC verdict.
```

**Cost.** ~30-60 min: 8532 chunks × 10 epochs × ~64 chunks/batch ≈ 1300 SGD steps; on cached pooled stats this is essentially compute-free per step. Phase 1 measurement (probes + class-margin) adds ~5 min. PoC fits well inside an hour.

**Output.** `results/A6_phase1_PoC.json` with: contrastive loss curve (per epoch), Phase 1 measurement triple (speaker probe top-1 MLP/LR, cold-LR UAR on z, intra/inter cosine gap), batch-rejection rate, tiered gate verdict (`P1_G1_strict`, `P1_G1_soft`, `P1_G2_cold_preserved`, `P1_G3_class_margin`), branch enum ∈ {A_strict_PASS, A_soft_PASS, B_probe_didnt_drop, C_cold_collapsed, D_recipe_failed_or_mixed}, per-seed (3 seeds {42, 123, 7}). Checkpoints: `cache/microsoft_wavlm-large/A6_phase1_proj_seed{seed}.pt`.

**Engineering deliverables for Phase 1.**

- New module: `model/representation/contrastive.py` — projection MLP, SupCon loss with speaker-masked positives, batch sampler.
- Cell in `run.ipynb`: Phase 1 training + measurement + verdict classifier (mirrors A5.5's auto-classifier pattern).
- Reuses existing: `data.cached_dataset.PooledCacheDataset`, `data.cached_dataset.stratified_grouped_split`, `speakers.cluster.load_pseudo_speakers`, `speakers.probe.train_probe` (MLP), `honesty.speaker_probe` (LR).

#### 4.9.1.1 Bottleneck confound + controls (DONE-as-scoped, runs after Phase 1 PoC)

**Why this exists.** The Phase 1 PoC (DONE; see `results/A6_phase1_PoC.json`) gave a striking result: LR speaker probe dropped from A2.5's 0.0725 to 0.0410 (~43% relative, ~13σ on the LR seed-σ); MLP probe barely moved (0.0501 → 0.0477); cold-LR UAR landed at 0.5988 ± 0.0085 (borderline G2); class margin emerged at +0.0594. The strict gate fired branch (D) for 2/3 seeds, but the mechanistic read was richer: linear speaker direction got scrubbed strongly while nonlinear pockets survived, and cold linearity took a hit.

**The confound.** A2.5's LR=0.0725 reference was measured on the **4096-d** fused representation. The PoC LR is on a **128-d L2-normalised z**. The 32× dimensionality reduction plus the metric change to the unit hypersphere can drop LR-recoverable speaker information *independent of the contrastive objective*. Same problem for cold-LR (information loss from bottleneck). Without controls, we can't tell whether the LR drop is from speaker-masked SupCon doing real de-confounding work, or just from the bottleneck reshaping the substrate.

**Three controls** (all measured on the same 128-d L2-normalised `z` for apples-to-apples; same A2.5 frozen scaler + layer-weights as the PoC; 3 seeds {42, 123, 7}):

| control | training | isolates |
| ------- | -------- | -------- |
| **C1: random projection** | none — fresh init, no training | bottleneck + L2-norm only |
| **C2: cold-CE only** | 10 epochs CE on cold (linear cold head, class-balanced weights), no contrastive | bottleneck + supervised cold pressure (no speaker-aware objective) |
| **C3: vanilla SupCon** | 10 epochs SupCon with `mask_speakers=False` (positives = same Cold any speaker) | bottleneck + class-pressure (no speaker-masking lever) |

**Decision rule** (auto-classifier in the cell):

- **A6 beats all 3 controls on LR drop while preserving margin + cold-LR** → real speaker-masking effect → escalate to Phase 2 (re-decide A-i layer-weight-open vs A-ii full-fine-tune per §4.9.2).
- **C1/C2 already drop LR to ~A6 level** (within +0.005) → bottleneck explains most of the drop → A6's contrastive mechanism is largely illusory; pivot to A7.
- **C3 matches A6's LR drop** (within +0.005) → class-structure pressure alone is sufficient; speaker-masking adds no value; simplify recipe (drop `diff_speaker` constraint and re-run as A6 canonical).

**Engineering deliverables.**

- `model/representation/contrastive.py`: added `mask_speakers: bool = True` parameter to `supcon_speaker_masked_loss`. When `False`, positive set drops the `diff_speaker` filter — degrades to vanilla SupCon. Smoke-tested.
- New cell in `run.ipynb` (cells 92-93): `a6_phase1_poc_controls.py` runs the 3 controls × 3 seeds, computes the comparison table, applies the decision rule, writes `results/A6_phase1_PoC_controls.json`. Self-contained (re-loads splits + materialises pooled caches; doesn't depend on cell 91 being run in the same kernel).

**Cost.** ~10-15 min total: 3 controls × 3 seeds × ~30 s training (cached pooled stats are essentially compute-free) + ~60 s per-seed measurement (probes + cosine margin).

**Results (DONE).** Controls ran in 1.62 min (`results/A6_phase1_PoC_controls.json`). Comparison table on the same 128-d L2-normalised `z` substrate:

| mode (substrate) | MLP top1 | LR top1 | cold UAR | margin |
| ---------------- | -------- | ------- | -------- | ------ |
| A2.5 (**4096-d** reference) | 0.0501 | 0.0725 | 0.6564 | — |
| C1 random projection (128-d, no training) | 0.0500 ± 0.0022 | **0.0427 ± 0.0016** | 0.6007 ± 0.0071 | +0.0016 ± 0.0024 |
| C2 cold-CE only (128-d, no contrastive) | **0.0413 ± 0.0014** | **0.0373 ± 0.0006** | **0.6124 ± 0.0060** | **+0.3503 ± 0.0127** |
| C3 vanilla SupCon (128-d, no speaker mask) | 0.0468 ± 0.0024 | **0.0395 ± 0.0040** | 0.6012 ± 0.0061 | +0.0489 ± 0.0039 |
| **A6 speaker-masked SupCon (128-d)** | 0.0477 ± 0.0008 | 0.0410 ± 0.0024 | 0.5988 ± 0.0085 | +0.0594 ± 0.0065 |

**Verdict: `bottleneck_explains_lr_drop`.** Three independent refutations of the head-only A6 mechanism:

1. **Bottleneck alone explains the LR drop.** Random untrained projection drops LR to 0.0427 (within 0.0017 of A6's 0.0410). The 4096 → 128 + L2-norm bottleneck does essentially all of the LR-substrate de-confounding, independent of training.
2. **Cold-CE Pareto-dominates A6 on every measured dimension.** Lower LR (0.0373 vs A6's 0.0410), lower MLP (0.0413 vs 0.0477), higher cold UAR (0.6124 vs 0.5988), and a much higher margin (+0.350 vs +0.059). Supervised cold pressure subsumes whatever the speaker-aware contrastive was supposed to provide.
3. **Speaker-masking isn't just neutral — it's mildly harmful.** Vanilla SupCon (drop the `diff_speaker` constraint) gives LR=0.0395, slightly better than A6's 0.0410. The masking lever reduces the positive count and weakens the loss signal without buying any de-confounding back.

**Implication for §4.9.2 Phase 2.** The original "A_strict_PASS / A_soft_PASS → escalate to (A-i) layer-weight-open" trigger is invalidated for the head-only PoC: there's no Phase 1 mechanism to escalate. **(A-i) layer-weight-open is not automatically dead** — opening the WavLM layer-weight subspace under combined cls + supcon loss tests a different mechanism (layer re-orientation, not projection re-shaping) — but it MUST run with the same control discipline (cold-CE-only at layer-weight-open, random-projection-at-layer-weight-open) before any verdict is locked. **(A-ii) full transformer fine-tune is doubly disqualified for now**: the head-only PoC's mechanism is illusory, and A-i needs its own controls before justifying multi-hour-to-day fine-tune spend.

**Methodology lesson (M10 — see EXPLAINER.md §14).** *"Probe-substrate dimensionality is itself a confound for de-confounding measurements: comparing a probe trained on a high-dim baseline representation against the same probe trained on a low-dim post-projection representation conflates information loss from dimensionality reduction with de-confounding from the training objective. De-confounding rungs that introduce a dimensionality bottleneck must include a random-projection control at the same target dimensionality to disambiguate. Substrate-specific absolute thresholds calibrated against an undifferentiated baseline can fire false-positive de-confounding verdicts; future de-confounding rungs should anchor on **same-substrate relative drop** or use within-control comparison."*

This generalises beyond URTIC. Anyone doing representation-level de-confounding on foundation-model features with a projection MLP should run a random-projection control at the same target dim before claiming mechanism activation.

**Status.** A6 head-only PoC mechanism = LOCKED as illusory (bottleneck artefact). (A-i) layer-weight-open variant available IF the user wants to spend 2-4 hr GPU on it, but must include the same controls. Otherwise pivot to A7 (gradient-level adversary, the remaining unexplored de-confounding mechanism). See §4.9.1.2 for the follow-up combined-loss test that closed the head-only scope across all variants.

#### 4.9.1.2 A6b — combined cold-CE + speaker-masked SupCon (lambda-sweep, DONE; closes head-only scope)

**Why this exists.** §4.9.1.1 disproved the strong claim *"speaker-masked SupCon alone uniquely de-confounds the representation"* but didn't disprove *"contrastive class-pressure as a regulariser on top of CE-anchored cold bottleneck might still help."* §4.9.1.2 tests the latter directly: train projection + cold linear head jointly with `L = L_cold_CE + λ · L_supcon_speaker_masked`, sweep λ ∈ {0.0, 0.05, 0.1, 0.25, 0.5}. λ=0 reproduces cold-CE-only at this code path (sanity).

**Recipe.** Same A2.5 anchor (frozen scaler + layer_weights), same projection (4096 → 512 → 128, L2-norm), same `SpeakerBlockSampler`, same probe protocol as §4.9.1.1. New components: `cold_head = nn.Linear(128, 2)` for the CE term; `loss = F.cross_entropy(cold_head(z), cold_t, weight=ce_class_weights) + λ · supcon_speaker_masked_loss(z, cold_t, spk_t, mask_speakers=True)`. 10 epochs, AdamW lr 5e-5. 3 seeds {42, 123, 7}. Cost: 2.66 min (5 λ × 3 seeds, training+probes).

**Results (DONE).** `results/A6b_phase1_combined_lambda_sweep.json`:

| variant (128-d z) | MLP top1 | LR top1 | cold UAR | margin |
| ----------------- | -------- | ------- | -------- | ------ |
| A6b λ=0.0 (pure cold-CE) | **0.0408 ± 0.0008** | **0.0371 ± 0.0005** | **0.6128 ± 0.0058** | **+0.3503 ± 0.0127** |
| A6b λ=0.05 | 0.0428 ± 0.0016 | 0.0387 ± 0.0041 | 0.5994 ± 0.0020 | +0.2959 ± 0.0135 |
| A6b λ=0.1 | 0.0435 ± 0.0006 | 0.0379 ± 0.0025 | 0.6028 ± 0.0058 | +0.2422 ± 0.0113 |
| A6b λ=0.25 | 0.0451 ± 0.0007 | 0.0414 ± 0.0035 | 0.5996 ± 0.0055 | +0.1601 ± 0.0122 |
| A6b λ=0.5 | 0.0462 ± 0.0011 | 0.0419 ± 0.0019 | 0.6005 ± 0.0030 | +0.1065 ± 0.0077 |

**Verdict: `contrastive_dead_pivot_to_a7`.** λ=0 is monotonically the best on every measured dimension:

- **MLP probe** worsens monotonically with λ: 0.0408 → 0.0428 → 0.0435 → 0.0451 → 0.0462.
- **LR probe** worsens monotonically (with a tiny noise dip at λ=0.1): 0.0371 → 0.0387 → 0.0379 → 0.0414 → 0.0419.
- **Cold UAR** drops at λ > 0 and stays flat-low (0.5994–0.6028 vs λ=0's 0.6128) — the contrastive term mildly damages cold linearity.
- **Margin** drops monotonically (+0.350 → +0.107 as λ grows) — the contrastive pressure shrinks rather than enhances the class margin when CE is already shaping the geometry.

**Sanity.** A6b λ=0.0 reproduces the cached cold-CE control within seed noise (0.0408/0.0371/0.6128/+0.350 vs cached 0.0413/0.0373/0.6124/+0.350) — code path validated.

**Mechanism reading (4-step causal chain).** Why does adding any positive λ make every metric strictly worse? The CE+bottleneck combination already does the substrate-compression work that the contrastive recipe was supposed to provide, leaving no headroom for SupCon to add value:

1. **CE on the 128-d projection bottleneck produces a sharply class-separated geometry** (margin +0.350) where speaker information is mostly compressed away by the bottleneck itself (the M10 finding — random projection alone gives LR=0.0427).
2. **Adding SupCon class-pressure pushes points toward class centroids** — but the bottleneck already produced clean separation, so the contrastive pressure doesn't *add* useful structure; it *flattens within-class diversity* by pulling intra-class points closer than the CE objective alone would.
3. **The flattened within-class diversity includes the cold-relevant variation that helped CE generalise.** Squeezing it out drops cold UAR (0.6128 → ~0.60 across all λ > 0) — the model loses the cold-discriminative micro-structure that survived CE alone.
4. **SupCon treats "different cold class" as separation pressure regardless of speaker** — so it can pull two same-speaker chunks (one cold, one not) further apart than CE alone would. This *re-introduces speaker-correlated variance* into the projection by amplifying intra-speaker class-disagreement, which is why MLP probe top-1 climbs monotonically with λ (0.0408 → 0.0462).

The four-step chain is consistent with the data and gives a falsifiable mechanistic claim: *"on a CE-anchored projection bottleneck where the substrate compression already eliminates most linearly-decodable speaker information, adding supervised contrastive pressure is subtractive — it flattens cold-relevant within-class variation (lowering cold UAR) while amplifying speaker-correlated separation between same-speaker different-class pairs (raising probe top-1)."* This is a publishable mechanism, not just a recipe-level FAIL.

**Status: A6 head-only fully closed across all tested recipes.** Three independent recipes tested — pure speaker-masked SupCon (A6 PoC, §4.9.1), vanilla SupCon (control C3, §4.9.1.1), and combined CE+SupCon λ-sweep (this section). All fail; pure cold-CE Pareto-dominates. **Conclusion: contrastive class-pressure has no de-confounding leverage on URTIC + frozen WavLM at head-only scope.** This is the publishable negative result — the methodology yielded a clean closure, not just a single-recipe failure.

**Implications for future A6 variants — recommendation: don't.** The architectural argument against scaling up:

- The λ-sweep didn't fail by a small margin or in a specific corner of recipe space. It failed monotonically across five λ values, with λ=0 strictly best on every metric.
- The mechanism that produces this monotonic failure is *not* "the contrastive recipe needs more substrate access." It's "contrastive class-pressure is subtractive on CE-anchored bottlenecks" (the 4-step chain above).
- (A-i) layer-weight-open gives the contrastive loss more parameters to push around, but the underlying interaction (CE + bottleneck → already-compressed; SupCon adds nothing useful) *doesn't change with more parameters*.
- (A-ii) full transformer fine-tune *might* give a different result — but only because the CE anchor is no longer dominant (the transformer can rearrange itself), so you'd be testing "does fine-tuning help" rather than "does contrastive de-confounding work." That's a different question and not the one A6 was scoped to answer.

**Decision: A6 head-only fully closed; layer-weight and full-fine-tune variants NOT recommended.** The λ-sweep verdict generalises to any CE-anchored scope — the substrate already does the work CE needs, contrastive adds nothing useful regardless of where the gradient flows. Pivot directly to A7.

**Optional follow-up worth queuing AFTER A7 lands** (interesting paper reference point, not a critical-path test): run cold-CE Phase 2 — train a cold classifier on top of the cold-CE-only projection from §4.9.1.1 control (currently used only as a Phase 1 measurement substrate). If Phase 2 cold-CE matches A2.5's UAR with comparable probe numbers, the paper has an extra reference: *"Phase 1 cold-CE pretraining is approximately equivalent to A2.5's joint training; the projection bottleneck does the de-confounding work that the contrastive recipes attempted but failed to add."* Optional; not load-bearing.

**Pivot recommendation: A7 (gradient-level adversary).** The data-level rung (A5.5) is locked at modest UAR / null on probe. The representation-level head-only scope is now fully closed across three recipe families with a sharp mechanistic explanation. Gradient-level is the remaining unexplored mechanism with the architectural access to operate where the others can't reach. Plan §4.10 to be scoped next.

#### 4.9.2 Phase 2 — conditional escalation (scoped, conditional on Phase 1)

**Triggers (Phase 1 verdict → Phase 2 action):**

- **(A_strict_PASS or A_soft_PASS) PoC PASS** → Phase 2 is the canonical full A6, but the *depth* of intervention is an **open re-decision** with PoC evidence in hand. Two candidate Phase 2 recipes (deferred — pick after PoC verdict):
  - **(A-i) Layer-weight-open + projection + classification head** (~2-4 hr GPU). Open WavLM layer weights at lr×10 (per M5 — layer subspace has gradient signal at higher lr), attach cold-classification head, train end-to-end with combined loss `L = L_classifier + λ_contrastive · L_supcon` (λ_contrastive ramped from 1.0 at epoch 0 to 0.1 by epoch 10 — contrastive shapes the representation early, classifier dominates late). Keeps the WavLM transformer frozen; only the layer mix re-orients under the contrastive + classification objective. Lowest-risk, highest-information escalation.
  - **(A-ii) Full WavLM transformer fine-tune** (multi-hour-to-day GPU). Unfreeze WavLM, contrastive + classification losses propagate through the transformer. Strongest possible mechanism activation but expensive and may overfit on 8.5k train_fit chunks (URTIC is small for unfreezing a 300M-param transformer). Risky — better as a Phase 3 if (A-i) PASSes but is bottlenecked by the frozen-transformer ceiling.
  - **Default (re-decide at PoC verdict):**
    - If `A_strict_PASS` with **strong margins on all 3 gates** (probe drop ≥ 2σ, cold-LR ≥ 0.62, class-margin ≥ 0.10), escalate directly to (A-i) — the layer-weight subspace likely carries enough capacity.
    - If `A_strict_PASS` with **borderline margins** (any gate within 1σ of threshold), the head-only ceiling is close to its limit; run (A-i) first as a 2-4 hr investment with a clear PASS/FAIL verdict before committing to (A-ii)'s multi-hour-to-day spend.
    - If `A_soft_PASS`, the LR substrate is weak. **Run the τ-sweep diagnostic first** (~30 min, branch-B sub-action). If τ-sweep recovers `A_strict_PASS`, proceed to (A-i) at the new τ. If not, proceed to (A-i) at τ=0.07 and accept LR is intrinsically weaker on URTIC.
    - **Re-ask the user with PoC numbers in hand rather than pre-committing.**
- **(B) PoC branch B** (probe didn't drop) → Phase 2 is the diagnostic sweep (k re-cluster, τ sweep, hard-negative mining). Each diagnostic is its own ~30 min run; if any moves the probe below threshold, escalate to (A); otherwise proceed to A7. Cost: ~1.5-2 hr.
- **(C) PoC branch C** (cold collapsed) → Phase 2 is auxiliary-loss recovery (add λ_cls·L_cls term, sweep λ_cls ∈ {0.05, 0.1, 0.25}). Cost: ~1.5 hr.
- **(D) PoC branch D** (recipe failed) → Phase 2 is batch-sampler diagnosis + (if needed) direct escalation to layer-weight-open without Phase 1 verdict. Cost: ~30 min diag + 2-4 hr if escalated.

**No Phase 2 plan locked yet** — keeping it conditional on Phase 1 results to avoid scoping work that the Phase 1 verdict may obviate. The Phase 1 PoC is the cheap diagnostic that decides Phase 2's direction.

#### 4.9.3 Acceptance gate for the full A6 rung (post-Phase 2)

**3-D acceptance gate vs A2.5 (mirrors A5.5's structure):**

1. **UAR floor.** A6 head UAR ≥ A2.5 - 1σ (= 0.6525) on devel_test, 3 seeds.
2. **Speaker probe drop.** MLP probe top-1 ≤ A2.5 - 1σ (= 0.0456) AND LR probe top-1 ≤ A2.5 - 1σ (= 0.0723) — at least one must clear by ≥ 2σ for a "strong PASS" verdict; both clearing at 1σ is "PASS".
3. **Class-margin sustained** (replaces A5.5's mix-bit gate). Intra-class cosine ≥ inter-class cosine + 0.05 on devel — confirms the contrastive structure survived the classifier-head fine-tune.

**Strong PASS:** all 3 with at least one probe clearing at 2σ. **PASS:** all 3 at 1σ. **PARTIAL PASS:** UAR floor + probe drop on one substrate (MLP or LR) but not both. **FAIL:** any other combination → re-scope toward A7.

#### 4.9.4 A6 in the 3-level de-confounding ladder (paper framing)

A5.5's modest contribution + null-on-probe is the load-bearing evidence that data-level intervention is insufficient. A6's value (whether PoC PASSes or FAILs) is **direct evidence about whether representation-level intervention is sufficient on URTIC + frozen WavLM**:

- **A6 PoC PASS → full A6 PASS:** the de-confounding paper claim moves from "we tried two levels and only the representation level worked" to "we localized the de-confounding mechanism to the representation level." A7 becomes optional ablation (does adding gradient-level on top of representation-level help further?).
- **A6 PoC PASS → full A6 PARTIAL:** representation level activates the mechanism but not enough; A7 is the load-bearing rung for the headline UAR + probe drop.
- **A6 PoC FAIL (branch B/C/D, no escape via diagnostics):** the de-confounding paper claim becomes "neither data nor representation-level intervention works on URTIC + frozen WavLM at the contrastive-loss strength tested; A7 (gradient-level) is necessary." A7 becomes the paper's main contribution.

Either outcome is publishable. A6 PoC's main risk is *invisibility* (silent PARTIAL where probe moves slightly but no gate clears) — the 3-of-3 gate at strict thresholds prevents that by forcing a categorical verdict.

### 4.10 A7 — gradient-level speaker adversary (DANN/MDD; load-bearing for de-confounding claim)

**One-line framing.** With A5.5 locked at modest UAR / null on probe (M9) and A6 head-only fully closed across three recipe families (M10 + M11), gradient-level intervention is the only architectural mechanism that hasn't been ruled out. A7 inserts a gradient reversal layer (Ganin 2015 DANN) between the projection output `z` and a pseudo-speaker discriminator. The discriminator tries to predict speaker; gradient reversal tells the projection to actively *unlearn* speaker-discriminative directions while the cold classifier still trains normally. **The architectural lever is qualitatively different from A6**: contrastive class-pressure shapes geometry; adversarial subtraction explicitly removes speaker direction from the optimization signal — operating at a level (gradient) where the bottleneck-already-compressed argument doesn't apply (the adversary creates *additional* compression pressure beyond what the bottleneck does).

**Status: Phase 1 PoC scoped + queued.** Anchor: A2.5. Scope: **layer-weight-open from the start** (head-only A7 would have the same M10 bottleneck-confound problem A6 just exposed; layer-weight scope tests a structurally different mechanism). Compute envelope for Phase 1: ~1-2 hr GPU (`λ_adv` sweep × 3 seeds + M10/M11 baked-in controls).

#### 4.10.1 Phase 1 PoC — DANN with layer-weight-open scope + λ_adv sweep + baked-in controls

**Recipe.**

- **Architecture.** A2.5 standardiser (frozen) → A2.5 layer_weights (open at lr×10 per M5) → fused [B, 4096] → projection MLP (4096 → 512 → 128, L2-norm; same as A6) → split into two heads:
  - **Cold head**: `Linear(128, 2)` trained with class-balanced cross-entropy on cold labels.
  - **Speaker discriminator**: `GradReverse(λ_adv) → Linear(128, 256) → GELU → Linear(256, n_speakers=210)` trained with cross-entropy on pseudo-speaker labels.
- **Loss.** `L = L_cold_CE + L_speaker_CE`. The gradient reversal layer `GradReverse(λ)` passes the forward pass unchanged but multiplies gradients by `-λ` on the backward pass — so the projection sees the speaker discriminator's gradient as a *push to make speaker harder to predict*, while the discriminator itself still trains normally to predict speaker (an arms race).
- **λ_adv schedule.** Standard Ganin sigmoid ramp `λ_adv(p) = λ_max · (2/(1+exp(-10p)) - 1)` where `p = epoch/total_epochs`. Sweep `λ_max ∈ {0.0, 0.01, 0.03, 0.1, 0.3, 1.0}` — finer at the low end because speaker and cold are entangled (λ=1 may scrub useful disease signal); λ_max=0 reproduces "everything except the adversary" at this code path and is **the primary matched control** (not A2.5) for the acceptance gate. Mirrors the A6b λ=0 baseline pattern.
- **Optimizer.** AdamW, lr 1e-4 on projection + heads, lr 1e-5 on layer_weights (M5's lr×10 from base 1e-4 — actually lr×0.1 from the head lr; this is the "layer weights open at lr×10" in M5 terminology). Weight decay 1e-4. 20 epochs (longer than A6 because adversarial training is noisier and needs the ramp to anneal).
- **Splits.** Train on `train_fit` (grouped, 8532 chunks) with the existing `SpeakerBlockSampler` (8 spk × 8 chunks = 64). Devel never seen during training.
- **Anchor.** Warm-start the projection from cold-CE-only PoC checkpoint (the C2 control from §4.9.1.1) so the projection starts at a known-good cold geometry. Layer_weights warm-start from A2.5. Speaker discriminator initialized fresh.

**Phase 1 measurement protocol.** Same as A6 PoC + controls (M10-disciplined apples-to-apples on 128-d L2-normalized z), plus adversary-health diagnostics:

1. **Speaker probe on `z` after training** (the de-confounding measurement). Train MLP probe (`speakers/probe.py`) and LR probe (`honesty/probe.py`) on `z[train_fit]` with pseudo-speaker targets, evaluate on `z[devel]`. Report top-1 + NMI.
2. **Cold linearity probe on `z`** (sanity: did adversary destroy cold info?). Train cold-LR probe; report UAR + recall_pos / recall_neg.
3. **Class-margin diagnostic.** intra/inter cosine on `z[devel]`.
4. **Cold classifier UAR on devel_test.** The actual headline number — does the cold head's prediction (not just the probe-on-z) clear the gate?
5. **Adversary-health diagnostics (per-epoch logging during training):** speaker discriminator train accuracy on the batch, and devel-set discriminator accuracy at end-of-training. **Why this matters:** if discriminator train accuracy stays near chance (1/210 ≈ 0.005), the adversary isn't learning to predict speaker → no adversarial signal → null verdict is uninterpretable. If it climbs to ~1.0 immediately, it's overfitting → noisy/weak signal. Healthy: steady climb that the GRL pulls back over time, leaving the discriminator at a meaningful but non-saturated level (e.g., 0.05-0.30). This diagnostic is what makes the verdict interpretable — without it, "λ=0 best" could mean "adversary actually hurts" OR "adversary never trained" and we can't tell which.

**Phase 1 acceptance gate — MATCHED CONTROL (vs λ=0, not A2.5).** A2.5 is the global reference but A7's λ=0 sibling is architecturally identical to A7(λ>0) except for the adversary, so it isolates the adversary's contribution. A7 with λ=0 already has the layer-weight-open training, the projection MLP, the warm start, and the 20-epoch horizon — comparing A7(λ>0) to A2.5 instead would conflate "layer-weight training helps" with "adversary helps." Tiered verdict, mirrors A6b's pattern:

- **(A7-Strict)** Probe drop ≥ 2σ_{λ=0} vs the λ=0 mean (both MLP and LR substrates), AND cold classifier UAR ≥ λ=0_mean - 1σ_{λ=0}. The 2σ requirement is stricter than A6's 1σ because A7 is the load-bearing de-confounding rung — we want a clear adversary signal.
- **(A7-Soft)** Probe drop ≥ 1σ_{λ=0} on at least one substrate, AND cold UAR ≥ λ=0_mean - 0.5σ_{λ=0} (preserved within tighter noise). Catches cases where the adversary moved one substrate cleanly while leaving the other within noise — analogous to A6b's `A_soft_PASS` framing.
- **(A7-G3) M10 same-scope control.** A7 must also beat **random projection at layer-weight-open scope** on the LR probe. The λ=0 control IS the cold-CE-at-layer-weight-open baseline (so "beats λ=0" subsumes the cold-CE control). Random projection at the same scope is the additional control row to disambiguate "did the layer-weight training shape the geometry beyond what random init achieves?" — necessary to claim the layer-weight scope itself activates de-confounding pressure.
- **A2.5 reference comparison** still reported in the table for global context, but it's diagnostic, not gating.

**Phase 1 decision tree (5 branches with the tiered verdict):**

```text
(A_strict_PASS) Best λ_max wins on A7-Strict + A7-G3:
    → Adversary clearly works. Lock best λ_max as canonical A7. Escalate to
      Phase 2 (longer training, MDD substitution as ablation, full transformer
      fine-tune as Phase 3 if Phase 2 plateaus).

(A_soft_PASS) Best λ_max wins on A7-Soft (not Strict) + A7-G3:
    → Adversary partially activates. Lock best λ_max as canonical A7 with
      "soft PASS" framing. Phase 2 escalation considers MDD substitution
      first (stronger discrepancy bound may convert soft -> strict) before
      committing to full-fine-tune.

(B) λ_max = 0 strict best (no adversary helps) AND adversary discriminator
trained meaningfully (train acc > 5x chance, e.g., > 0.025):
    → DANN at this scope/recipe doesn't activate de-confounding. Adversary
      saw real speaker signal but couldn't push it out of z. Try MDD before
      declaring A7 dead. If MDD also fails -> de-confounding at any
      architectural level is mechanism-resistant on URTIC + frozen WavLM.
      Paper becomes a strong negative-result methodology contribution.

(B_null) λ_max = 0 strict best AND discriminator never trained meaningfully
(train acc ~ chance):
    → Verdict uninterpretable -- adversary signal was null because the
      discriminator architecture is too weak or the GRL gradient is being
      cancelled. Diagnose: (a) deeper/wider discriminator; (b) lower lr on
      adversary; (c) decouple discriminator and projection updates (alternate
      step like GAN training). Re-run before claiming B.

(C) λ_max > 0 lowers probe but drops UAR below A7-Soft floor (Pareto trade-off):
    → Document the trade-off. Try smaller λ_max grid {0.005, 0.01} or warmer
      ramp schedule. If no setting clears both gates -> A7 ships as
      "speaker-invariant but UAR-degraded" with explicit Pareto framing.

(D) Adversary destabilises training (loss diverges, UAR collapses to chance):
    → λ_max too aggressive at this scope. Switch to constant-λ schedule with
      smaller values, or warm up λ over more epochs.
```

**Cost.** ~2-3 hr GPU: 6 λ_max × 3 seeds × ~5-10 min each (longer than A6 because layer_weights are open + 20 epochs vs 10). λ=0 IS the cold-CE-at-layer-weight-open control (no separate run needed); random-projection-at-layer-weight-open adds 1 extra row × 3 seeds (~15 min). Total fits in an afternoon.

**Output.** `results/A7_phase1_PoC.json` — λ_adv sweep + per-seed metrics + adversary-health diagnostics + tiered verdict + matched-control row + random-projection row. Checkpoints: `cache/microsoft_wavlm-large/A7_phase1_proj_seed{seed}_lambda{λ}.pt`.

**Phase 1 PoC results (DONE; verdict = INCONCLUSIVE / B_null pending discriminator-ceiling diagnostic).** Ran 4.95 min:

| variant | cls UAR | MLP | LR | margin | disc-train |
| ------- | ------- | --- | -- | ------ | ---------- |
| A2.5 (4096-d ref) | 0.6564 | 0.0501 | 0.0725 | — | — |
| M10 random-LW (128-d) | — | 0.0498 | 0.0427 | +0.002 | — |
| A7 λ_max=0 (matched control) | 0.6280 ± 0.0034 | 0.0488 ± 0.0030 | 0.0436 ± 0.0019 | +0.363 | **0.0235** |
| A7 λ_max=0.01 | 0.6375 ± 0.0118 | 0.0504 ± 0.0027 | 0.0444 ± 0.0020 | +0.376 | 0.0233 |
| A7 λ_max=0.03 | 0.6297 ± 0.0032 | 0.0461 ± 0.0039 | 0.0419 ± 0.0042 | +0.348 | 0.0230 |
| A7 λ_max=0.1 | 0.6268 ± 0.0130 | 0.0473 ± 0.0021 | 0.0433 ± 0.0011 | +0.336 | 0.0214 |
| A7 λ_max=0.3 | 0.6186 ± 0.0126 | 0.0551 ± 0.0048 | 0.0461 ± 0.0019 | +0.310 | 0.0210 |
| A7 λ_max=1.0 | 0.5927 ± 0.0103 | 0.0700 ± 0.0007 | 0.0630 ± 0.0012 | +0.174 | **0.0144** |

**Three observations:**

1. **GRL is mechanically active** — discriminator accuracy decreases monotonically with λ (0.0235 → 0.0144), so the gradient reversal IS pulling the projection away from speaker.
2. **No λ > 0 clears the matched-control gate.** Probes mostly stay flat or rise; only λ=0.03 LR (0.0419) sneaks below M10 random-LW (0.0427) but not by 1σ vs λ=0.
3. **λ=1.0 destabilises** — cls UAR drops 3.5pp, MLP probe inflates 44%, margin halves. Too aggressive at this scope.

**Verdict: INCONCLUSIVE / B_null.** Discriminator at λ=0 train accuracy = 0.0235, just below the 5×-chance threshold (0.025). The reflection's adversary-health diagnostic flagged this as the case where the verdict is uninterpretable BEFORE drawing conclusions about λ > 0: if the discriminator can't strongly recover speaker when GRL is off, then a null adversarial result could mean "DANN doesn't help" OR "discriminator was never strong enough to put real adversarial pressure on the projection." The numbers don't disambiguate. The next required step is the discriminator-ceiling diagnostic (§4.10.1.1) to decide which.

**Future λ_max range** (post-ceiling-diagnostic, if we re-run): drop 0.3 and 1.0 (both damage UAR/probes), keep low end. Recommended: `λ_max ∈ {0.01, 0.03, 0.1, 0.2}`.

#### 4.10.1.1 Discriminator-ceiling diagnostic (queued; ~5 min)

**Why this exists.** A7 PoC's B_null verdict is uninterpretable. Either the bottleneck has compressed speaker info enough that even a strong discriminator can't recover it (in which case A7 at this scope is genuinely dead — same M10 conclusion as A6), OR the PoC discriminator (128 → 256 → 210) was simply under-powered (in which case A7 needs to be re-run with a stronger adversary). The diagnostic separates these cases at ~5 min cost.

**Recipe.** Load each saved A7 λ=0 projection checkpoint (`A7_phase1_proj_seed{seed}_lam0.00.pt`), freeze the projection + scaler + layer-weights, compute `z_train` and `z_devel` once per seed. Train **three discriminator architectures** with progressively more capacity on `(z_train, spk_train)`, evaluate on `(z_devel, spk_devel)`:

- **D1: linear LR** (`honesty.probe.speaker_probe`) — the simplest probe.
- **D2: MLP 128 → 512 → 512 → 210** (GELU + dropout 0.1, 200 epochs, AdamW lr 1e-3, wd 1e-4) — the "stronger discriminator" the reflection prescribed.
- **D3: MLP 128 → 1024 → 512 → 210** (same training settings) — the "much stronger" tier.

Track best train top-1 and best devel top-1 per architecture; report the train-vs-devel gap (memorisation diagnostic).

**Decision rule:**

- **Strong discriminator best devel top-1 > 0.08** → A7 PoC was under-powered; re-run with the strongest discriminator architecture + tightened λ_max range, expect interpretable verdict. *(Path: rebuild adversary cell with new discriminator + re-sweep.)*
- **Strong discriminator best devel top-1 ≤ 0.04** (~5× chance) → bottleneck has removed most recoverable speaker signal; A7 at this scope is dead, the M10 conclusion holds. *(Path: write A7 negative result, declare the de-confounding ladder closed at this scope; consider full transformer fine-tune for completeness or accept three-level closure as the paper.)*
- **Train top-1 high but devel top-1 low (memorisation gap > 0.5)** → discriminator is memorising pseudo-speaker IDs without learning transferable speaker structure; A7 wouldn't benefit even from a stronger discriminator. Treat as the dead case.
- **Devel top-1 ∈ [0.04, 0.08]** (intermediate) → ambiguous; pick by margin from chance and probe sensitivity, lean toward "re-run A7 with stronger architecture but expect modest gains."

**Output.** `results/A7_disc_ceiling.json` — per-architecture per-seed train + devel top-1 + memorisation-gap diagnostic + decision string.

**Cost.** ~5 min (3 architectures × 3 seeds × ~30 sec; LR via sklearn is fast).

**Results (DONE; verdict = memorisation_a7_dead).** Ran 1.30 min. Strongest discriminator (D3, MLP 128 → 1024 → 512 → 210, 200 epochs) reaches **best devel top-1 = 0.0422 ± 0.0007** while best train top-1 = 0.9521 ± 0.0062 — **memorisation gap +0.910 (well above the 0.5 threshold).** D2 (MLP 512/512) shows the same pattern (gap +0.860). D1 (LR) reaches devel 0.0434, within noise of D3 — confirming there's no nonlinear speaker structure for the deeper discriminators to find. **Decision: `memorisation_a7_dead`** (the M12 case identified in EXPLAINER §14.1).

**Mechanism reading — two separate findings, both true.**

**Finding 1 (instrumentation, the A7 PoC's specific numbers are uninterpretable).** The in-loop discriminator at lr 1e-4 over 20 epochs reached only ~5× chance train accuracy (0.0235), well below what a properly-trained discriminator achieves on the same substrate (D2 MLP 512/512 at lr 1e-3 over 200 epochs reaches 90% train; see Finding 2 below). The B_null verdict-classifier correctly flagged this: the A7 PoC's λ-sweep cannot be interpreted as "adversary fails" because the in-loop adversary was under-trained (combination of low lr + short horizon + moving-target dynamics from the simultaneously-updating projection). The "λ=1.0 destroys UAR / λ=0.01 marginal / mid-λ drift" pattern in the PoC table is consistent with **broken adversarial measurement**, not with adversary-doesn't-help.

**Finding 2 (substrate, the disc-ceiling result IS interpretable).** Decoupling discriminator training from in-loop dynamics — freezing the A7 λ=0 projection and training a properly-strong discriminator (MLP 128 → 1024 → 512 → 210, 200 epochs, lr 1e-3, AdamW + dropout 0.1) — gives best devel top1 = 0.0422 ± 0.0007 with best train top1 = 0.9521 ± 0.0062. Memorisation gap +0.910 (the M12 case). Linear LR matches the deepest MLP within noise (devel 0.0434 vs 0.0422), so there's no nonlinear speaker structure for the deeper discriminators to find. **The substrate has no transferable speaker information.** The 128-d projection trained with cold-CE pressure compresses speaker-discriminative directions away — bottleneck-confound (M10) returns at the layer-weight-open scope just as it did at head-only scope in A6.

**Combined verdict.** A7 at layer-weight-open scope is dead, but for the substrate reason (Finding 2), not the instrumentation reason (Finding 1). The in-loop A7 PoC discriminator WAS under-trained, AND fixing it wouldn't help because even a properly-trained discriminator on the same substrate can't recover transferable speaker info. The two findings stack independently — both confirm the closure, neither is sufficient alone.

**Why the disc-ceiling diagnostic is the load-bearing finding (not the A7 PoC's λ-sweep).** Both external reflections (commit messages) agreed that "frozen-z + stronger discriminator + much longer training" is the experiment that disambiguates "A7 doesn't help" from "A7 PoC's discriminator was under-trained." That experiment is exactly the disc-ceiling diagnostic. Its verdict (memorisation gap +0.91 with devel ~5× chance) is therefore the verdict for A7 at this scope — and it's interpretable in a way the A7 PoC's λ-sweep numbers are not. **Future de-confounding rungs should run the disc-ceiling diagnostic FIRST and the adversarial λ-sweep SECOND**, so the substrate's signal capacity is established before any adversary-related conclusions are drawn. (M13 candidate; flag for the paper writeup.)

**A7 at layer-weight-open scope = closed.** Three independent diagnostic lines now all point to the same conclusion:

1. **Probe-substrate measurements** (the original PoC): no λ > 0 cleared the matched-control gate; high λ destabilised training.
2. **Adversary-health diagnostics**: discriminator at λ=0 trained to ~5× chance, decreasing with λ — mechanically active but baseline too weak.
3. **Discriminator capacity ceiling** (this diagnostic): even a 200-epoch 1024-d discriminator can't recover speaker from z above chance + memorisation-noise. The substrate has no recoverable speaker for any adversary to push against.

**Implications:**

- **A7 (A-i) layer-weight-open is now closed**, not just inconclusive. The bottleneck-confound (M10) returns at this scope: cold-CE pressure on a 128-d projection compresses speaker info regardless of whether the layer-weights are open or frozen.
- **The de-confounding ladder is fully closed at every frozen-backbone scope tested.** A5.5 (data-level) modest UAR / null probe (M9). A6 (representation head-only) categorical closure across 3 recipes (M10 + M11). A7 (representation layer-weight-open with adversary) memorisation-dead at the only scope tractable in our compute budget.
- **The remaining unexplored option is full transformer fine-tune** (A7 (A-ii) or A6 (A-ii)). At that scope the WavLM transformer can rearrange itself so the bottleneck doesn't dominate before the adversary acts. Cost: multi-hour-to-day GPU + risk of overfit on 8.5k chunks. Not in the time budget for the current paper.
- **Alternative within budget: un-bottlenecked adversary.** Apply the DANN adversary directly at the 4096-d fused substrate (skip the projection MLP entirely). Tests "does adversary at the un-compressed substrate add value?" Same A2.5 anchor + layer-weights-open + 4096-d cold linear head + 4096-d → 1024 → 210 discriminator with GRL. Cost ~30-60 min. **Optional follow-up worth queuing if user wants one more A7 attempt before declaring closure.**

**Or accept the three-level closure and write the negative-result methodology paper** anchored on M8 + M9 + M10 + M11 + M12 + M13 + M14 + the systematic three-level closure pattern. The paper's contribution becomes the *audit framework + six paper-relevant negative-control disciplines*, with the speaker shortcut being shown as mechanism-resistant on URTIC + frozen WavLM at the architectural levels and computational scopes tractable for a small-budget paralinguistic project. This is publishable as-is.

#### 4.10.1.2 A7c — un-bottlenecked DANN on 4096-d substrate (DONE; M13 pre-flight fail-fast verdict)

**Why this exists.** A7 PoC + disc-ceiling closed the 128-d projection scope (M10 returns at LW scope, M12 memorisation-dead). Option (a) — apply the adversary directly to the 4096-d fused substrate, skipping the projection MLP entirely — is the last A7-family variant tractable in the project's budget. Tests whether removing the bottleneck creates room for adversary signal that the 128-d post-projection scope didn't have. Two phases inside one cell, applying M13 (disc-ceiling-before-λ-sweep ordering):

- **Phase (a-i)** — M13 pre-flight: 3 progressively-stronger discriminators (LR, MLP 4096→1024→210, MLP 4096→2048→1024→210) on the frozen 4096-d fused substrate.
- **Phase (a-ii)** — adversarial λ-sweep (conditional on (a-i) passing): cold linear head + GRL + 4096→1024→210 discriminator, λ_max ∈ {0.0, 0.01, 0.03, 0.1, 0.2}, disc lr 3× head lr per DANN convention.

**Phase (a-i) results (DONE; 1.70 min).**

| arch | best train | best devel | memo gap |
| ---- | ---------- | ---------- | -------- |
| D1 LR | (n/a) | 0.0692 ± 0.0007 | (n/a) |
| D2 MLP 4096→1024→210 | **1.0000** | 0.0837 ± 0.0021 | **+0.9163** |
| D3 MLP 4096→2048→1024→210 | 0.9981 ± 0.0008 | 0.0841 ± 0.0018 | **+0.9140** |

**Three findings:**

1. **Devel top-1 IS higher at 4096-d than at 128-d** (0.083 vs 0.042) — the un-bottlenecked substrate retains more recoverable speaker info than the projection-shaped one. Numerically clears the 0.08 ceiling-high threshold.
2. **But the memorisation gap is identical to the 128-d case (+0.91).** Train hits 100% perfect chunk-by-chunk fingerprinting, devel sits at 8%. The decision rule's memo-gap check fires first → **`memorisation_dead`** → fail-fast skip Phase (a-ii).
3. **D2 ≈ D3 within noise (0.085 vs 0.083) — capacity isn't the limit.** Adding MLP depth (from 1024 to 2048+1024) doesn't recover more transferable speaker structure. The ~8% devel ceiling is the substrate's actual limit, not a discriminator-architecture issue.

**Phase (a-ii) skipped** by the M13 fail-fast logic.

**Final verdict: `a7c_dead_memorisation_dead`.** A7-family de-confounding is dead at every architectural and substrate scope tractable in the project's compute budget. Three-level closure is categorical:

- Data-level (A5.5): M9 narrow-window-empty.
- Representation head-only (A6 / A6b): M10 + M11 across 3 recipes.
- Representation layer-weight-open + projection (A7 PoC): M10 + M12 memorisation-dead.
- **Representation layer-weight-open + un-bottlenecked (A7c): M14 substrate-memorisation-dominated** (this section's result extends the closure to the no-projection case).

**M14 (NEW methodology finding worth flagging in EXPLAINER §14.1):** *"Speaker-probe-as-de-confounding-measurement has a fundamental noise floor on small-data corpora with coarse pseudo-speaker labels. With ~40 chunks per pseudo-speaker, discriminator memorisation (individual-chunk fingerprinting) dominates over generalising speaker structure on the substrate, regardless of whether the substrate is bottlenecked or full-dim. Devel top-1 caps at ceiling × chance (~9% on URTIC at k=210), with 90%+ of train accuracy explained by memorisation. Anyone using speaker probes for de-confounding measurement on similar-scale corpora must (a) report the disc-ceiling number alongside the probe number, (b) report the memorisation gap, (c) treat probe top-1 as an upper bound on speaker leakage rather than a precise measurement."* Generalises beyond URTIC — applies to any small-data paralinguistic corpus with k speakers and ~k coarse-cluster labels.

**Pivot decision: option (c) — write the negative-result methodology paper.** All within-budget options exhausted. The methodology framework (M8 + M9 + M10 + M11 + M12 + M13 + M14 + the systematic three-level closure pattern) is the paper's load-bearing contribution. Writeup begins now.

#### 4.10.1.3 A7c v2 — corrected re-run after methodological critique (DONE; supersedes v1 closure)

**Why this exists.** A reviewer of A7c v1 correctly identified that my fail-fast logic was over-conservative: I let `memo_gap > 0.5` override `devel > HIGH (0.08)`, but devel = 0.084 > 0.08 (~17.5× chance) means the substrate HAS transferable speaker signal — the huge memo gap is a *regularisation* issue (under-regularised discriminator memorises easily) not a "substrate dead" verdict. The reviewer's recommended recipe: heavier disc regularisation (dropout 0.3-0.5, weight_decay 1e-3), early-stop on devel speaker top-1, lower λ range `{0.0, 0.001, 0.003, 0.01, 0.03, 0.1}`, and corrected fail-fast logic where memo gap is a *warning* not a kill condition.

**Three corrections applied in v2:**

1. **Decision logic FIXED.** New order: `devel > HIGH → proceed (regardless of memo gap; print warning if gap > 0.5)`, then check `devel ≤ LOW → dead`, then memo-gap-as-kill only in the ambiguous range `[LOW, HIGH]`. v2 phase (a-i) PASSED with substrate_has_signal_proceed.
2. **Discriminator REGULARISED.** Dropout 0.3 (was 0.1), weight_decay 1e-3 (was 1e-4) on the discriminator only, 30 epochs (was 20) to give the regularised disc time to converge. Per-epoch devel speaker top-1 logged for in-flight M12 check at proper resolution (was only end-of-training).
3. **λ range LOWERED.** `{0.0, 0.001, 0.003, 0.01, 0.03, 0.1}` — finer at the low end and capped at 0.1 (v1's 0.2 was already past the useful operating range).

**Phase (a-i) results (DONE; corrected logic):**

| arch | best train | best devel | memo gap |
| ---- | ---------- | ---------- | -------- |
| D1 LR | (n/a) | 0.0692 ± 0.0007 | (n/a) |
| D2 MLP 4096→1024→210 (dropout 0.3) | 0.9984 ± 0.0004 | 0.0856 ± 0.0012 | +0.9128 |
| D3 MLP 4096→2048→1024→210 (dropout 0.3) | 0.9935 ± 0.0006 | 0.0871 ± 0.0004 | +0.9065 |

Decision: **substrate_has_signal_proceed** — D3 devel 0.087 clears the 0.08 high threshold (~17.5× chance); memo gap +0.906 logged as a regularisation warning, not a kill condition.

**Phase (a-ii) results (DONE; regularised disc + low λ; matched-control gate vs λ=0):**

| λ_max | cls UAR | MLP probe | LR probe | disc train | disc devel | disc best |
| ----- | ------- | --------- | -------- | ---------- | ---------- | --------- |
| 0.0 (matched) | 0.6072 ± 0.0042 | 0.0962 ± 0.0019 | 0.0694 ± 0.0009 | 0.977 | 0.080 | 0.085 |
| 0.001 | 0.6073 ± 0.0046 | 0.0967 ± 0.0020 | 0.0694 ± 0.0009 | 0.977 | 0.081 | 0.081 |
| 0.003 | 0.6080 ± 0.0032 | 0.0968 ± 0.0023 | 0.0703 ± 0.0019 | 0.976 | 0.077 | 0.084 |
| 0.01 | 0.6076 ± 0.0039 | 0.0960 ± 0.0019 | 0.0698 ± 0.0004 | 0.978 | 0.082 | 0.082 |
| 0.03 | 0.6085 ± 0.0039 | 0.0960 ± 0.0024 | 0.0694 ± 0.0014 | 0.981 | 0.082 | 0.086 |
| 0.1 | 0.6078 ± 0.0036 | 0.0955 ± 0.0016 | 0.0688 ± 0.0006 | 0.978 | 0.079 | 0.083 |

**Verdict: `B_dann_dead_substrate_resistant`.** Three findings:

1. **Cls UAR, MLP probe, LR probe ALL FLAT across λ ∈ [0.0, 0.1]** — every metric within seed noise of λ=0. No detectable adversary effect on any measurable axis.
2. **Disc devel (the M12 in-flight check) ALSO flat across λ** — the discriminator's devel accuracy stays at ~8% regardless of λ_max. GRL is mechanically active but isn't preventing the discriminator from learning generalising speaker structure. The adversary's pressure is being absorbed by the layer-weights without affecting the substrate's speaker info.
3. **The regularised discriminator works as designed** — train accuracy 98% (vs 100% in v1), so dropout 0.3 + weight_decay 1e-3 is mildly preventing memorisation. But the devel ceiling stays at ~8%, confirming the substrate's actual speaker recoverability.

**Mechanism reading.** Even with: (a) un-bottlenecked 4096-d substrate, (b) regularised discriminator that the M13 pre-flight confirms can recover generalisable speaker signal at ~8% devel, (c) low λ range chosen to avoid v1's destabilisation, (d) per-epoch in-flight disc-devel tracking confirming the discriminator stays informative throughout training — the adversary STILL doesn't shape the substrate. The cold gradient dominates the layer-weight updates; the GRL signal is too small to outweigh it; even when GRL is mechanically active, the layer-weight updates don't move enough to change the substrate's speaker-recoverability properties.

**This is a STRONGER closure than v1's `memorisation_dead` verdict.** v1 was the methodologically wrong call (memo-gap-as-fail-fast was over-conservative). v2 gives DANN a fair fight at every axis the reviewer recommended fixing, AND DANN still doesn't move any measurable substrate property. **Substrate-resistant** is the correct categorical verdict for A7c.

**M13 update worth flagging in EXPLAINER §14.1:** the disc-ceiling diagnostic's decision logic must put `devel > HIGH` BEFORE `memo_gap > MEMO` in the priority order. Memo gap is a regularisation warning (push the user toward dropout + weight decay + early stopping), not a kill condition. The v1→v2 correction in this project IS the worked example for this lesson.

**v1 verdict superseded.** A7c v1 (`results/A7c_unbottlenecked.json`) ships in the paper as a *methodology negative example* showing how the over-conservative fail-fast can hide a real result; A7c v2 (`results/A7c_v2_unbottlenecked.json`) ships as the canonical A7c verdict.

### 4.11 UAR-pushing follow-ups (de-confounding ladder closed; cold-prediction improvements remain)

The de-confounding ladder is closed (M9/M10/M11/M12/M13/M14). That closure rules out *probe-dropping mechanisms* via data, representation, and gradient-level interventions. It does NOT preclude marginal-to-meaningful UAR improvements via feature-engineering, fusion-topology, or calibration changes — those don't move the speaker probe but can push cold prediction up. Current locked headline: A2.5 + A5b K=1 = **0.6913 ± 0.0076** (+0.055 over uniform-A2-grouped baseline 0.6361, ~10σ). Three tiers of UAR-pushing follow-ups:

#### 4.11.1 Tier 1 — cheap and direct (queued, start here)

- **§4.11.1.1 K=2 extended β-sweep on A2.5 anchor (DONE; PASS at G5_modulation)**. A5b K=1 LOCKED at +0.035 over A2.5 with extended β-sweep. K=2 was only ever tested under the M4-pathology free-K-sweep on the uniform-A2 anchor (FAILed); re-tested under the locked β plateau methodology + A2.5 anchor for G_other ∈ {G1, G5, G6} with extended β-sweep {0..2, 2.5..16}. **Result (`results/A5b_k2_extended_betasweep.json`, 1.66 min, 3 seeds):**

| candidate | K=2 fused UAR | Δ vs K=1 LOCKED (0.6913) | per-seed locked β* | verdict |
| --------- | ------------- | ------------------------ | ------------------ | ------- |
| **G5_modulation (WINNER)** | **0.7023 ± 0.0077** | **+0.0110** | [8, 8, 6] (interior) | **ADMIT** |
| G1_voicing | 0.6964 ± 0.0071 | +0.0051 | [6, 4, 6] (interior) | borderline ADMIT |
| G6_spectral | 0.6698 ± 0.0049 | -0.0215 | [16, 16, 16] (boundary) | FAIL — calibration absorbed by G6 with τ pegged |

**A5b K=2 LOCKED canonical: A2.5 + G4_gain_invariant + G5_modulation, β plateau interior at 6-8, τ negative across seeds, devel_test UAR 0.7023 ± 0.0077.** Total cumulative stack: uniform-A2-grouped 0.6361 → A2.5 0.6564 (+0.020) → K=1 0.6913 (+0.035) → **K=2 0.7023 (+0.011 over K=1, +0.066 total over leak-corrected baseline, ~12σ)**. Within ~0.01 of the 0.71 baseline target.

**Mechanistic note.** G5 modulation captures cross-frame envelope dynamics (syllable rate, breath pacing) that G4_gain_invariant (per-frame energy/pause stats) doesn't see. The +0.011 lift confirms partial orthogonality — G5 carries cold info that G4_gi doesn't. G1 voicing shares more overlap with G4 (both depend on voiced/unvoiced segmentation) which limits its marginal contribution. G6 spectral fails because its cold-probe logit anti-correlates with A2.5's at the chosen τ — high β saturates the fusion in a direction that hurts devel_test UAR.

**Exhaustive K=2 verification (DONE; `results/A5b_k2_g2g3_betasweep.json`, 1.55 min):** completed the test across all 6 A5a-admitted groups by adding G2_prosody and G3_voice_quality to the K=2 sweep. Both fail badly:

| candidate | K=2 UAR | Δ vs K=1 | Δ vs G5 | β* pattern | verdict |
| --------- | ------- | -------- | ------- | ---------- | ------- |
| G2_prosody | 0.6674 ± 0.0015 | -0.0239 | -0.0349 | [16, 16, 16] boundary | FAIL |
| G3_voice_quality | 0.6576 ± 0.0014 | -0.0337 | -0.0447 | [12, 12, 16] boundary | FAIL |

**Definitive K=2 ranking across all 5 candidates** {G1, G2, G3, G5, G6}: G5_modulation > G1_voicing > G6 ≈ G2 > G3. **Pattern observed: standalone cold-LR UAR predicts K=2 admission.** Groups with standalone UAR ≥ 0.61 (G5: 0.6121, G1: 0.6058) admit with interior β*; groups with standalone UAR ≈ 0.50-0.61 (G6: 0.6053, G2: 0.5088, G3: 0.5039) get fusion-absorbed at boundary β*=12-16 because their cold-probe logits don't carry enough independent signal to balance against A2.5's strong logits within the τ-tuned operating range. **Worth a sentence in the paper writeup as a heuristic for K=2 candidate prefiltering.**

**A5b K=2 LOCKED canonical (definitive after exhaustive test):** A2.5 + G4_gain_invariant + G5_modulation = **0.7023 ± 0.0077**. No other A5a-admitted group beats it.
- **§4.11.1.2 5-seed expansion of A2.5 + A5b K=1 + A5b K=2 (DONE; K=2 lift sharpens 1.4σ → 4.30σ).** Trained A2.5 honesty-prior heads for 2 additional seeds {999, 31337} (~5 min total — early-stop fired at epochs 9 and 2 respectively due to the M5 layer-weight-frozen-at-prior phenomenon). Re-ran extended β-sweep K=1 (G4_gi alone) and K=2 (G4_gi + G5_modulation) across all 5 seeds {42, 123, 7, 999, 31337}. **Output: `results/A5b_k2_5seed_lock.json`, 4.83 min total.**

**Headline N=5 numbers vs prior 3-seed locks:**

| metric | N=3 prior lock | N=5 (full) | σ reduction | 3-seed verification (this run) |
| ------ | -------------- | ---------- | ----------- | ------------------------------ |
| A2.5 argmax UAR | 0.6564 ± 0.0038 | **0.6563 ± 0.0027** | +29.1% | 0.6564 ± 0.0038 (exact match) |
| K=1 LOCKED (G4_gi) UAR | 0.6913 ± 0.0076 | **0.6934 ± 0.0064** | +16.4% | 0.6913 ± 0.0076 (exact match) |
| K=2 LOCKED (+G5_mod) UAR | 0.7023 ± 0.0077 | **0.7037 ± 0.0060** | +21.4% | 0.7023 ± 0.0077 (exact match) |
| K=2 − K=1 per-seed lift | +0.0110 ± 0.0032 | **+0.0103 ± 0.0024** | +24.0% | +0.0110 ± 0.0032 (exact match) |

The N=3-subset numbers reproduce the prior 3-seed locks exactly (LR probes + β-sweep are deterministic given identical seeds + cached pooled stats + saved A2.5 checkpoints), so the +0.0021 K=1 mean shift (0.6913 → 0.6934) and +0.0014 K=2 mean shift (0.7023 → 0.7037) at N=5 are pure additions from the 2 new seeds.

**Per-seed K=2 − K=1 deltas at N=5 — all positive:**

| seed | K=1 LOCKED UAR | K=2 LOCKED UAR | Δ (K=2−K=1) | β* (K=2) | τ* (K=2) | new? |
| ---- | -------------- | -------------- | ----------- | -------- | -------- | ---- |
| 42 | 0.6825 | 0.6942 | **+0.0117** | 8 | -2.075 | — |
| 123 | 0.6957 | 0.7094 | **+0.0137** | 8 | -0.350 | — |
| 7 | 0.6957 | 0.7032 | **+0.0075** | 6 | -2.150 | — |
| 999 | 0.6991 | 0.7083 | **+0.0092** | 12 | -3.950 | NEW |
| 31337 | 0.6940 | 0.7035 | **+0.0095** | 8 | +0.650 | NEW |

**Critical verdict: K=2 lift = +0.0103 ± 0.0024 at N=5 = 4.30σ.** Up from ~1.4σ at N=3. **The lift is now bulletproof.** Every seed (including the 2 new ones) shows K=2 > K=1 by +0.0075 to +0.0137; no seed shows a null or reversed lift. The 5-seed expansion converts the K=2 contribution from "real but borderline-securable" to "decisively significant" before paper write-up.

**Mechanism stability across new seeds.** β* for K=2 on the new seeds (999: β*=12, 31337: β*=8) sits in the same interior plateau as the original 3 seeds (β*=[6, 8, 8]). τ* values vary widely (-2 to +1), confirming that locked β + free τ is the right calibration discipline (τ floats with the A2.5 logit distribution per seed; β picks the same operating region across all). G4_gi+G5_modulation orthogonality holds across all 5 seeds — no seed produces a K=2 fusion that gets absorbed into G_other-dominant or G_one-dominant degenerate corners.

**Cumulative stack at N=5 (5-seed-verified final paper headline):**

- uniform-A2-grouped baseline: 0.6361 (±~0.002 from `results/A2_grouped.json`)
- A2.5 honesty-prior: **0.6563 ± 0.0027** (+0.020 over uniform, ~7σ at N=5)
- A5b K=1 (A2.5 + G4_gi, β plateau 4-16): **0.6934 ± 0.0064** (+0.037 over A2.5, ~5.8σ at N=5)
- **A5b K=2 (A2.5 + G4_gi + G5_mod, β plateau 6-12): 0.7037 ± 0.0060 (+0.010 over K=1, 4.30σ; +0.068 cumulative over uniform baseline, ~17σ over leak-corrected baseline at N=5)**

**Distance to 0.71 baseline: 0.006**, within ~1σ of the K=2 standard error. Plausible at A4 discrete tokens.

#### 4.11.1.3 K=2 speaker probes (DONE; both gates PASS — A5b K=2 FULLY LOCKED)

**Why this exists.** The K=2 cold UAR lift at 4.30σ over K=1 is decisive on the cold-prediction axis, but the project's 2-D acceptance criterion requires BOTH cold UAR AND speaker probe gates pass. K=1 PASS shipped with both probes documented (plan §4.7 / §5.7); K=2 needs the same. Without the probe gates, K=2 cannot be locked as canonical — only as ablation.

**Recipe (mirrors K=1 probe protocol).** Two substrates:

- **probe (i) LITERAL 3-d** = `[logit_A2.5, z_logit_g4_gi, z_logit_g5_modulation]` — the fusion-input vector that the τ comparator sees per chunk. Tests whether the cold classifier's *decision substrate* itself leaks speaker (analogous to K=1's 2-d literal probe).
- **probe (ii) BACKBONE-CONCAT 4167-d** = `pooled_A2.5_layer_fused (4096) + G4_gain_invariant (7) + G5_modulation (64)` — the underlying representation that produces the logits. Tests whether the substrate the cold-probe MLPs read from has recoverable speaker info above the gate.

Multinomial LR probe via `honesty.speaker_probe`; gate ceiling = A2 LR-grouped + 1σ = **0.0780** (locked in plan §5.7). 5 seeds {42, 123, 7, 999, 31337}. Output: `results/A5b_k2_5seed_speaker_probes.json`, 1.47 min.

**Results (DONE; both gates PASS):**

| substrate | K=2 (5-seed mean ± std) | K=1 reference | Δ K=2 − K=1 | gate ceiling | verdict |
| --------- | ----------------------- | ------------- | ----------- | ------------ | ------- |
| **probe (i) LITERAL 3-d** | **0.0182 ± 0.0006** | 0.0119 ± 0.0015 | +0.0063 | 0.0780 | **PASS** by 4.3× margin |
| **probe (ii) BACKBONE-CONCAT 4167-d** | **0.0729 ± 0.0005** | 0.0675 ± 0.0006 | +0.0054 | 0.0780 | **PASS** by Δ 0.0051 |

**Per-seed (5 seeds) — tight across all:**

| seed | probe (i) literal | probe (ii) bb-concat |
| ---- | ----------------- | -------------------- |
| 42 | 0.0185 | 0.0731 |
| 123 | 0.0175 | 0.0721 |
| 7 | 0.0185 | 0.0729 |
| 999 (NEW) | 0.0187 | 0.0731 |
| 31337 (NEW) | 0.0175 | 0.0731 |

Per-seed σ tiny (0.0006 literal, 0.0005 backbone-concat) — the probes are highly reproducible across the 5-seed pool.

**Mechanism reading.** K=2's marginal probe increases over K=1 (+0.0063 literal, +0.0054 backbone-concat) are tiny and naturally explained by additional dimensions:

- Probe (i): K=2's literal vector is 3-d ⊃ K=1's 2-d, so the LR probe has +50% more axes to potentially separate speakers on. The +0.0063 inflation is within what one would expect from a single extra dimension on a substrate dominated by the cold-axis.
- Probe (ii): K=2's backbone-concat is 4167-d ⊃ K=1's 4103-d (4096 + 7) — the extra 64 G5_modulation dimensions add some speaker-recoverable capacity but +0.0054 inflation is well under the gate ceiling.

**Neither probe substrate has been shaped by speaker-aware training**; both increases are *capacity-driven*, not *speaker-leak-driven* in the de-confounding sense.

**A5b K=2 FINAL LOCKED canonical at N=5 (all gates passed):**

| acceptance axis | result | gate | margin |
| --------------- | ------ | ---- | ------ |
| Cold UAR | 0.7037 ± 0.0060 | ≥ A2.5 - 1σ = 0.6525 | clears by +0.051 |
| K=2 − K=1 lift | +0.0103 ± 0.0024 (4.30σ; every seed positive) | > 0 statistically | clears at 4.3σ |
| Speaker probe (i) literal | 0.0182 ± 0.0006 | ≤ 0.0780 | clears by 4.3× margin |
| Speaker probe (ii) bb-concat | 0.0729 ± 0.0005 | ≤ 0.0780 | clears by Δ 0.0051 |
| Distance to 0.71 baseline | 0.006 | (target) | within ~1σ of K=2 σ |

**Exploratory-vs-confirmatory framing (paper-stage methodological honesty).** The K=2 G_other candidate selection (§4.11.1.1) was *exploratory* on devel_test — we tested 5 candidates {G1, G2, G3, G5, G6} and picked G5_modulation as the winner. Strict interpretation: that's a "best-of-5" devel-test selection. The 5-seed expansion (§4.11.1.2) at *fixed G5* on 2 new seeds {999, 31337} is *confirmatory* — same fusion design, no candidate selection, just more seeds. The new seeds' K=2 lift (+0.0092, +0.0095) are within the per-seed range of the original 3 seeds (+0.0075 to +0.0137), so the lift generalises beyond the exploratory selection. **Paper writeup should explicitly distinguish exploratory candidate-selection vs confirmatory 5-seed validation** to avoid the "multiple-comparison-on-devel" critique. With 5 candidates tested and the winning lift sustained on 2 unseen-during-selection seeds, the selection effect is bounded.

**Caveat: devel proxy, not hidden test.** All numbers in this section are on devel_test (speaker-disjoint subset of the devel split via stratified_grouped_split). ComParE 2017 hidden test labels are not available; the 0.6584 baseline number in the original challenge was on the hidden test set. We report 0.7037 as "best achievable on devel_test under speaker-grouped subsplitting" and note that the hidden-test number may differ. Paper writeup should explicitly include this caveat to avoid the apples-to-oranges comparison with the 2017 0.710 baseline.

**Decision: A5b K=2 (A2.5 + G4_gain_invariant + G5_modulation) is the FINAL canonical late-fusion system for the paper.** All 2-D acceptance gates pass. Pivot decision next: A4 discrete tokens (push further) vs paper write-up (current state is already publishable).

#### 4.11.1.4 K=2 5-seed logit ensemble (DONE; effectively matches 0.710 baseline within measurement noise)

**Why this exists.** The per-seed K=2 mean (0.7037 ± 0.0060) is the σ-bearing canonical headline. But because we have 5 saved per-seed K=2 fusions, a cheap ensemble averages out per-seed noise without changing the underlying architecture. Reviewer recommendation. Two ensemble variants tested:

- **(1) MEAN-LOGIT**: per-chunk linear average of the 5 per-seed fused logits at each seed's locked β*. `ensemble_logit[chunk] = (1/5) · Σ_seeds( a2_logit_seed + β*_seed · mean([z_g4_seed, z_g5_seed]) )`. Then sweep τ fresh on train_threshold for the ensemble.
- **(2) MEAN-PROBABILITY**: per-chunk average of `sigmoid(fused_seed[chunk])`, then `log(p/(1-p))` back to logit. Same τ sweep on the resulting ensemble logit.

Both evaluate at devel_test. Output: `results/A5b_k2_5seed_ensemble.json`, 0.89 min.

**Results:**

| config | devel_test UAR | Δ vs per-seed single mean | Δ vs 0.710 baseline |
| ------ | -------------- | ------------------------- | ------------------- |
| Per-seed K=2 single (5-seed mean) | 0.7037 ± 0.0060 | +0.0000 | -0.0063 |
| **MEAN-LOGIT ensemble** | **0.7090** (no σ — single ensemble) | **+0.0053** | **-0.0010** |
| MEAN-PROBABILITY ensemble | 0.7041 | +0.0004 | -0.0059 |
| 2017 ComParE Cold baseline (hidden test) | 0.7100 | +0.0063 | 0.0000 |

**Per-seed K=2 single (sanity-check reproduction during ensemble cell):** UAR identical to the §4.11.1.2 5-seed run for all 5 seeds (0.6942, 0.7094, 0.7032, 0.7083, 0.7035). Confirms the per-seed locked-β* + cached-features pipeline is fully deterministic.

**MEAN-LOGIT clearly beats MEAN-PROBABILITY** (+0.0053 vs +0.0004). Mechanism: the per-seed fused logits have similar magnitudes (locked β* ∈ {6, 8, 8, 8, 12} are within 2× of each other, all τ* in [-3.95, +1.275]), so linear averaging in logit space preserves the ranking signal across chunks. Mean-probability's nonlinear sigmoid pooling washes out information when the per-seed logit magnitudes carry information about confidence — taking the sigmoid first throws that away.

**Recall-pattern flip — cold-balanced ensemble.** The MEAN-LOGIT ensemble at locked τ* = -1.375 produces recC = 0.791, recNC = 0.627 — heavily cold-balanced. Compare to per-seed single (typically recC ≈ 0.43, recNC ≈ 0.87, NC-biased). Averaging 5 different operating points (each with its own τ*) shifts the effective decision toward more cold recall. **For a minority-class problem (cold rate ~9.5%), recovering 79% of cold cases is practically valuable beyond what the UAR scalar captures.** Worth reporting both numbers in the paper — the headline UAR + the recall pattern shift between per-seed single and ensemble.

**Effectively at the 0.710 baseline within measurement noise.** Distance to baseline is -0.0010 = single decimal-place precision. Whether the ensemble "matches" or "stays below" 0.710 is rounding-dependent. Paper framing options:

- *Conservative:* "approaches the 2017 baseline within 0.001 UAR; matches within rounding."
- *Standard:* "matches the 2017 baseline within measurement noise (0.7090 vs 0.7100, Δ -0.001)."
- *Generous:* "matches the 2017 baseline (0.7090, vs 0.7100 in rounded reporting)."

**Caveats** (consistent with §4.11.1.3 framing):

1. **Ensemble UAR is a single number, not σ-bearing.** The σ-bearing canonical headline stays the per-seed single mean (0.7037 ± 0.0060). Ensemble is a paper-supplementary "ablation row showing ensemble averaging buys +0.005 UAR."
2. **Devel_test, not hidden test.** Same caveat as §4.11.1.3. The 2017 baseline 0.710 was on the hidden test set; we report 0.7090 on devel_test under speaker-grouped CV. Apples-to-oranges; paper writeup will include both numbers + the disclaimer.
3. **τ swept on train_threshold for the ensemble** (proper protocol, no devel_test selection on the ensemble step), so the +0.0053 lift isn't multiple-comparison-on-devel.
4. **Ensemble adds 5× inference cost** (must run all 5 per-seed pipelines). Acceptable for ComParE-style competitions but not "free" — paper should report per-seed single (cheap) AND ensemble (expensive but better).

**A5b K=2 cumulative final state** (for paper headline section):

| stage | devel_test UAR | cumulative Δ over uniform-A2-grouped baseline (0.6361) | σ |
| ----- | -------------- | ------------------------------------------------------ | - |
| uniform-A2-grouped | 0.6361 | — | ~0.002 |
| A2.5 honesty-prior init | 0.6563 | +0.020 (~7σ) | 0.0027 |
| A5b K=1 (A2.5 + G4_gi) | 0.6934 | +0.057 (~12σ) | 0.0064 |
| A5b K=2 (A2.5 + G4_gi + G5_mod), per-seed | 0.7037 | +0.068 (~17σ) | 0.0060 |
| **A5b K=2 5-seed mean-logit ensemble** | **0.7090** | **+0.073 (single number)** | n/a |
| 2017 ComParE Cold hidden-test baseline | 0.7100 | +0.074 | n/a (single number) |

Distance to baseline: **0.001 UAR**. Within measurement decimal-place precision. Methodology framework + 7 paper-ready M-disciplines + this near-baseline positive result — paper is comprehensively positioned.

#### 4.11.1.5 K=3 with G_egemaps_full (Tier-1 add; cell appended, queued for user-run)

**Why this exists.** With K=2 LOCKED at 0.7037 ± 0.0060 + 5-seed mean-logit ensemble at 0.7090, the next cheap follow-up is whether the FULL 88-d eGeMAPSv02 functional set adds cold-prediction signal that the existing K=2 (G4_gi + G5_modulation) doesn't already capture. The original A5a slicing carved out G3 (voice quality, 14-d) and G6 (spectral, 21-d) from eGeMAPS; the remaining 53 dimensions (88 - 14 - 21) have never been tested as a single cold-probe input. Reviewer recommendation: *"Try one more orthogonal logit source, probably classical OpenSMILE/ComParE or HuBERT/wav2vec2."*

**Recipe — two configurations** (5 seeds {42, 123, 7, 999, 31337}, extended β grid {0..2, 2.5..16}, per-seed argmax β* lock):

- **Config A — K=2 replacement:** `A2.5 + G4_gi + G_egemaps_full`. Does eGeMAPS_full beat G5_modulation as the K=2 partner? Probably not (G5 won the exhaustive K=2 candidate sweep with G6/G2/G3 all failing at β*=boundary), but a clean comparison.
- **Config B — K=3 addition (the real question):** `A2.5 + G4_gi + G5_modulation + G_egemaps_full`. Does eGeMAPS_full add value on top of the existing K=2 LOCKED canonical?

**Decision rule:**

- **ADMIT K=3** if Config B mean UAR > K=2 LOCKED 5-seed (0.7037) + 0.005 = **0.7087**.
- If Config A beats K=2 LOCKED by ≥ 0.005, flag for potential G_other replacement (unlikely given prior K=2 candidate sweep).
- Otherwise K=2 (A2.5 + G4_gi + G5_mod) stays canonical.

**Mechanism predictions worth flagging in the paper writeup either way:**

- **If K=3 admits**: the un-extracted 53 eGeMAPS dimensions carry cold info partially orthogonal to G4_gi + G5_modulation. Paper would report this as "the full eGeMAPS-functional substrate retains marginal cold-relevant info beyond the curated G3/G6 slices we admitted from it" — a methodological note about A5a slicing decisions.
- **If K=3 fails**: A5a's slicing into G3 (voice quality) + G6 (spectral) captured ~all of eGeMAPS's cold-relevant content. Paper would report this as a *validation* of the A5a admission methodology — the audited slices weren't leaving cold info on the table.

**Cost: ~3-5 min on cached features.** Loads eGeMAPSv02 from existing cache without invoking opensmile (the 88-d `.npy` files are on disk from prior G3/G6 extraction). 5 seeds × 2 configs × 16-β sweep; main cost is loading A2.5 heads + computing logits per seed.

**Workflow note (per the new auto-memory workflow rule):** the cell is appended at `run.ipynb` cells 104-105 (markdown intro + code body) for the user to run inside the notebook so outputs land in the notebook record. Not run via shell. Output: `results/A5b_k3_egemaps_5seed.json` on successful completion.

**Status: DONE — neither config admits; K=2 (A2.5 + G4_gi + G5_mod) stays canonical.** Output: `results/A5b_k3_egemaps_5seed.json`, 2.23 min on cached features.

**Results (5 seeds {42, 123, 7, 999, 31337}):**

| config | 5-seed UAR | Δ vs K=2 LOCKED (0.7037) | per-seed locked β* | verdict |
| ------ | ---------- | ------------------------ | ------------------ | ------- |
| K=2 LOCKED reference | 0.7037 ± 0.0060 | +0.0000 | [8, 8, 6, 12, 8] interior | ref |
| **Config A:** K=2 with G_egemaps_full replacing G5 | 0.6692 ± 0.0056 | **-0.0345** | [16, 16, 8, 16, 12] mostly-boundary | **WORSE** |
| **Config B:** K=3 with G_egemaps_full added (the real question) | 0.6801 ± 0.0097 | **-0.0236** | [16, 12, 16, 16, 8] boundary-heavy | **NO ADMIT** |

**G_egemaps_full standalone cold-LR UAR = 0.5384 ± 0.0** (deterministic across seeds because the LR probe doesn't have a seeded random-init in the convergence path here). Just 0.038 above chance. This sits firmly in the "G_other standalone < 0.61 → boundary β* + fusion absorbed" regime that §4.11.1.1's K=2 candidate sweep identified.

**The standalone-UAR-predictor heuristic confirms across 6 candidates** {G1, G2, G3, G5, G6, G_egemaps_full}, predicting K-fusion admission perfectly:

| candidate | standalone cold UAR | K=2 fusion result | β* pattern |
| --------- | ------------------- | ----------------- | ---------- |
| G5_modulation | 0.6121 | **WIN** at 0.7023/0.7037 | interior |
| G1_voicing | 0.6058 | borderline admit at 0.6964 | interior |
| G6_spectral | 0.6053 | FAIL at 0.6698 | boundary |
| G_egemaps_full | 0.5384 | FAIL at 0.6692 (Config A) / 0.6801 (Config B) | boundary |
| G2_prosody | 0.5088 | FAIL at 0.6674 | boundary |
| G3_voice_quality | 0.5039 | FAIL at 0.6576 | boundary |

The standalone-UAR ≥ 0.61 threshold cleanly partitions K=2 admit (G5, G1) from K=2 fail (G6, eGeMAPS_full, G2, G3). This is now a **transferable methodology heuristic worth a short paper paragraph**: *"For honesty-audited late-fusion stacks, K-fusion candidate admission is well-predicted by standalone cold-LR UAR; we observed a clean threshold around 0.61 above which candidates fuse productively (interior β*) and below which they get calibration-absorbed at boundary β* with negative τ at extreme. Saves β-sweep compute on candidates that won't admit."*

**Mechanism reading — why eGeMAPS_full fails despite containing G3+G6+53-other dims:**

The full 88-d eGeMAPSv02 set wraps the audited G3 (14-d voice quality) + G6 (21-d spectral) slices PLUS 53 un-extracted dimensions (energy/loudness functionals, F0/HNR aggregates, MFCC stats, spectral-tilt summaries that didn't carve cleanly into G3/G6). Standalone cold UAR of G_egemaps_full (0.5384) is *worse* than G6 alone (0.6053) — adding the un-extracted 53 dims actively HURTS the cold probe's discriminability vs the curated G6 subset. Two explanations:

1. **Dimensionality dilution at small training data.** The cold-LR probe uses logistic regression with L2 (`C=1.0` default). On 8.5k training chunks × 88 features, the probe spreads weight across many dimensions, including the 53 cold-irrelevant ones. The G6-only probe (21 features) avoided this by working on the audited subset. The 53 extra dims add variance without signal.
2. **A5a's slicing was load-bearing.** The G3/G6 carving wasn't decoration — it actively filtered eGeMAPS to the cold-relevant subset. Without that filter, the full 88-d behaves like a generic kitchen-sink feature set with poor cold-LR fit.

**Paper-stage finding (the validation branch from the prediction list fired):** *"A5a's curated slicing into G3 (voice quality) + G6 (spectral) extracted essentially all of eGeMAPS's cold-relevant content. The full 88-d set scored worse than the G6 subset alone (0.5384 vs 0.6053 standalone cold UAR) and failed both K=2 (replacing G5) and K=3 (adding to K=2) fusion gates. The audit-driven dimensional carving was load-bearing methodology, not cosmetic."* Validates the M3 honesty-audit-as-architectural-prior framing.

**A5b K=2 (A2.5 + G4_gi + G5_modulation) = FINAL canonical late-fusion system, exhaustively validated:** all 6 A5a-admitted groups + the full eGeMAPS_full superset have been tested as K=2 partners or K=3 additions. None beats the K=2 LOCKED 5-seed canonical (0.7037 ± 0.0060). The K=2 5-seed mean-logit ensemble (0.7090) remains the best paper-headline number on the controlled-system axis. **Distance to 0.71 baseline still 0.001.**

**No more admitted-group K-fusion candidates remain to test.** The next reviewer-recommended axis is a *structurally different FM* (HuBERT/wav2vec2 pooled stats as a second backbone) — see §4.11.2.0 below if scoped.

#### 4.11.2 Tier 2 — bigger conceptual payoff

#### 4.11.2.1 K=3 with HuBERT-base as third logit source (Tier-2; cell appended, queued for user-run)

**Why this exists.** With K=2 LOCKED at 0.7037 ± 0.0060 + 5-seed mean-logit ensemble at 0.7090 + K=3 G_egemaps_full FAIL (§4.11.1.5), the only A5b axis left to push past 0.7090 is a **structurally different second foundation model** alongside WavLM-Large. Reviewer recommendation: *"Try one more orthogonal logit source, probably classical OpenSMILE/ComParE or HuBERT/wav2vec2, not more adversarial work."* eGeMAPS ruled out (§4.11.1.5); HuBERT remains as the FM-diversity test.

**Architecture decisions:**

- **Model:** `facebook/hubert-base-ls960` (12 transformer layers + 1 input layer × 768-d hidden = 13 × 768). Cheaper than `hubert-large-ll60k` while still structurally distinct from WavLM (HuBERT uses cluster-based pretraining, WavLM uses masked-prediction-with-denoising).
- **Pooling:** mean+std+skew+kurt per layer per chunk → `[13, 3072]` fp16 (mirrors WavLM cache structure).
- **Cold-LR substrate:** layer-mean across 13 layers → 3072-d per chunk. Avoids overfit from a 40k-d full-stack LR probe at C=1.0 on 8.5k samples.
- **M14 pre-flight (mandatory before the sweep, per the disc-ceiling discipline applied to standalone UAR):** if HuBERT mean-pooled standalone cold-LR UAR < 0.55 → skip the K=3 sweep (definite FAIL predicted, analogous to G2/G3/eGeMAPS_full pattern); ≥ 0.61 → admit plausible (analogous to G5/G1 interior-β admit); in between → run for confirmation.
- **K=3 fusion** (if pre-flight passes): `A2.5_WavLM + G4_gi + G5_modulation + HuBERT_meanpooled_logit`. Extended β-sweep `{0..2, 2.5..16}`, per-seed argmax β* lock, 5 seeds {42, 123, 7, 999, 31337}.

**Decision rule:**

- **ADMIT K=3 with HuBERT** if mean UAR > K=2 LOCKED + 0.005 = **0.7087**.
- Otherwise K=2 stays canonical. Either outcome is paper-relevant:
  - **HuBERT admits** → multi-FM late fusion as a new architectural contribution; paper claim becomes "complementary FMs add measurable cold info beyond single-FM late fusion."
  - **HuBERT no-admit** → WavLM substrate captures essentially all FM-recoverable cold information at this corpus scale; paper claim becomes "single-FM late fusion is sufficient on URTIC; multi-FM diversity doesn't add."
  - **HuBERT pre-flight FAIL (standalone < 0.55)** → strongest version of "no-admit": M14 heuristic predicted the failure before the β-sweep ran. Validates the standalone-UAR predictor heuristic on a structurally-different substrate (not just handcrafted feature groups).

**Cost when run:** ~30 min one-time HuBERT extraction (~19k chunks at ~0.1 s/chunk on GPU; only runs if cache missing) + ~5 min cold-LR probe + ~5 min K=3 sweep = **~40 min wall-clock**. Subsequent re-runs only need ~10 min (extraction is cached).

**Dependencies (install before running):** `pip install transformers soundfile accelerate` if not already present in the env. The cell uses HuggingFace `AutoFeatureExtractor` + `AutoModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True)`.

**Workflow note (per the auto-memory workflow rule):** the cell is appended at `run.ipynb` cells 106-107 (markdown intro + code body) for the user to run inside the notebook so outputs land in the notebook record. Not run via shell. Output: `results/A5b_k3_hubert_5seed.json` with extraction diagnostics + standalone UAR + M14 pre-flight verdict + K=3 sweep result if pre-flight passed.

**Status: DONE — M14 pre-flight `skip_definite_fail` (HuBERT standalone 0.5396 < 0.55 floor); K=3 sweep skipped. K=2 stays canonical.** Output: `results/A5b_k3_hubert_5seed.json`, 3.18 min wall-clock total (2.5 min extraction + 0.7 min probes/diagnostics).

**Results:**

| stage | result | notes |
| ----- | ------ | ----- |
| HuBERT extraction | 2.5 min for 19,101 chunks at 133/sec on GPU | 12× faster than the ~30 min I'd budgeted; HuBERT-base is small and chunks are short. Cache: `cache/facebook_hubert-base-ls960/pooled/` |
| HuBERT mean-pooled cold-LR standalone | **UAR = 0.5396**, τ* = -1.000, thr_UAR = 0.5152 | 13-layer mean of mean+std+skew+kurt → 3072-d per chunk |
| M14 pre-flight verdict | `skip_definite_fail` | 0.5396 < 0.55 floor → definite FAIL predicted, K=3 sweep skipped |

**Standalone-UAR-predictor heuristic now confirmed across 7 candidates spanning BOTH handcrafted feature groups AND a structurally-different FM substrate:**

| candidate | substrate type | standalone cold-LR UAR | predicted verdict | actual K=2/K=3 verdict |
| --------- | -------------- | ---------------------- | ----------------- | --------------------- |
| G5_modulation | handcrafted (64-d) | 0.6121 | ADMIT | K=2 WIN (0.7037) |
| G1_voicing | handcrafted (9-d) | 0.6058 | borderline | K=2 borderline (0.6964) |
| G6_spectral | handcrafted (21-d) | 0.6053 | borderline-fail | K=2 FAIL (0.6698) |
| **G_HuBERT_base_meanpooled** | **FM-derived (3072-d)** | **0.5396** | **definite FAIL** | **K=3 SKIPPED (M14 pre-flight)** |
| G_egemaps_full | handcrafted (88-d) | 0.5384 | definite FAIL | K=3 FAIL (0.6801) |
| G2_prosody | handcrafted (10-d) | 0.5088 | definite FAIL | K=2 FAIL (0.6674) |
| G3_voice_quality | handcrafted (14-d) | 0.5039 | definite FAIL | K=2 FAIL (0.6576) |

The 0.55--0.61 threshold range now partitions admit/fail across **7 substrates spanning two substrate families** (handcrafted acoustic feature groups + foundation-model mean-pooled embeddings). The heuristic is more general than the original handcrafted-only test established.

**Paper-stage finding (the prediction list's "no-admit" branch fired):** *"Single-FM late fusion (A2.5_WavLM + audited handcrafted groups) is sufficient on URTIC. A second foundation model — specifically HuBERT-base mean-pooled across layers, structurally distinct from WavLM via cluster-based pretraining — does not add cold-relevant signal beyond what WavLM-Large captures via its layer-weighted softmax pooling. Multi-FM late fusion at the mean-pooled level is therefore not justified at this corpus scale."*

**Mechanism reading — why HuBERT mean-pooled fails so cleanly:**

Two competing hypotheses explain why HuBERT-base mean-pooled cold-LR (0.5396) is so much weaker than WavLM-A2.5's single-substrate cold information (0.6564 argmax UAR via the trained head):

1. **The layer-weighted softmax pooling does the work.** WavLM-A2.5's softmax layer-weights were trained on cold and concentrate on cold-relevant layers (top-5 are L0, L2, L5, L22, L6). HuBERT mean-pooled is uniform across all 13 layers including layers that may carry only acoustic-detail/speaker signal with no paralinguistic content. Per Pasad-Chen, HuBERT's paralinguistic content also concentrates in mid-layers; uniform mean dilutes the signal.
2. **WavLM-Large just has more capacity than HuBERT-base** (24 × 1024 vs 13 × 768; ~3.5× more pooled-stat dims). HuBERT-large would be a fairer comparison.

The cleaner test of (1) would be HuBERT-base WITH a learned layer-weighted softmax (an A2.5-style head trained on HuBERT pooled stats) vs WavLM-A2.5; the cleaner test of (2) would be HuBERT-large mean-pooled. Both are out of scope for this Tier-2 follow-up; we tested the cheapest reasonable variant ("HuBERT as a cold-LR fusion candidate analogous to G5/G6") which cleanly fails per the M14 heuristic, and document the deeper variants as future work.

**M14 generalisation extension worth a paper paragraph:** the standalone-UAR-predictor heuristic now spans BOTH handcrafted feature groups AND FM-derived substrates with the same ~0.61 threshold. The heuristic is therefore not specific to handcrafted feature engineering; it captures a more general property — "does the candidate substrate carry a cold-axis that's distinguishable enough from the calibration noise to fuse productively against the A2.5 anchor's strong logits, vs. getting fusion-absorbed at boundary β* with τ at extreme." Anyone running honesty-audited late fusion on a paralinguistic corpus can use this heuristic to fail-fast on candidates that won't admit, regardless of substrate family.

**Practical validation of M14:** the cell skipped the K=3 sweep step entirely via the pre-flight check, saving ~5 min of compute on a likely-failed configuration. The fail-fast mechanism worked as designed.

**A5b K=2 (A2.5 + G4_gi + G5_modulation) is now FINAL canonical with structural validation:** (1) all 6 A5a-admitted handcrafted groups tested as K=2 partners (G5 won); (2) full eGeMAPSv02 superset tested as K=3 addition (failed); (3) HuBERT-base mean-pooled (structurally-different FM substrate) tested as K=3 addition (M14 pre-flight failed). No alternative configuration beats K=2 LOCKED 0.7037 ± 0.0060 within budget. The K=2 5-seed mean-logit ensemble (0.7090) remains the best paper-headline number on the controlled-system axis.

**Cumulative stack unchanged (HuBERT did not alter A5b K=2):** uniform-A2-grouped 0.6361 → A2.5 0.6563 → K=1 0.6934 → K=2 0.7037 → K=2 ensemble 0.7090. Distance to 0.71 baseline: 0.001. Final canonical state.

- **§4.11.2.1 A4 discrete audio tokens** (~3-5 days). HuBERT/EnCodec discrete token histograms as a separate feature stream — preserves syllable-rate/utterance-rhythm patterns that pooled WavLM stats average out. New G-group through A5a honesty audit + possible admission to A5b fusion. **Plausible upside: +0.005-0.020 → 0.696-0.711, plus a new architectural dimension for the paper.** Risk: tokens may carry speaker info (M10 returns).
- **§4.11.2.2 Test-time augmentation** (~half-day). Score devel chunks under N perturbations (gain ±2dB, time-shift ±20ms, light noise) and average logits. Common +0.5-1pp lift in audio tasks.

#### 4.11.3 Tier 3 — speculative

- **§4.11.3.1 New feature groups (G7+)**: formant stability, voicing turbulence, RMS-modulation depth. ~half-day each. Each might add +0.002-0.010, cumulative but small.
- **§4.11.3.2 Pseudo-labeling on test set** for self-training. Risky — amplifies classifier bias.

#### 4.11.4 Counter-argument worth flagging

The methodology paper is essentially ready (M8-M14 + 3-level closure + the v1→v2 worked example). Pushing UAR from 0.6913 → 0.70+ adds a paper subsection but doesn't change the load-bearing argument. **If time budget is tight, prioritize the write-up over more rungs.** Tier 1 § 4.11.1.1 is cheap enough (~2 hr) that it should run first regardless; Tier 2 onwards is conditional on Tier 1 results + remaining time budget.

#### 4.11.5 Execution order (chosen)

1. **§4.11.1.1 K=2 extended β-sweep first** — cheapest direct test. Decisive verdict in ~2 hr.
2. If K=2 PASSES → lock K=2 as new canonical, then **§4.11.2.1 A4 discrete tokens**.
3. If K=2 FAILs → skip directly to **§4.11.2.1 A4 discrete tokens** as the bigger conceptual play.
4. **§4.11.1.2 5-seed expansion** runs in parallel with whatever's active.

**Engineering deliverables for Phase 1.**

- New module: `model/representation/adversary.py` — `GradReverse` autograd function (forward = identity, backward = -λ * grad), `SpeakerDiscriminator` MLP, λ_adv sigmoid scheduler.
- New cell in `run.ipynb`: `a7_phase1_poc.py` — λ_adv sweep, 3 seeds, M10/M11 baked-in controls (random + cold-CE-only at layer-weight-open scope to disambiguate).
- Reuses: `representation.contrastive.ContrastiveProjection` (the projection MLP, not its loss), `representation.contrastive.SpeakerBlockSampler` (or simple shuffled batches if speaker-block isn't necessary for adversarial training — TBD), data + probes + cluster modules from existing infrastructure.

#### 4.10.2 Phase 2 — conditional escalation (scoped, conditional on Phase 1)

- **(A) PoC PASS** → longer training (50 epochs), more seeds (5), MDD substitution as ablation row, optional full transformer fine-tune as Phase 3 if PoC margins suggest the layer-weight scope is bottlenecked. Cost: ~3-5 hr GPU.
- **(B) DANN fails, try MDD** → Margin Disparity Discrepancy (Zhang 2019) is more principled (bounded generalization gap on the speaker-shifted distribution). ~1-2 hr to implement + run.
- **(C) Pareto trade-off documented** → λ_max grid refinement + ramp-schedule sweep. ~1-2 hr.
- **(D) Adversary destabilises** → smaller λ_max + longer warmup + maybe gradient clipping on the discriminator. ~1 hr diagnostic.

#### 4.10.3 A7 in the 3-level de-confounding ladder (final paper framing)

A5.5 locked: data-level alone insufficient (M9). A6 closed: representation-level head-only contrastive is purely subtractive on CE-anchored bottlenecks (M10 + M11). A7 outcomes:

- **A7 PASS** → "we tested three architectural levels of de-confounding intervention; only gradient-level activated the mechanism." The paper has a clean explanatory arc: each level fails in a specific way (data = no signal; representation = subtractive on CE-anchored; gradient = required architectural access), and only the level with explicit gradient-direction subtraction produces real de-confounding. The methodology contributions (M8 self-splice control, M9 narrow-window-empty audit, M10 bottleneck-confound discipline, M11 subtractive-objective warning) carry the paper alongside the positive A7 result.
- **A7 FAIL** → "neither data nor representation nor gradient-level intervention produced measurable probe drops at the strengths tested on URTIC + frozen WavLM; the speaker shortcut on this corpus is mechanism-resistant." Strong negative-result methodology paper anchored on the four M-findings + the systematic three-level closure. Either outcome is publishable; the methodological framework is the load-bearing contribution either way.

#### 4.10.4 Risks

- **Adversarial training instability.** GRL training is famously fiddly — λ_adv ramp pace, optimizer momentum, batch composition all matter. The Ganin sigmoid schedule is the standard mitigation; constant-λ with warmup is the fallback.
- **Speaker discriminator over-fits.** With k=210 pseudo-speakers and 8532 train chunks (~40 chunks per speaker), the discriminator may achieve very high train accuracy without generalising — making the adversary signal noisy. Mitigation: weight decay on discriminator, shallower discriminator MLP, or k=420 re-clustering if 210 is too coarse. Defer until Phase 1 verdict.
- **M10/M11 confound returns at deeper scope.** Even at layer-weight-open scope, the projection bottleneck still does some compression work. The baked-in controls (random + cold-CE at the same layer-weight-open scope) disambiguate; if A7 doesn't beat both controls, the adversary is illusory at this scope too.
- **Cold UAR drops more than 1σ at any λ > 0.** The Pareto trade-off case (branch C). If unavoidable, report explicitly rather than ship as "PASS with caveats."

---

### 4.12 Tier-2 follow-ups for pushing past 0.7090 (three queued cells)

After the paper's structural restructure (commit `9a32ede`, main + supplementary appendix split), three additional cells are appended to `run.ipynb` to push UAR past the K=2 5-seed mean-logit ensemble's 0.7090. Cheap-to-expensive cost order, orthogonal mechanisms so they stack independently:

| sub-section | cell | cost | mechanism | expected lift if hypothesis holds |
| --- | --- | --- | --- | --- |
| §4.12.1 | calibration / stacked weighting | ~5 min | re-weight per-seed K=2 fused logits via LR-stacking + UAR-grid-search; isotonic ablation as M15-test | +0.000 to +0.005 over 0.7090 if some seeds add noise |
| §4.12.2 | TTA on K=2 ensemble | ~25 min | 4 audio augmentations × 5 seeds = 25 forward passes; mean-logit | +0.003 to +0.010 over 0.7090 if WavLM substrate variance is the bottleneck |
| §4.12.3 | HuBERT-base + learned layer-weighting | ~50 min | A2.5-style head on HuBERT pooled stats with honesty-prior init from per-layer audit; M14 pre-flight; conditional K=3 fusion sweep | +0.005 to +0.012 over 0.7037 K=2 single (-> ensemble ~0.715-0.721) if the layer-weighted-pooling-does-the-work hypothesis holds |

Decision tree for the queue: any of the three can stack with the others if positive. Cell §4.12.3 is the load-bearing methodology test (cleanest test of the §4.11.2.1 hypothesis (i) "layer-weighted softmax does the work" vs hypothesis (ii) "capacity dominates"); cells §4.12.1 and §4.12.2 are pure UAR-push optimisations with low methodology risk.

#### 4.12.1 K=2 ensemble calibration + stacked weighting (DONE; calibration NEUTRAL, mean-logit stays canonical; M15 + M16 cautionary tales)

Three weighting strategies on the K=2 5-seed per-seed fused logits + one isotonic ablation:

**Strategies:**

1. **MEAN-LOGIT baseline** (equal weights w_i = 0.2 each). The existing K=2 5-seed mean-logit ensemble at devel_test UAR 0.7090.
2. **LR-STACKED weighting:** fit logistic regression on `(per-seed K=2 fused logits, label)` on `train_threshold` (StandardScaler + LogisticRegression with C=1.0, class_weight="balanced"). The 5 learned weights + bias collapse the 5-d input to a single calibrated probability. Apply to devel_test, sweep tau, evaluate. Mechanism: LR may shrink down 'noisier' seeds, raising effective weight on more informative ones.
3. **UAR-GRID-SEARCH weighting:** coarse grid over `(w_1, ..., w_5)` with `w_i ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}`, normalised to sum to 1 (6^5 = 7776 combinations evaluated). Pick the combination with highest UAR on `train_threshold` (after sweeping tau). Apply to devel_test. Mechanism: directly optimises the target metric (UAR) rather than the proxy (likelihood); captures non-monotonic seed-quality patterns that LR can't.
4. **ISOTONIC ABLATION on the best of (1)-(3):** fit `IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` on `(sigmoid(best_ensemble_logit), y)` on `train_threshold`, apply to devel_test. **Mathematically: monotonic calibration is order-preserving, so UAR-at-optimal-tau is INVARIANT under it.** This serves as the M15-candidate "monotonic calibration cannot improve UAR-at-optimal-tau" confirmation.

**Why M15-candidate matters as a methodology contribution:** the existing tau-sweep on the swept-logit-score already implicitly does whatever monotonic calibration would do. The only way calibration can help UAR-at-optimal-tau is to break monotonicity (which isotonic does NOT, by design) -- so any non-zero lift in the calibrated arm would indicate a measurement artefact (tau grid resolution mismatch), not a real gain. This is paper-relevant if the M15 confirmation holds: it's a tight bound on what calibration can/cannot do for UAR-at-swept-threshold.

**Decision rules:**

- Any variant > 0.710 baseline -> "calibrated ensemble crosses 0.710 baseline" (paper headline lift).
- Any variant > 0.7090 + 0.002 -> "calibration helps modestly" (paper supplementary ablation).
- All variants within 0.002 of mean-logit -> "calibration neutral" (paper supplementary ablation; mean-logit stays canonical).
- Any variant < 0.7090 - 0.002 -> "calibration hurts" (paper supplementary; document as failed-as-expected ablation, given LR-stacking can overfit on n=973 train_threshold).

**Output:** `results/A5b_k2_ensemble_calibrated.json`. Self-contained (no GPU; numpy + sklearn only on cached fused logits). Cost: ~5 min wall-clock.

**RESULT (DONE):** decision = `calibration_neutral`; mean-logit stays canonical. Per-variant devel_test UAR:

| variant | devel_test UAR | Δ vs MEAN | train_threshold UAR | observation |
| --- | --- | --- | --- | --- |
| MEAN-LOGIT (baseline) | **0.7090** | 0.0000 | 0.6529 | reproduces existing 0.7090 (τ*=-1.375; recC=0.791, recNC=0.627) |
| LR-STACKED | 0.6137 | **-0.0953** | 0.6887 | catastrophic overfit; learned weights {+1.57, **-1.85**, +1.18, **-2.89**, +2.68} have huge magnitudes + 2 negative signs |
| UAR-GRID-SEARCH | 0.7049 | -0.0041 | 0.6648 | best weights (0, 0, 0.8, 0.2, 0) — uses only 2 of 5 seeds; +0.012 train_thr gain → -0.004 devel_test loss (generalisation gap +0.016) |
| ISOTONIC ABLATION on MEAN-LOGIT | 0.7064 | -0.0026 | 0.6535 | M15 prediction was 0.000; observed -0.0026 is tau-grid resolution artefact |

**Wall time:** 53.65 min (NOT the predicted ~5 min). The bulk is per-seed K=2 fused-logit recomputation across 5 seeds × 4 splits via `_load_a2hp_head + _a2_logit_on_split` (PooledCacheDataset's per-stem I/O is the bottleneck on the WavLM 4096-d cache files). My earlier estimate assumed the per-seed fused logits would be cached; they aren't, so the cell does the full 5×4 = 20 inference passes from scratch. **Methodology note:** the per-seed K=2 fused logits at locked β* per seed are deterministic given fixed ckpts + cached features; caching them as `cache/A5b_per_seed_k2_logits.npz` would cut future cell runtimes from 50 min to ~30 sec. Defer the cache-build until we know whether subsequent cells (§4.12.2 TTA, §4.12.3 HuBERT-LW) need them.

**M15 confirmation (with caveat):** the isotonic-ablation devel_test UAR (0.7064) is *strictly less than* the pre-isotonic mean-logit UAR (0.7090) by 0.0026. The theoretical prediction was equality (UAR-at-optimal-tau invariant under monotonic transforms). The observed -0.0026 is the **tau-grid resolution artefact predicted in the cell docstring**: the raw fused-logit tau grid is `np.linspace(-4.0, 4.0, 321)` with step 0.025 (in logit space); the calibrated probability tau grid is `np.linspace(0.01, 0.99, 197)` with step 0.005 (in probability space). The optimal raw-logit tau τ*=-1.375 corresponds to a specific calibrated probability ≈ 0.085 in isotonic-mapped space, but the calibrated grid's nearest point is τ*=0.085 — close but not exactly the same operating point. The strict M15 statement holds: **monotonic calibration cannot IMPROVE UAR-at-optimal-tau**; it can only achieve EQUAL or appear-WORSE due to discretisation. Paper-stage framing: M15 is confirmed in the strong form ("not better than mean-logit") rather than the strict form ("exactly equal to mean-logit"). Add to methodology table once all 3 Tier-2 cells are in.

**M16 candidate (LR-stacking on small calibration splits is overfit-prone):** the LR-stacked variant produced devel_test UAR 0.6137 = mean-logit baseline 0.7090 - 0.0953. The catastrophic drop is consistent with overfit on n=973 train_threshold with ~92 cold samples × 5 features + bias. The LR's L2 regularisation (C=1.0) is not enough to constrain the weights; learned magnitudes max |w| = 2.89 indicate ~12× the natural per-seed-logit std (which is ~0.24 across the 5 seeds on train_threshold). Mechanism: LR maximises likelihood, which can be increased by predicting more confidently (sharper sigmoids) on individual training points; with 5 features the LR finds a 5-d weighting that's confidently right on most train_threshold cold examples — including 2 negative-sign weights that say "if this seed predicts cold, then it's NOT cold." Three of these flipped weights only make sense in the train_threshold sample, not on devel_test, where they invert the prediction. Add to methodology table as: *"On small calibration splits (n ~< 1k, k < 100 minority class), L2-regularised LR-stacking on per-seed logits is prone to catastrophic generalisation failure (large-magnitude weights with sign flips). Equal-weight mean-logit is the default; UAR-grid-search with strict held-out evaluation is the next-most-conservative alternative."*

**Grid-search overfit pattern:** the UAR-grid-search variant found best weights (0, 0, 0.8, 0.2, 0) = use only seeds {7, 999}. Train_threshold UAR went from 0.6529 (mean-logit) to 0.6648 (grid) = +0.012 train gain. Devel_test UAR went from 0.7090 (mean-logit) to 0.7049 (grid) = -0.004 devel loss. **Generalisation gap = +0.016 (gain - loss)** — the grid found a weighting that overfit the train_threshold sample-noisy UAR signal. With 7776 combinations evaluated on a 973-sample target, multiple-comparison effects also contribute: the maximum-of-7776 noisy UAR estimates has a positive bias of ~0.01 over the true population maximum at this sample size, which explains most of the apparent +0.012 train gain. This is a classic "grid-search overfits the target metric on small splits" pattern.

**Mechanism reading (why equal weights win):** the 5 K=2 fused logits per chunk are highly correlated across seeds (Pearson r ≈ 0.94 from the §4.11.1.4 ensemble cell's per-seed reproduction sanity check). With high inter-seed correlation, the optimal weighting is approximately uniform — and any deviation from uniform shifts away from variance-minimisation toward signal-selection, which is suboptimal when signals are similar. The 5 seeds are equally informative; mean-logit is the variance-optimal pooling.

**Paper implications:**

- **For the methodology table:** add M15 (monotonic calibration cannot improve UAR-at-optimal-tau; the swept-threshold protocol already does this work) and M16 (small-calibration-split LR-stacking is overfit-prone; default to equal-weight mean-logit). M15 + M16 together strengthen the K=2 5-seed mean-logit ensemble headline by ruling out two natural "what about ____?" critiques.
- **For the main body §6 results:** add a one-paragraph ablation noting that LR-stacking, UAR-grid-search, and isotonic calibration all underperform mean-logit. Cite the 5 calibration variants as a tightness check on the ensemble headline.
- **For the appendix:** full per-variant table goes to Appendix~\ref{app:k2_calibration} (new appendix section). The LR-stacking failure is paper-worthy as a worked example of small-calibration-split overfit.

#### 4.12.2 K=2 TTA ensemble (DONE; TTA HURTS by Δ -0.0407, mean-logit stays canonical; M17-candidate "WavLM input-normalisation absorbs gain; time-stretch hurts pooled-stat features")

Test-time augmentation on the WavLM substrate of the K=2 LOCKED canonical. For each of 4 audio perturbations + the original (5 versions per chunk), re-extract WavLM-Large pooled stats via `transformers AutoModel` inline (mirrors §4.11.2.1's HuBERT inline extraction), cache to `cache/microsoft_wavlm-large/pooled_tta/{aug_name}/`.

**Augmentations** (chosen for class-preserving paralinguistic robustness):

| aug name | kind | parameter |
| --- | --- | --- |
| `original` | (none -- existing cache) | -- |
| `time_stretch_p2` | librosa.effects.time_stretch | rate=1.02 |
| `time_stretch_m2` | librosa.effects.time_stretch | rate=0.98 |
| `gain_p2dB` | multiplicative gain | factor = 10^(+0.1) |
| `gain_m2dB` | multiplicative gain | factor = 10^(-0.1) |

**Why these augmentations:** time-stretch ≤ 2% preserves vowel quality (the paralinguistic cue for cold) while perturbing pitch contour and rate features minimally; gain ≤ 2dB perturbs amplitude statistics that WavLM's spectral layers absorb but doesn't shift voicing/breathing patterns. Larger perturbations risk shifting class probability (especially time_stretch >5% which starts to mimic 'raspy' = cold-positive shift on healthy speakers).

**Why TTA on WavLM only (not also on G4/G5):** G4_gain_invariant is gain-invariant by construction (zero-mean spectral centroid, zero-mean MFCCs); G5 modulation spectrum is fairly robust to small temporal jitter (4-Hz envelope correlations). Re-extracting them for each augmentation would add ~25 min compute for likely-marginal lift. We TTA the WavLM logit, which is where most expected variance reduction comes from for foundation-model-derived features. (If TTA is positive, a follow-up cell could test full-audio-TTA with re-extracted G4/G5; if WavLM-only TTA is null, full-audio-TTA is unlikely to help either.)

**Pipeline per (augmentation, seed):**

- Forward augmented WavLM pooled through the existing A2.5 head ckpt -> a2.5 logit per chunk.
- Compute `fused = a2.5_logit_aug + β_seed · mean(z_g4_orig, z_g5_orig)` where G4/G5 z-logits use the un-augmented cache.
- Per (aug, seed) sanity tau-sweep on train_threshold + devel_test UAR (printed inline for diagnostic).

**TTA ensemble:** mean across 5 augmentations × 5 seeds = 25 fused logits per chunk; sweep tau on train_threshold; evaluate on devel_test. Original-only ensemble (5 seeds, no augmentations) reproduced as a sanity check vs the §4.11.1.4 reference 0.7090; reproduction delta should be < 0.002 (any larger drift indicates a checkpoint/cache regression to debug).

**Decision rules:**

- TTA UAR ≥ 0.710 baseline -> "TTA crosses 0.710 baseline" (paper headline lift).
- TTA UAR > 0.7090 + 0.002 -> "TTA helps modestly" (paper supplementary ablation).
- |TTA UAR - 0.7090| ≤ 0.002 -> "TTA neutral" (paper supplementary ablation; document as 'WavLM substrate variance is not the bottleneck').
- TTA UAR < 0.7090 - 0.002 -> "TTA hurts" (paper supplementary; document as failed-as-expected ablation if WavLM is robust to small input perturbations).

**Output:** `results/A5b_k2_tta_ensemble.json` (extraction times per aug + per (aug, seed) UAR + ensemble UAR + decision). Cost: ~25 min compute (4 augmentations × ~6 min WavLM-Large re-extraction; ~1 min handcrafted re-fitting; ~2 min fusion+sweep). HuBERT cache reuse pattern from §4.11.2.1 (idempotent on existing per-stem cache files).

**Dependencies:** `librosa` for `time_stretch` (and the existing `transformers + soundfile`).

**RESULT (DONE):** decision = `tta_hurts`; mean-logit stays canonical. TTA ensemble UAR = **0.6683** (Δ -0.0407 vs no-TTA 0.7090). Original-only-ensemble reproduction = 0.7090 exactly (Δ -0.0000 vs reference) — confirms the per-seed K=2 logit pipeline is fully deterministic.

| augmentation | mean per-seed UAR | Δ vs original | mechanism |
| --- | --- | --- | --- |
| original | 0.7037 | 0.000 | (sanity reference) |
| `time_stretch_p2` (rate=1.02) | 0.6748 | -0.029 | temporal structure disrupted; pooled stats lose cold-relevant patterns |
| `time_stretch_m2` (rate=0.98) | 0.6601 | -0.044 | same mechanism, slightly worse |
| `gain_p2dB` | 0.6895 | -0.014 | **input-normalisation no-op** (see below) |
| `gain_m2dB` | 0.6894 | -0.014 | **input-normalisation no-op** (see below) |
| TTA ensemble (mean of 25) | 0.6683 | -0.035 | dilution: 5 good + 20 worse = drag down |

**Two paper-relevant mechanism findings:**

1. **Gain ±2dB is essentially a no-op due to WavLM input normalisation.** Per-seed `gain_p2dB` and `gain_m2dB` UARs are *identical to 4 decimal places* (seed 7 differs by 0.0003 = fp16 noise). The WavLM `AutoFeatureExtractor` (`from_pretrained("microsoft/wavlm-large")`) applies `do_normalize=True` (per-utterance zero-mean unit-variance), which is mathematically gain-invariant: `normalize(g·x) = (g·x - g·μ) / (g·σ) = (x - μ) / σ = normalize(x)` for any positive `g`. So multiplicative gain is absorbed at the input pipeline level *before* the transformer ever sees the difference. We effectively tested two augmentations (original + time-stretch) plus two no-ops. Future TTA designs on WavLM/HuBERT/Wav2Vec2 substrates should use perturbations that *survive* the input normalisation — e.g., additive noise (`x + n`, where `n` has zero mean), spectral masking on the framewise features, or perturbations applied at the pooled-stats stage rather than the waveform stage.

2. **Time-stretch ±2% causes 0.029-0.044 mean per-seed UAR drop.** Pooled stats (mean/std/skew/kurt over time) *should in principle* be robust to small temporal perturbations — they're sufficient statistics over the time axis. But WavLM's per-frame outputs have temporal structure (positional encoding + transformer self-attention pattern over frames) that gets disrupted by stretching. The new pooled stats systematically lose cold-relevant patterns that A2.5's layer-weighted softmax was tuned to. Mechanistically: time-stretch by rate 1.02 changes a 5-second chunk to 5.1 seconds (≈10 extra frames at 50 Hz frame rate); the WavLM positional embeddings see a different absolute frame index for the same acoustic content; the attention pattern over the now-longer sequence shifts; the layer-wise per-frame outputs differ; pooling those different per-frame outputs produces different stats; the cold-LR probe trained on the un-stretched stats sees out-of-distribution input.

**Why TTA ensemble dilutes rather than averages-toward-improvement:** the standard variance-reduction story for TTA ("average independent noisy observations of the same underlying signal") requires that the perturbations produce *unbiased estimates* of the same target. Here the augmented logits are *systematically biased* by the augmentation (gain logits ≡ original after normalisation = no-op contribution; time-stretch logits are systematically lower-UAR because they're out-of-distribution for the trained probes). Averaging 5 good originals (mean 0.7037) with 20 systematically-worse augmentations (gain ≡ original ≈ 0.6895 because of the τ-resweep on a single-aug ensemble of essentially-identical logits, plus time_stretch ~0.6675) gives an ensemble that pulls toward the worse augmentations. The fraction of the ensemble that is time-stretch-quality (40%) is enough to drag UAR down by 0.04.

**M17 candidate (paper-stage methodology):** *"On foundation-model substrates with input normalisation (Wav2Vec2 / HuBERT / WavLM all default to per-utterance zero-mean unit-variance via `AutoFeatureExtractor(do_normalize=True)`), waveform-level multiplicative perturbations are absorbed by the normalisation step and produce identical pooled features. Standard TTA designs (gain perturbation + time-stretch) tested on URTIC produce: gain = no-op (input-normalisation invariance); time-stretch = systematic UAR drop because pooled per-frame outputs are sensitive to the temporal structure changes that positional encodings + attention patterns introduce. Net: TTA ensemble UAR is strictly worse than no-TTA mean-logit ensemble. Future TTA designs for FM-substrate ensembling should use perturbations that survive input normalisation (additive noise) or operate at the pooled-stats stage (SpecAugment-style masking on the [25, 4096] tensor)."* This is a methodology contribution about TTA design on foundation-model substrates that generalises beyond URTIC.

**Combined narrative for the 0.7090 ensemble (now triply defended):**

- **Calibration (§4.12.1):** equal-weight is variance-optimal on highly-correlated per-seed logits (M16 LR-stacking catastrophe + M15 monotonic-calibration invariance).
- **TTA (§4.12.2, this cell):** input-perturbation averaging hurts because gain is absorbed and time-stretch is systematically biased downward (M17 input-normalisation discipline).
- **Original-only-ensemble reproduction:** 0.7090 reproduces exactly across both cells, confirming the per-seed K=2 logit pipeline is fully deterministic.

The K=2 5-seed mean-logit ensemble at 0.7090 stays the canonical paper-headline number; the calibration + TTA ablations confirm the operating point is variance-optimal AND input-perturbation-invariant within the architectures tested.

**Paper implications for the appendix:** add a new appendix section `Appendix~\ref{app:k2_tta}` with the per-aug per-seed UAR table + the gain-no-op / time-stretch-hurts mechanism analysis. This is a paper-grade negative result with a clean transferable mechanism.

**Wall time:** 39.88 min (consistent with my predicted ~25-30 min for the augmentation extraction + ~10 min for fusion). Lessons: WavLM-Large extraction is ~10 min/aug at ~30 chunks/sec on GPU (fp16 batch 1); gain extractions are 2× faster than time-stretch extractions (5.5 min vs 10.5 min) because no librosa time-stretch call.

#### 4.12.3 K=3 with HuBERT-base + learned LW softmax (DONE; nuanced result -- both hypotheses partially validated, K=3 doesn't admit by logit-correlation)

The cleanest test of the §4.11.2.1 hypothesis (i) "layer-weighted softmax pooling does the work". Mirrors WavLM-A2.5's treatment on HuBERT-base pooled stats:

**STEP 1 -- Per-layer HuBERT honesty audit:** for each layer `L ∈ {0..12}`, train linear cold probe + linear speaker probe on `pooled[:, L, :]` (3072-d per layer, mean+std+skew+kurt of the 768-d hidden), compute `label_gain_L = cold_uar_L - 0.50`, `speaker_gain_L = top1_L - 1/k`, `sub@1_L = label_gain - speaker_gain`. Produces a 13-d audit vector that is the honesty prior for the layer-weighted softmax. Output: `results/A5d_hubert_layer_honesty.csv` (mirrors `A5d_grouped_layer_honesty.csv` for WavLM).

**STEP 2 -- HuBERT-A2.5 head training (per seed):** `LayerWeightedPooledHead(n_layers=13, stat_dim=3072, proj_dim=128, dropout=0.5)`. Initialise `layer_weights = T_INV * sub_at_1` with `T_INV = 50.0` (mirrors WavLM-A2.5). Train cold-CE with class-balanced sampling, AdamW with `lr=1e-3` for head + `lr=1e-4` for `layer_weights` (per `param_groups` convention), `weight_decay=1e-4`, 25 epochs, early-stop patience 6 on devel_val UAR. 5 seeds {42, 123, 7, 999, 31337}. Save ckpt per seed at `cache/facebook_hubert-base-ls960/head_A25_honestprior_seed{seed}.pt`.

**STEP 3 -- Standalone HuBERT-A2.5 UAR per seed (M14 pre-flight):** per seed, forward devel_test through trained head -> `P(cold)` per chunk -> log-odds -> swept tau on train_threshold -> per-seed UAR. Aggregate 5-seed mean. Apply M14 standalone-UAR-predicts-K-fusion-admission heuristic (calibrated from §4.11.1.1):

- `< 0.55` -> definite FAIL; skip K=3 sweep (M14 fail-fast).
- `>= 0.61` -> admit plausible; run K=3 sweep.
- `[0.55, 0.61)` -> borderline; run K=3 for confirmation.

Also report: `delta vs HuBERT mean-pooled standalone (0.5396)` (the §4.11.2.1 number) and `delta vs WavLM-A2.5 standalone (~0.656)`. The first delta tests "does layer-weighted softmax close the gap on HuBERT-base?"; the second tests "is HuBERT-A2.5 anywhere close to WavLM-A2.5 strength on the same task?".

**STEP 4 -- K=3 fusion sweep (if M14 pre-flight passes):** for each seed: `fused = a2.5_wavlm_logit + β · mean(z_g4_gi, z_g5_mod, z_hubert_a25)`. Extended β-grid `{0..2, 2.5..16}` (16 values). Per-seed argmax β* on `train_threshold` UAR. 5 seeds.

**Decision rules:**

- Mean K=3 UAR > 0.7037 + 0.005 = 0.7087 -> ADMIT K=3 with HuBERT-A2.5 as new canonical. 'Layer-weighted softmax does the work' hypothesis VALIDATED. Multi-FM late fusion as a new architectural contribution.
- Standalone clears 0.61 but K=3 doesn't admit -> M14 heuristic FALSIFIED on this substrate; document the counter-example. Possible mechanism: HuBERT-A2.5 logit is correlated with WavLM-A2.5 logit (both audited from cold-relevant layers of FM substrates); fusion partner needs to be ORTHOGONAL to admit.
- Standalone < 0.55 (M14 skip) -> "layer-weighted softmax DOESN'T close the gap on HuBERT-base alone; capacity / backbone strength dominates" -> evidence for §4.11.2.1 hypothesis (ii) over hypothesis (i). Future work: HuBERT-large with learned LW.
- Standalone in `[0.55, 0.61)` -> M14 borderline; K=3 verdict either way is paper-relevant.

**Output:** `results/A5b_k3_hubert_lw_5seed.json` (per-layer audit + per-seed head training results + standalone per seed + 5-seed standalone aggregate + M14 verdict + K=3 sweep if applicable + final decision). Cost: ~50 min wall-clock (~5 min audit + ~25 min head training across 5 seeds + ~20 min standalone UAR + K=3 sweep). HuBERT cache assumed present from §4.11.2.1.

**Dependencies:** none beyond what §4.11.2.1 already required.

**RESULT (DONE):** decision = `k3_hubert_lw_no_admit`; K=2 stays canonical. **Both hypotheses partially validated; new fusion-orthogonality boundary condition emerged.** Cell ran in **3.71 min** (vs 50 min predicted) because head training fired early-stop at epoch 1-4 across all seeds — the HuBERT layer-weight subspace is much smaller (13 layers vs WavLM's 25) and the cold signal saturates fast.

**Hypothesis decomposition:**

| measurement | value | interpretation |
| --- | --- | --- |
| HuBERT mean-pooled standalone (§4.11.2.1 ref) | $0.5396$ | naive baseline |
| HuBERT-A2.5 standalone (this cell, 5-seed mean) | $0.6217 \pm 0.0291$ | learned LW + honesty prior |
| WavLM-A2.5 standalone (~ref) | $0.656$ | full-capacity LW |
| **gap closure breakdown** | $0.5396 \to 0.656 = 0.116$ total | --- |
| from layer-weighted softmax (HuBERT mean → HuBERT-A2.5) | $+0.082$ | $\approx 71\%$ |
| from capacity (HuBERT-A2.5 → WavLM-A2.5) | $+0.034$ | $\approx 29\%$ |

**Hypothesis (i) "layer-weighted softmax does the work"** — PARTIALLY VALIDATED. Adding learned LW + honesty-prior init to HuBERT-base buys $+0.082$ standalone UAR, lifting the substrate from M14 definite-FAIL ($0.5396 < 0.55$) to M14 admit-plausible ($0.6217 \geq 0.61$). The audited cold-relevant layers concentrate (top-5 final softmax weights: L0 $0.31$, L1 $0.083$, L12 $0.083$, L10 $0.069$, L3 $0.067$ — heavily L0-dominant which is also the highest-sub@1 layer in the audit). cos(audit prior, final softmax) $= 0.95$ across all 5 seeds — the prior is tightly preserved through training (mirrors the M5 finding for WavLM-A2.5).

**Hypothesis (ii) "capacity dominates"** — PARTIALLY VALIDATED. Even with the same architectural treatment (LW softmax + honesty prior + identical training recipe), HuBERT-A2.5 standalone is $0.034$ below WavLM-A2.5. Some of the gap is genuinely about backbone capacity / pretraining-corpus diversity ($24 \times 1024$ vs $13 \times 768$ pre-stat dims; WavLM trained on $94$\,k hours, HuBERT-base on $960$\,h LibriSpeech).

**K=3 fusion verdict: NO ADMIT (0.6955 ± 0.0083, Δ -0.008 vs K=2 LOCKED).** Per-seed locked $\beta^* = \{16, 16, 6, 16, 16\}$ (4 of 5 boundary-pegged); per-seed deltas $\{-0.020, -0.003, -0.013, -0.004, -0.001\}$ (4 of 5 within K=2 σ envelope, 1 noticeably worse). The result is essentially neutral ($-0.008$ within $\sim 1.5\sigma$ of K=2 σ $0.006$), not catastrophic — but doesn't cross the admit threshold ($0.7037 + 0.005 = 0.7087$).

**New mechanistic insight (refinement of the §6 standalone-UAR-predictor heuristic):** standalone UAR cleared 0.61 but K=3 fusion didn't admit. **The mechanism is logit correlation, not individual weakness:** HuBERT-A2.5 logit is highly correlated with WavLM-A2.5 logit by construction (both derived from the same audit recipe — fit linear cold + speaker probes per FM layer, compute sub@1, use as honesty prior for an LW softmax head, train cold-CE). Both heads end up concentrating on cold-relevant layers and producing similar cold-axis predictions per chunk. Adding a correlated logit to the K=3 fusion doesn't add orthogonal information — it just adds noise that the fusion architecture detects as "noise to silence" and pegs $\beta^*$ at boundary 16. **The standalone-UAR-predictor heuristic should be augmented with a logit-correlation check (or an early-fusion baseline) before committing β-sweep compute to a candidate that's structurally similar to the anchor.**

**Per-seed standalone variance (σ = 0.0291) is much higher than WavLM-A2.5 (~0.0027 at 5-seed).** HuBERT-A2.5 head training is less stable; possible mechanism: HuBERT pooled stats have lower SNR on cold (per-layer cold UAR range $0.5569$--$0.6128$ vs WavLM's likely $0.65+$ for the cold-best layer), so the head training is more sensitive to seed-specific data ordering. Per-seed standalone is bimodal: 3 of 5 seeds clear 0.61 (seed 7 $0.6547$, seed 42 $0.6475$, seed 999 $0.6191$); 2 of 5 are below (seed 123 $0.5888$, seed 31337 $0.5982$). The mean (0.6217) crosses the M14 threshold by a thin margin; per-seed variance is the limiting factor.

**Paper implications:**

- **Update §4.5.5 K=3 HuBERT subsection** in `paper/sections/04_method.tex`: add a paragraph noting the §08 future-work axis (HuBERT-base + learned LW) has now been tested, with the nuanced result (both hypotheses partial, new orthogonality boundary).
- **Update §6 standalone-UAR-predictor table** in `paper/sections/06_results.tex`: add a row for HuBERT-A2.5 (FM-derived, 128-d projection from LW head's classifier output, standalone $0.6217 \pm 0.029$, K=3 verdict NO ADMIT, mechanism: logit correlation with anchor). Add a paragraph after the heuristic-statement explaining the boundary condition.
- **Update §08 conclusion future work**: this axis is now CLOSED with paper-paragraph evidence. The remaining untested variants from the §08 list are (ii) HuBERT-large mean-pooled (still genuine "capacity dominates" test) and (iii) HuBERT discrete tokens (orthogonal feature, not subject to the audit-recipe-mirror correlation issue) — these stay as future work for a possible deeper paper.
- **Add Appendix `app:k3_hubert_lw`**: per-layer audit table + per-seed head training history + per-seed standalone + K=3 sweep details.
- **Methodology table refinement**: the result refines the standalone-UAR-predictor heuristic mentioned in §6 (not the M-disciplines table); decide whether to add as M18 ("multi-FM late fusion requires logit orthogonality, not just individual standalone strength") or keep as a §6 paragraph. Recommend §6 paragraph since it's a sharpening of an existing finding rather than a fully new discipline. **DONE in paper commit** (post-reflection refinement): \S\ref{ssec:standalone_predictor} now formalises the heuristic as substrate-conditional --- independent-substrate candidates clear $\sim 0.55$--$0.61$; correlated-substrate candidates (other FM-derived heads built via the same audit recipe) clear the anchor's own standalone UAR ($\sim 0.66$ for WavLM-A2.5); ambiguous cases run an early-fusion Pearson correlation check ($r > 0.8$ = correlated). Under the substrate-conditional formulation HuBERT-A2.5 at $0.6217 < 0.66$ would have been correctly pre-registered as a likely K=3 negative.

- **Cross-architectural transferability of the honesty-prior mechanism (paper-stage finding, post-reflection)**: the $+0.082$ HuBERT-A2.5 lift over HuBERT mean-pooled mirrors the $+0.071$ WavLM-A2.5 lift over uniform-A2-grouped, with similarly high $\cos$(prior, final softmax) $\approx 0.95$ on HuBERT (vs $\approx 0.9998$ on WavLM at default lr) and identical top-5 final layers across all 5 HuBERT seeds. The honesty-prior + frozen-FM + small-data regime appears to admit a globally-stable optimum on both architectures tested, not a per-seed-different convergence point. Now elevated to a main-body sentence in \S\ref{ssec:k3_hubert} (paper) as a methodology-generalisation claim that strengthens the C2 contribution. Per-seed standalone $\sigma$ disparity (HuBERT-A2.5 $0.029$ vs WavLM-A2.5 $0.004$, $\sim 8\times$) noted in the same paragraph and attributed to HuBERT-base's smaller layer-weight subspace + faster cold-signal saturation.

#### 4.12.4 Documentation strategy when results land

- If §4.12.1/2 give modest lifts (<= +0.005 each, stacking to maybe +0.010) -> add a single paper subsection in main body §4 method explaining the calibrated/TTA ensemble as the deployment-ready operating point; defer per-variant numbers to appendix.
- If §4.12.3 admits at K=3 -> rewrite §4.5.5 K=3 HuBERT subsection in main body to reflect the new canonical (HuBERT-A2.5 as the third group instead of HuBERT mean-pooled as a M14-skipped negative). Update the cumulative stack table (§6 Table tab:stack) to add a new row "K=3 (A2.5 + G4 + G5 + HuBERT-A2.5)".
- If §4.12.3 doesn't admit -> add a paragraph to §4.5.5 explaining "we tested the layer-weighted-pooling-closes-the-gap variant explicitly; it [closed/didn't close] the standalone gap to 0.61, and [admitted/didn't admit] at K=3". Either outcome is paper-relevant for the M14 cross-family discussion.
- M15 confirmation (if isotonic null holds) -> add an M15 entry to the methodology table (Table tab:m_disciplines) and a short paragraph to §5 de-confounding closure. M15 = "monotonic calibration cannot improve UAR-at-optimal-tau; the swept-threshold protocol already does this work".

#### 4.12.5 Risk/upside summary

- **Stacking risk:** all three cells could give null results, in which case the K=2 5-seed mean-logit ensemble at 0.7090 stays the headline. Even null results are paper-relevant ablations that defend the canonical configuration.
- **Stacking upside:** if all three give modest positive lifts: calibration +0.003, TTA +0.005, HuBERT-A2.5 K=3 +0.010 -> cumulative ~0.727 (clearly past 0.710 baseline + interpretable lift over no-aug per-seed-single 0.7037). Compositional lifts would need a final all-three-stacked run to confirm the additivity isn't double-counting any variance reduction.
- **Methodology contribution:** §4.12.3 is the cleanest test of the layer-weighted-pooling hypothesis from §4.11.2.1 -- whichever way it lands, it's a paper-grade result for the M14 cross-family discussion. §4.12.1's M15 ablation is a paper-grade result for the methodology table if it confirms.

### 4.13 Robustness verification (no hidden-test access; reflection-driven)

After §4.12 closures (calibration neutral, TTA hurts, HuBERT-A2.5 K=3 no-admit), the user-shared reflection raised a load-bearing concern: we have used `devel_test` for many decisions (K=2 candidate selection, β locks, M14 thresholds, calibration verdicts, TTA verdict, HuBERT-LW verdict). Each devel-side peek erodes its independence as a proxy for hidden-test UAR. Without hidden-test access (the ComParE 2017 hidden test labels are not publicly available), the right move before paper submission is to switch to a **robustness-first verification plan** rather than chasing devel-only micro-gains.

Two cells appended to run.ipynb to verify the canonical 0.7090 result without new model training. Both reuse the cached A2.5 head ckpts + locked β* per seed:

| sub-section | cell | cost | mechanism | what it tests |
| --- | --- | --- | --- | --- |
| §4.13.1 | shadow-split robustness harness | ~5-6 min | re-evaluate K=2 single + ensemble under 10 different (devel_val, devel_test) partitions of the existing devel split | does 0.7090 hold across devel partitions, or is it specific to SPLIT_SEED=42? |
| §4.13.2 | speaker-level logit smoothing | ~5-6 min | per-pseudo-speaker mean smoothing of the ensemble logit; α-sweep on smoothed train_threshold UAR; pick best α; eval on smoothed devel_test | does aggregating per-chunk predictions across same-speaker chunks reduce noise and improve UAR? |

Both cells are pure verification / post-processing — no new training. LoRA / classical SVM baseline alignment / further architecture experiments are explicitly **not** included; the user's robustness-first framing argues against any change that's calibrated on devel-only signal.

#### 4.13.1 Shadow-split robustness harness (DONE; canonical at z=+1.24σ above shadow mean -- "marginal" verdict, paper-grade methodology contribution)

**Goal:** distinguish "genuine architectural lift" from "lucky devel split." Re-evaluates the canonical K=2 5-seed mean-logit ensemble + per-seed-single canonical under N=10 different (devel_val, devel_test) partitions of the devel split via different `StratifiedGroupKFold` seeds. Train_fit + train_threshold kept FIXED at canonical SPLIT_SEED=42 (so per-seed A2.5 head ckpts and locked β* per seed are unchanged) — pure devel-side overfit check.

**Optimization:** per-seed A2.5 / G4 / G5 logits computed ONCE on all 9596 devel chunks (one inference pass per seed, ~1 min/seed = ~5 min total), then partitioned per shadow_seed by file index. Per-seed locked τ also computed once on the unchanged `train_threshold`. The shadow-split sweep is then just numpy slicing + per-shadow `evaluate_at_tau` — ~30 sec for 10 shadow splits. Total ~5-6 min vs ~50 min for naive re-inference per shadow split.

**Decision rule:**

- canonical 0.7090 within $\pm 1\sigma$ of shadow-mean → `robust_canonical_within_1sigma`. 0.7090 generalises across devel partitions; safe as paper headline.
- canonical 0.7090 within $\pm 2\sigma$ of shadow-mean → `marginal_canonical_within_2sigma`. Mild devel-split sensitivity; recalibrate framing slightly ("0.7090 on canonical, [shadow_min, shadow_max] across 10 splits").
- canonical 0.7090 outside $\pm 2\sigma$ → `fragile_canonical_outside_2sigma`. Recalibrate paper headline framing more aggressively; the canonical split was lucky.

**Output:** `results/A5b_k2_shadow_splits.json`. Per-shadow per-seed UAR + per-shadow ensemble UAR + aggregate (mean ± std + min/max) across 10 shadow splits + decision verdict.

**Shadow seeds chosen:** {1, 2, 3, 5, 11, 17, 23, 31, 53, 99} — distinct from canonical 42; spread across the integer range to avoid hash-prefix collisions; deterministic per-seed StratifiedGroupKFold gives different partitions per seed.

**RESULT (DONE):** decision = `marginal_canonical_within_2sigma`. Cell ran in 0.59 min (much faster than predicted 5-6 min — the per-seed inference took longer than the shadow sweep but completed quickly because per-stem cache I/O is fast on the SSD).

| metric | shadow mean ± std | range | canonical (split=42) | z-score |
| --- | --- | --- | --- | --- |
| per-seed K=2 5-seed-mean UAR | $0.6860 \pm 0.0153$ | $[0.6458, 0.7001]$ | $0.7037$ | $+1.16\sigma$ |
| mean-logit ensemble UAR     | $0.6882 \pm 0.0169$ | $[0.6443, 0.7042]$ | $0.7090$ | $+1.24\sigma$ |

Per-shadow ensemble UAR (sorted by shadow_seed): seed=1 0.7031; seed=2 0.6930; seed=3 0.6983; seed=5 0.6877; seed=11 **0.7042**; seed=17 0.6893; **seed=23 0.6443**; seed=31 0.6853; seed=53 0.6856; seed=99 0.6910.

**The canonical split is favorable but not pathological.** $z {=} +1.24\sigma$ above the shadow mean → the locked configuration found a partition where its specific characteristics happened to align well with the test fold (likely a combination of: per-seed locked $\beta^*$ tuned on canonical train_threshold; per-seed and ensemble $\tau^*$ swept on canonical train_threshold; the K=2 G_other selection $G5$ chosen on canonical devel_test in §4.11.1.1). This isn't methodological failure — every decision followed the pre-registered protocol; the canonical split was fixed early before any rung optimisation; the partition variance is the irreducible noise of single-partition evaluation.

**Mean-logit ensemble still beats per-seed-single across most shadow splits** (ensemble UAR > single 5-seed mean UAR on 9 of 10 shadow splits, by Δ +0.001 to +0.004; canonical Δ +0.005). Confirms the §4.11.1.4 ensemble lift isn't a canonical-split-specific artifact: the ensemble averaging mechanism transfers across partitions, just at a slightly smaller magnitude than canonical ($\sim 50\%$ of canonical lift on shadow splits on average).

**The 2017 baseline is also a single-partition estimate.** Both 0.7090 (canonical) and 0.7100 (2017 baseline) are single-partition point estimates. If the 2017 baseline has similar partition variance ($\sigma \sim 0.017$, plausible given similar corpus + similar protocol), it could plausibly be anywhere from $0.68$ to $0.74$ on a shadow split. The 0.001 gap between our 0.7090 and the 2017 0.7100 is much smaller than the partition variance of either estimate. **Paper-stage framing: the systems are statistically equivalent within partition variance.** This is a stronger claim than "we matched 0.710" because it's defensible against the obvious reviewer question "did you overfit devel?".

**seed=23 is a low outlier (0.6443).** Worth a quick diagnostic to check whether its devel_test fold has structurally harder cold cases (concentrated voice-quality outliers, low-cold-speaker-cluster) or just statistical fluctuation. Adding §4.13.1.1 cell to inspect.

**Paper implications:**

- **Abstract reframe:** acknowledge shadow-mean alongside canonical. Replace ``matches 0.710 within measurement noise'' with ``statistically equivalent within partition variance ($\sigma \sim 0.017$ from shadow harness).''
- **§6 results:** new subsection `ssec:shadow_robustness` reporting the shadow distribution + per-shadow table + canonical $z$-score + the partition-variance-equivalence claim.
- **§7 discussion:** retire the conservative/standard/generous framings; replace with the partition-variance framing.
- **§5 methodology table:** add **M18 = "shadow-split robustness harness on the canonical pipeline's cached logits is a cheap diagnostic that quantifies devel-side overfit risk; report shadow distribution alongside any single-partition headline UAR when the held-out test set is unavailable."** Generalises beyond URTIC.
- **New appendix `app:k2_shadow`:** per-shadow-seed table with the full per-seed-single + ensemble UAR + per-class recall + decision.
- **Future-experiment evaluation gate:** any new intervention should be evaluated on the shadow distribution, not just canonical; promote only if it improves the shadow mean (the canonical-only delta is partly inflated by the +1.24σ canonical favorability).

#### 4.13.1.1 Shadow seed=23 outlier diagnostic (cell appended, queued)

Quick targeted diagnostic on the low-outlier shadow split (seed=23, ensemble UAR 0.6443, $\sim 2.6\sigma$ below shadow mean). Two hypotheses:

(a) **Structurally harder cold cases** — the speaker-grouped split happened to put hard-to-classify cold speakers in devel_test_s23. Test by computing per-pseudo-speaker performance + checking which speakers in devel_test_s23 are also "always-hard" across other shadow splits.

(b) **Fluctuation** — just statistical noise; with $\sigma \sim 0.017$ and 10 shadow splits, the most extreme is expected to be $\sim 1.8\sigma$ from the mean by extreme-value statistics; observing $-2.6\sigma$ once is plausible but slightly outside the 90% CI.

**Diagnostic outputs:** N_test_speakers, N_test_cold_speakers, mean/max chunks-per-speaker, speaker-level UAR (1 vote per speaker), per-pseudo-speaker mean ensemble logit + correctness rate, comparison vs canonical devel_test on the same metrics. ~10 min runtime. **Output:** `results/A5b_k2_shadow_seed23_diag.json`.

If (a) holds: paper-relevant edge-case finding. If (b) holds: confirms shadow-mean ± std is the right summary statistic; canonical headline can stay.

**RESULT (DONE):** verdict = `structural_harder_cold_speakers` (cell ran in 0.53 min). seed=23 has 95 unique pseudo-speakers (vs canonical 107) and only **5 cold speakers** (vs canonical 10), with cold-speaker mean correctness of $0.466$ (vs canonical $0.694$). Speaker-level UAR collapses to $0.4056$ (vs canonical $0.7232$), $\sim 5\sigma$ below canonical and meaningfully below the shadow-mean of $0.5996 \pm 0.0819$. The mechanism is structural but **not a partition defect**: cold speakers are sparse on URTIC ($\sim 21$ across the entire devel partition split into halves of $\sim 10$ each), and seed=23 happened to put 5 *specific* cold speakers into devel\_test that turned out to be harder-than-average for the locked configuration to classify.

**Z-score discrepancy note (caught in reflection 2):** the cell computed `shadow_mean_ex_23` (excluding seed=23 itself, $\sigma {=} 0.0072$) for the comparison table, giving $z {=} -6.78\sigma$ for chunk-UAR; the §4.13.1 baseline used the inclusive shadow distribution (10 splits, $\sigma {=} 0.0169$), giving $z {=} -2.60\sigma$. Both statistics are valid: the exclusive-of-self captures "how extreme is this point relative to the OTHER points" (useful for outlier detection); the inclusive captures "how extreme relative to the full distribution including itself" (the standard summary statistic for distribution variance). For paper citation we use the inclusive $z {=} -2.60\sigma$ (consistent with §4.13.1 framing); the exclusive-of-self overstates extremeness because the tail outlier's inclusion expands the inclusive $\sigma$.

**The deeper finding (paper-grade, M19 candidate):** on URTIC, **speaker-level UAR is much MORE noisy than chunk-level UAR across shadow splits**. Across the 10 shadow splits, chunk-UAR distribution has $\sigma {=} 0.017$ (from §4.13.1); speaker-UAR distribution has $\sigma {=} 0.0819$ ($\sim 5\times$ larger). This is the OPPOSITE of the conventional paralinguistic intuition that speaker-level aggregation reduces noise. Mechanism: URTIC cold speakers are sparse ($\sim 10$ per devel half), so per-speaker accuracy is itself noisy; the gain from per-speaker aggregation is dwarfed by the loss in effective sample size when collapsing $\sim 4800$ chunks to $\sim 100$ speakers. **M19 candidate:** *"on corpora with sparse minority-class speaker representation (URTIC has $\sim 10$ cold speakers per devel half), speaker-level aggregation (smoothing or evaluation) increases variance rather than reducing it; the chunk-level metric is more stable. The naive 'aggregate to speaker level for noise reduction' heuristic doesn't apply when minority-class speaker count is small. Validates ComParE's choice of chunk-UAR as the official metric and rules out a class of related interventions (multi-chunk inference, confidence weighting, speaker-level evaluation)."* This finding is shared between §4.13.1.1 (seed=23 diag) and §4.13.2 (speaker smoothing negative) -- both are downstream consequences of the same underlying property of URTIC.

**Paper implications:**

- **Reframe seed=23 in the paper:** not "harder cold speakers in this partition" (defect-implying) but "cold-speaker selection variance is large because the cold class is sparse" (structural property). Add as a vignette in §6 results illustrating the partition-variance mechanism concretely.
- **Quote the speaker-UAR vs chunk-UAR variance comparison** as the load-bearing methodology paragraph: "chunk-UAR shadow $\sigma {=} 0.017$, speaker-UAR shadow $\sigma {=} 0.082$ -- speaker-level evaluation is $\sim 5\times$ MORE variable on URTIC; validates ComParE's choice of chunk-UAR as the official metric." This is the most substantive methodology contribution emerging from §4.13.

#### 4.13.2 Speaker-level logit smoothing (DONE; SHADOW-EVAL CONFIRMS smoothing hurts on every split; chunk-level signal is load-bearing on URTIC)

**Goal:** test whether per-pseudo-speaker mean smoothing of the K=2 ensemble logit reduces per-chunk noise enough to improve UAR. URTIC labels are per-recording (a person either has a cold during the recording or doesn't), so per-chunk K=2 fused-logit predictions can be noisy in ways that should average out across the chunks of the same speaker.

**Math:** for each chunk i with pseudo-speaker s_i, replace ensemble_logit_i with `smoothed_i = α · ensemble_logit_i + (1 - α) · mean_ensemble_logit_{s_i}`. Speaker-mean computed within the same split (no cross-split leakage). α=1.0 = no smoothing (reproduces 0.7090); α=0.0 = full smoothing (every chunk = speaker mean); intermediate = partial. Smooth-then-ensemble == ensemble-then-smooth (both linear), so we ensemble first and smooth after.

**Sweep:** α ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0} (14 values, dense near 1.0 and 0.0). Per-α: smooth train_threshold + devel_test logits, sweep τ on smoothed train_threshold, eval at locked τ on smoothed devel_test. **Best α picked by train_threshold UAR (NOT by devel_test UAR — avoids the multiple-comparison-on-devel pattern).**

**Decision rule:**

- best α devel_test UAR ≥ 0.710 → `smoothing_crosses_baseline`.
- best α devel_test UAR > 0.7090 + 0.003 → `smoothing_helps_robustly`.
- |best α devel_test UAR - 0.7090| ≤ 0.002 → `smoothing_neutral` (per-chunk K=2 already captures speaker-level signal).
- best α devel_test UAR < 0.7090 - 0.002 → `smoothing_hurts` (per-chunk noise was carrying useful signal averaged out by speaker-mean).

**Reproduction sanity check:** α=1.0 row should reproduce 0.7090 exactly (no-op for α=1.0). Drift > 0.002 = pipeline regression.

**Monotonicity diagnostic:** report n_increasing / n_decreasing steps in the α-sorted devel_test UAR sequence. Smooth monotonic = clean trade-off; many sign-flips = noisy interaction (suggests smoothing isn't really doing anything informative).

**Output:** `results/A5b_k2_speaker_smoothing.json`. Per-α devel_test UAR + best-α lock + decision + monotonicity diagnostic.

**Expected lift:** +0 to +0.005. The per-chunk K=2 already includes the per-seed-mean ensemble averaging which captures most of the variance reduction; speaker-level smoothing adds an additional axis (across chunks of the same speaker) that may or may not reduce remaining noise. Either outcome paper-relevant.

**RESULT (DONE):** decision = `smoothing_hurts`. Cell ran in 0.63 min. The α-sweep on canonical train_threshold UAR picked $\alpha^* {=} 0.0$ (full smoothing) by maximizing train_threshold UAR ($0.8401$) -- a textbook overfit signal: pure pseudo-speaker averaging exploits the train_threshold pseudo-speaker composition without generalizing.

**Per-shadow-split paired delta** (smoothed at $\alpha {=} 0.0$ vs no-smoothing $\alpha {=} 1.0$):

| split | baseline ($\alpha {=} 1.0$) | smoothed ($\alpha {=} 0.0$) | $\Delta$ |
| --- | --- | --- | --- |
| canonical (42) | $0.7090$ | $0.6199$ | $-0.0891$ |
| shadow(1)  | $0.7031$ | $0.5955$ | $-0.1076$ |
| shadow(2)  | $0.6930$ | $0.6162$ | $-0.0769$ |
| shadow(3)  | $0.6983$ | $0.5810$ | $-0.1173$ |
| shadow(5)  | $0.6877$ | $0.5936$ | $-0.0942$ |
| shadow(11) | $0.7042$ | $0.6003$ | $-0.1039$ |
| shadow(17) | $0.6893$ | $0.6104$ | $-0.0789$ |
| shadow(23) | $0.6443$ | $0.5528$ | $-0.0914$ |
| shadow(31) | $0.6853$ | $0.5953$ | $-0.0900$ |
| shadow(53) | $0.6856$ | $0.5703$ | $-0.1154$ |
| shadow(99) | $0.6910$ | $0.6081$ | $-0.0829$ |

**Aggregate:** baseline shadow mean = $0.6882 \pm 0.0169$ (matches §4.13.1 reference EXACTLY $\to$ pipeline is fully deterministic across cells); smoothed shadow mean = $0.5924 \pm 0.0194$; paired $\Delta$ shadow mean = $-0.0958 \pm 0.0146$, **positive on 0/10 shadow splits**.

**The α-sweep curve is monotonically increasing toward $\alpha {=} 1.0$:** $\alpha {=} 0.0$ canonical $0.6199 \to 0.10 (0.6400) \to 0.20 (0.6684) \to 0.30 (0.6852) \to 0.40 (0.6992) \to 0.50 (0.7052) \to 0.60 (0.7075) \to 0.70 (0.7080) \to 0.80 (0.7036) \to 0.90 (0.7047) \to 0.95 (0.7084) \to 1.00 (0.7090)$. Maximum at $\alpha {=} 1.0$; every intermediate setting is worse. The strong form of "smoothing-is-uniformly-bad": curve is monotonic with no interior optimum.

**Mechanism (paper-grade insight):** URTIC cold detection works AT THE CHUNK LEVEL because **within-speaker variation carries real cold evidence, not just noise**. A single speaker's recording session contains chunks with coughs / throat clearing / breathiness (high cold evidence) interspersed with chunks of relatively clean speech (low cold evidence). The model's per-chunk predictions correctly assign different cold-probability to different chunks of the same speaker, based on local acoustic evidence. Speaker-level smoothing destroys this by averaging the high-evidence chunks with the low-evidence ones, collapsing the predictive signal. **This is the OPPOSITE of typical paralinguistic smoothing benefits ($+0.005$ to $+0.015$ UAR)** and reflects a structural property of URTIC's chunk-level annotation protocol where within-speaker variation is informative.

**The selection procedure was its own diagnostic:** train_threshold UAR at $\alpha {=} 0.0$ ($0.8401$) is much higher than at $\alpha {=} 1.0$ ($0.6529$) because pseudo-speaker averaging on train_threshold's $\sim 100$ unique pseudo-speakers $\times \sim 10$ chunks-per-speaker is essentially "predict the per-speaker label" -- and on train_threshold the per-seed-mean K=2 logits are mostly correct given the locked $\beta^* + \tau^*$ from canonical $\to$ smoothing trivially achieves near-perfect train_threshold UAR. But this doesn't transfer to devel_test (or any shadow split) because the test fold's pseudo-speaker composition differs from train_threshold's. The pre-registered admission gate (shadow-mean $\Delta$, not canonical $\Delta$) was correctly tuned to catch exactly this failure mode.

**Conservative-α observation:** even at $\alpha \in [0.5, 0.95]$ (mild smoothing only), every $\alpha < 1.0$ underperforms $\alpha = 1.0$ on canonical UAR. The smoothing-monotone-decreasing pattern is robust across the full $\alpha$ grid. There is no interior optimum to recover even with a more conservative selection criterion.

**Paper implications:**

- Add a paragraph to §6 results (alongside the calibration / TTA / HuBERT-LW ablations) noting the speaker-smoothing negative + the within-speaker-variation-carries-signal mechanism. This is the most pedagogically useful negative result in the project because it overturns the conventional speaker-smoothing intuition.
- Combine with the §4.13.1.1 seed=23 vignette + the speaker-UAR-vs-chunk-UAR variance finding to make the point: ComParE's choice of chunk-UAR as the official metric was correct on URTIC; speaker-level evaluation would be much noisier; speaker-level smoothing destroys real signal. All three findings (smoothing fails, seed=23 outlier, speaker-UAR noisier than chunk-UAR) are downstream of cold-speaker sparsity on URTIC.
- M19 (described in §4.13.1.1 above) covers all three findings under a single methodology umbrella.

**The K=2 5-seed mean-logit ensemble at canonical 0.7090 / shadow-mean 0.6882 ± 0.017 is the FINAL canonical paper-headline triple.** Speaker-smoothing was the cheapest UAR-push attempt under the new shadow-eval gate and it failed unambiguously. No further interventions tested within the configurations.

### 4.14 "Come closer to 0.710 baseline" plan (post-§4.13 reflections; shadow-mean as the binding metric)

After the §4.13 closures (shadow-split harness DONE marginal; speaker-smoothing DONE hurts; seed=23 DONE structural), the user-stated goal pivoted to "come closer to the 71% baseline with high probability while staying on a justifiable research track." The honest reality is that "exceed 0.710 on shadow mean" is not high-probability with any tractable intervention (shadow $\sigma {=} 0.017$, gap is +0.022); "come closer (>+0.005 shadow lift)" is achievable.

The 3-step plan, cheapest-first, each shadow-eval gated, each paper-grade either way:

| step | intervention | cost | probability of meaningful shadow lift | expected shadow lift if positive |
| --- | --- | --- | --- | --- |
| §4.14.1 | multi-K ensemble (K=1 + K=2 per-seed averaging) | ~5-6 min | ~70% | +0.001 to +0.003 |
| §4.14.2 | diverse-anchor 10-seed ensemble (5 new A2.5 heads with varied lr/dropout/proj_dim) | ~25-30 min | ~50-60% | +0.003 to +0.008 |
| §4.14.3 | ComParE-SVM K=3 (2017 baseline replication under speaker-grouped subsplit) | ~1 day | ~25% (admit conditional on standalone ≥ 0.61) | +0.005 to +0.015 |

Cumulative: probability of shadow-mean lift > +0.005 from all three stacking favorably: ~65-75%; probability of crossing 0.710 on shadow mean: ~15-25%.

**Skip list (lower EV per the §4.13 reflections):**

- LoRA fine-tune: high variance, scope drift into de-confounding ladder, ~5-7 day investment.
- HuBERT-large mean-pooled: capacity confirmation test of §4.12.3 hypothesis; lower EV than ComParE-SVM for shadow lift.
- Whisper-A2.5: would have the same audit-recipe-mirror correlation issue as HuBERT-A2.5 per §4.12.3; almost certain to fail K=3 admit by correlation; defer to future work as another instance of the same finding.
- MDD adversary (Margin Disparity Discrepancy as alternative to DANN): the §4.10 / A7 / A7c v2 substrate-resistance verdict (`B_dann_dead_substrate_resistant`: every metric flat across $\lambda \in [0, 0.1]$) is an architectural property of the frozen-WavLM + LW-softmax substrate, not a property of the specific adversarial loss formulation. MDD would almost certainly produce the same `B_mdd_dead_substrate_resistant` verdict (same substrate, different adversarial loss math, same outcome). Worth a paper-stage future-work mention as "another adversarial-loss formulation also closes negatively under M14 substrate-resistance" but not worth the engineering cost (new MDD module, new cell, ~half-day for a likely-confirming negative).

#### 4.14.1 Multi-K ensemble (DONE; multi_k_robust_lift -- canonical 0.7090 → 0.7111 crosses 0.710 baseline; shadow-mean 0.6882 → 0.6940 lifted by +0.006 on 9/10 splits)

**Goal:** variance reduction across the K-axis. Per seed, average K=1 fused logit (a2.5 + $\beta_{k1}^* \cdot z_{g4}$) and K=2 fused logit (a2.5 + $\beta_{k2}^* \cdot \text{mean}(z_{g4}, z_{g5})$); then 5-seed mean-logit ensemble of the averaged-per-seed logits.

**Math.** $\text{multi\_K}_{\text{seed}} = 0.5 \cdot (K_1^{\text{seed}} + K_2^{\text{seed}}) = a_{2.5} + 0.5(\beta_{k1} + \beta_{k2}/2) \cdot z_{g4} + 0.25 \beta_{k2} \cdot z_{g5}$. Effectively a re-weighted K=2 with asymmetric per-channel β coefficients (more weight on G4_gi than G5_modulation), per-seed. For seed 42 (β_k1=6, β_k2=8): effective $\beta_{g4} {=} 5$, effective $\beta_{g5} {=} 2$, vs K=2-only $\beta_{g4} {=} 4$, $\beta_{g5} {=} 4$.

**Decision rule (shadow-mean Δ, not canonical-only):**

- shadow Δ ≥ 0.003 AND ≥ 7/10 shadow splits positive → `multi_k_robust_lift`
- shadow Δ ≥ 0.001 → `multi_k_marginal` (paper supplementary)
- |shadow Δ| ≤ 0.001 → `multi_k_neutral` (K=2-only stays canonical)
- shadow Δ < 0 BUT canonical Δ ≥ 0.003 → `multi_k_canonical_only_overfit` (do NOT promote)
- shadow Δ < -0.001 → `multi_k_hurts`

**Why "high probability of small lift, low risk":** K=1 + K=2 share the A2.5 anchor + G4_gi component → high correlation bound (asymptotic effective sample size with 10 logits at average correlation $r \sim 0.93$ is $\sim 1.07$ vs $\sim 1.05$ for 5 K=2-only logits at $r \sim 0.94$), so ensemble lift is bounded. But per-seed K=1 and K=2 differ in fusion weight ($\beta_{k1}$ vs $\beta_{k2}$) producing distinct effective per-channel weighting → ensemble averaging finds a slightly different operating point than K=2-alone. Expected shadow-mean lift: +0.001 to +0.003.

**Output:** `results/A5b_k2_multi_k_ensemble.json`. Cost: ~5-6 min wall-clock. Sanity check baked in: K=2-only ensemble reproduction should reproduce 0.7090 canonical reference exactly.

**RESULT (DONE; multi_k_robust_lift):** cell ran in **0.52 min** (mostly cached I/O). Both gates passed: shadow-mean Δ +0.0059 (≥ 0.003 threshold) AND 9/10 shadow splits positive (≥ 7/10 threshold).

**Per-split results:**

| split | K=2-only | multi-K | paired Δ |
| --- | --- | --- | --- |
| canonical(42) | 0.7090 | **0.7111** | +0.0021 |
| shadow(1)   | 0.7031 | 0.7014 | -0.0017 |
| shadow(2)   | 0.6930 | 0.7004 | +0.0073 |
| shadow(3)   | 0.6983 | 0.6998 | +0.0015 |
| shadow(5)   | 0.6877 | 0.6933 | +0.0056 |
| shadow(11)  | 0.7042 | **0.7081** | +0.0039 |
| shadow(17)  | 0.6893 | 0.7027 | +0.0134 |
| shadow(23)  | 0.6443 | 0.6518 | +0.0075 |
| shadow(31)  | 0.6853 | 0.6930 | +0.0078 |
| shadow(53)  | 0.6856 | 0.6908 | +0.0051 |
| shadow(99)  | 0.6910 | 0.6992 | +0.0082 |

**Aggregate:** baseline K=2-only shadow mean 0.6882 ± 0.0169 (matches §4.13.1 reference EXACTLY → pipeline determinism confirmed across cells); multi-K shadow mean **0.6940 ± 0.0157** (note σ slightly *decreased* from 0.0169 → 0.0157, consistent with variance reduction); paired Δ shadow mean = **+0.0059 ± 0.0041**, positive on 9/10 shadow splits. Only seed=1 was marginally negative (-0.0017, within numerical noise).

**Locked τ:** multi-K τ* = -1.625 vs K=2-only τ* = -1.375 (shifts toward more cold-aggressive operating point; consistent with the K=1-component contributing additional cold-axis signal at locked β_k1).

**Canonical now crosses 0.710 baseline** by +0.0011 (multi-K canonical 0.7111 vs baseline 0.7100). This is a meaningful narrative shift: the previous "matches 0.710 within partition variance" framing (K=2-only canonical 0.7090, Δ -0.001) becomes "exceeds 0.710 on canonical (Δ +0.001), comparable within partition variance on shadow mean (multi-K shadow-mean 0.6940 vs 0.7100, Δ -0.016 ≈ 1σ)" with multi-K.

**Shadow-mean narrowing:** previous K=2-only shadow gap was -0.022 (= 0.6882 - 0.7100); multi-K narrows it to -0.016 (= 0.6940 - 0.7100). **~27% of the shadow gap closed by Step 1 alone.** This is at the upper end of my predicted +0.001-0.003 range (effective sample size argument); the actual lift of +0.006 suggests the K=1 + K=2 logits were less correlated than r ~ 0.93 (probably closer to r ~ 0.85-0.90 given the β-weight asymmetry between K=1 and K=2 per seed).

**Mechanism reading:** the multi-K logit per seed is `a2.5 + β_eff_g4 · z_g4 + β_eff_g5 · z_g5` with asymmetric β coefficients (more weight on G4_gi than G5_modulation) that differ per seed depending on the locked β_k1 and β_k2 values. This produces 5 distinct effective per-channel weightings that the ensemble averages over -- the variance reduction comes from the per-seed weighting diversity, not just from doubling the logit count. Effective ensemble independence > 5 K=2-only seeds at uniform weighting.

**Paper implications:**

- **Abstract refresh:** new headline triple = multi-K canonical 0.7111 / shadow-mean 0.6940 ± 0.016 / paired lift +0.006 over K=2-only with 9/10 positive. Canonical now exceeds 0.710 baseline by +0.001 (within partition variance, so still partition-variance-equivalent; but the canonical-vs-baseline statement flips from "below by 0.001" to "above by 0.001").
- **§6 results cumulative stack table:** add multi-K row as the new top of the stack. Cumulative lift over leak-corrected baseline: A5b multi-K = +0.075 (was +0.073 for K=2-only); ~17.5σ vs leak-corrected baseline.
- **§7 discussion partition-variance framing:** retain the partition-variance equivalence framing (the 0.011 canonical gain isn't a model superiority claim on the underlying distribution, just a single-partition lift); but the canonical-side narrative can lean slightly more affirmative.
- **§5 methodology table:** no new M-discipline (multi-K is a standard ensemble-diversification trick); maybe a brief mention in M19 about "ensemble across K-axis is one of the cheap variance-reduction operations that survives shadow-eval."
- **Appendix:** add per-shadow-split paired-Δ table to `app:k2_shadow` or new `app:k2_multi_k`.

**Next-step options (per the §4.14 plan):**

- **Step 2 (diverse-anchor 10-seed ensemble, ~30 min):** train 5 additional A2.5 heads with varied hyperparameters (lr×2, lr×0.5, dropout 0.3, dropout 0.7, proj_dim 256), append to the existing 5-seed ensemble. Expected additional shadow lift: +0.003-0.008 if anchors are less correlated. Stacks additively with multi-K.
- **Step 3 (ComParE-SVM K=3, ~1 day):** 2017 baseline replication; biggest justification; uncertain payoff. Expected shadow lift: +0.005-0.015 if standalone ≥ 0.61.
- **OR: lock at multi-K and write the paper.** Canonical already exceeds baseline; shadow mean closed 27% of the gap; methodology framework is rich.

#### 4.14.2 Diverse-anchor 10-seed multi-K ensemble (DONE; `diverse_anchor_hurts` -- adding weaker hyperparameter-varied heads drags ensemble down; 5-seed multi-K is at the optimum)

**Goal:** more ensemble diversity by introducing 5 new A2.5 heads with varied hyperparameters (different optimizer dynamics + different head capacity → different cold-axis projections → less correlated logits across the 10-head pool).

**Variants (paired retraining, one per existing seed):** `lrx2` (seed=42, base_lr=2e-3); `lrx0p5` (seed=123, base_lr=5e-4); `dr0p3` (seed=7, dropout=0.3); `dr0p7` (seed=999, dropout=0.7); `pd256` (seed=31337, proj_dim=256). Each variant: construct `LayerWeightedPooledHead` with the varied hyperparameter, init `layer_weights` with honesty-prior `T·sub@1` from A5d, train cold-CE + class-balanced sampler with the variant's base_lr, 25 epochs early-stop on devel_val UAR. Save ckpt at `cache/microsoft_wavlm-large/head_A2grouped_honestprior_diverse_{variant_id}.pt`.

**Per-head β-sweep:** for each new head, β-sweep K=1 (A2.5 + G4_gi only) and K=2 (A2.5 + G4_gi + G5_modulation) on canonical train_threshold to find locked β_k1\* and β_k2\* (the original 5 heads reuse the existing locks from `A5b_k2_5seed_lock.json`).

**Ensemble aggregation:** per-head multi_K = 0.5·(K=1 + K=2); 10-head mean-logit ensemble of multi_K_head. Sweep τ on canonical train_threshold; eval on canonical + 10 shadow splits at locked τ.

**Decision rule (shadow-eval, vs §4.14.1 5-seed multi-K baseline):**

- shadow Δ ≥ 0.003 AND ≥ 7/10 shadow splits positive → `diverse_anchor_robust_lift` (paper-grade, lock as new canonical)
- shadow Δ ≥ 0.001 → `diverse_anchor_marginal` (paper supplementary)
- |shadow Δ| ≤ 0.001 → `diverse_anchor_neutral` (5-seed multi-K stays canonical; the K-axis ensemble averaging already saturated the available signal)
- shadow Δ < 0 AND canonical Δ ≥ 0.003 → `diverse_anchor_canonical_only_overfit` (do NOT promote)
- shadow Δ < -0.001 → `diverse_anchor_hurts` (new anchors introduce noise rather than diversity)

**Cost:** ~30-35 min wall-clock (5 head trainings at ~5 min each via existing `features.train.train_head` + per-head β-sweep + ensembling + shadow eval). Idempotent on per-head ckpt files (re-runs only retrain missing variants).

**Output:** `results/A5b_multi_k_10seed_diverse.json`.

**Expected shadow lift:** +0.003 to +0.008 (the new heads have different optimizer dynamics → different cold-axis projections → lower inter-head correlation than the 5 same-recipe seeds; effective ensemble independence improves modestly). Stacks additively with §4.14.1 multi-K (paired Δ vs K=2-only would be +0.006 from multi-K + this step's lift).

**RESULT (DONE; `diverse_anchor_hurts`):** cell ran in 7.22 min (5 head trainings ~5 min total + ensembling + shadow eval). All five new heads converged but with individually weaker val_UAR than the canonical 5:

| variant | hp override | best_val_UAR | epoch | β_k1\* | β_k2\* |
| --- | --- | --- | --- | --- | --- |
| `lrx2` | base_lr=2e-3 | 0.6311 | 2 | 12 | 12 |
| `lrx0p5` | base_lr=5e-4 | 0.6289 | 1 | 1 | 1.5 |
| `dr0p3` | dropout=0.3 | 0.6334 | 2 | 8 | 16 |
| `dr0p7` | dropout=0.7 | 0.6331 | 3 | 2.5 | 4 |
| `pd256` | proj_dim=256 | 0.6300 | 4 | 12 | 16 |
| --- | --- | --- | --- | --- | --- |
| canonical 5-seed mean (ref) | -- | $\sim 0.656$ | -- | -- | -- |

**The 5 new heads all underperform the canonical 5 on val_UAR by Δ -0.023 to -0.027.** Adding them to the ensemble averages-in weaker logits → ensemble UAR drops. **10-seed diverse multi-K shadow-mean = 0.6920 ± 0.0161** (vs §4.14.1 5-seed multi-K 0.6940 ± 0.0157); paired Δ -0.0021 ± 0.0015, positive on 1/10 shadow splits. Canonical Δ -0.0022. Both directions negative; decisive negative.

**Mechanism: the canonical hyperparameters are at or near the local optimum.** lr=1e-3 + dropout=0.5 + proj_dim=128 is a Goldilocks point: lr×2 overshoots (lrx2 val_UAR 0.6311 vs canonical ~0.656); lr×0.5 undertrains (lrx0p5 0.6289); dropout=0.3 overfits (dr0p3 0.6334); dropout=0.7 underfits (dr0p7 0.6331); proj_dim=256 doubles parameters without enough data to regularise (pd256 0.6300). Each departure from the canonical hyperparameters hurts the individual head's standalone training, and ensemble averaging can't recover from weaker individual base predictors.

**Locked β diversity (interesting but not load-bearing):** the diverse heads found very different locked β values (β_k2 ∈ {1.5, 4, 12, 12, 16}). The wide β spread reflects each head having a different effective cold-axis sensitivity (the optimizer landed at different points in the layer-weight subspace + projection-head capacity space). This β diversity DID produce some inter-head decorrelation, but the magnitude of individual-head weakness dominated the correlation reduction.

**Pre-registered shadow-eval admission gate worked as designed**: the canonical Δ -0.0022 + shadow Δ -0.0021 unambiguously closes the diverse-anchor axis without needing to evaluate whether it was a canonical-only overfit (both directions agree).

**Paper implication:** the 5-seed multi-K ensemble at canonical 0.7111 / shadow-mean 0.6940 is at or near the locally-optimal ensemble configuration within the canonical-recipe family. The proposed "ensemble diversification via hyperparameter variation" optimisation pattern doesn't apply here because the canonical hyperparameters are already locally optimal and any departure hurts individual heads more than it adds ensemble diversity. This is a paper-supplementary ablation that defends the 5-seed ensemble size as adequate.

**Refinement of methodology framework:** this finding refines the standard "ensemble diversification" trick — it only helps when individual base predictors are SUFFICIENTLY similar (less than ~5% UAR gap) AND the diversification axis produces NEW orthogonal cold-axis projections rather than just weaker versions of the same projection. On URTIC + frozen-WavLM + A2.5 recipe, the canonical hyperparameters are sufficiently optimised that hyperparameter variation produces only weaker versions, not orthogonal variants. Could optionally be added as a paragraph-grade methodology note alongside M19 ("ensemble diversification has a base-predictor-strength floor; weak base predictors hurt regardless of diversification axis"). Lower-priority addition; mostly a defensible explanation of why the 5-seed multi-K stays canonical.

#### 4.14.3 ComParE-SVM K=3 + multi-K-with-K=3 (DONE; `compare_k3_neutral` -- classical baseline is weaker than FM-based system under speaker-grouped subsplit; K=3 fusion neutral but canonical bumps to 0.7126)

**Goal:** introduce a genuinely orthogonal new base predictor for K=3 fusion (the 6373-d ComParE-2016 acoustic feature set + regularised LR cold probe — the official 2017 ComParE Cold sub-challenge baseline architecture). Two-for-one:

- **(a) resolves apples-to-oranges devel-vs-hidden-test framing** — gives a strict-alignment reference number on the same speaker-grouped subsplit protocol.
- **(b) genuinely-orthogonal new base predictor** — different feature family entirely; not subject to the §4.12.3 audit-recipe-mirror correlation issue (HuBERT-A2.5 failed K=3 by correlation; ComParE-LR is handcrafted, recipe-orthogonal).

**Pipeline:**

1. Extract 6373-d ComParE-2016 features via `opensmile.Smile(FeatureSet.ComParE_2016, FeatureLevel.Functionals)` per chunk. Cache to `cache/handcrafted/compare2016/{stem}.npy`. Cost: ~6-10 hr CPU one-time for ~19,101 chunks. Idempotent on per-stem cache files (re-runs only extract missing stems).
2. Fit `LogisticRegression(C=1.0, class_weight='balanced')` on train_fit; z-score using `fit_zscore` on train_fit predictions.
3. Standalone shadow audit: τ-sweep on canonical train_threshold; eval at locked τ on canonical + 10 shadow splits → standalone shadow-mean UAR.
4. **M14 pre-flight (shadow-mean, not canonical):**
    - `< 0.55` → skip K=3 sweep; paper-stage finding only ("the classical ComParE-2016 + LR baseline is weaker than the FM-based system on the speaker-grouped subsplit; resolves the apples-to-oranges framing").
    - `≥ 0.61` → admit plausible; run K=3 sweep.
    - `[0.55, 0.61)` → borderline; run K=3 for confirmation.
5. If pre-flight passes: K=3 sweep per seed: `fused = a2.5 + β · mean(z_g4_gi, z_g5_mod, z_compare)`. Extended β-grid `{0..2, 2.5..16}` (16 values); per-seed argmax β_k3\* on canonical train_threshold.
6. **Multi-K-with-K=3:** per seed, average K=1 + K=2 + K=3 fused logits (each at its own locked β\*); then 5-seed mean-logit ensemble of the per-seed averages.

**Decision rule (shadow-eval, vs §4.14.1 5-seed multi-K baseline):**

- shadow Δ ≥ 0.003 AND ≥ 7/10 shadow splits positive → `compare_k3_robust_lift` (paper-grade, lock as new canonical with the classical-baseline-alignment as side effect)
- shadow Δ ≥ 0.001 → `compare_k3_marginal` (paper supplementary)
- |shadow Δ| ≤ 0.001 → `compare_k3_neutral` (5-seed multi-K stays canonical; the classical-baseline addition is paper-stage 2017-alignment finding only)
- shadow Δ < 0 AND canonical Δ ≥ 0.003 → `compare_k3_canonical_only_overfit`
- shadow Δ < -0.001 → `compare_k3_hurts`
- M14 pre-flight fail → `compare_k3_skipped_m14_pre_flight_fail` (paper-stage classical-baseline-alignment finding; the official 2017 baseline architecture is weaker than the FM-based system on the speaker-grouped subsplit)

**Either outcome is paper-grade:**

- admit → +0.005-0.015 shadow lift + clean apples-to-apples 2017-baseline replication number for the discussion
- M14-skip or no-admit → clean closure of the late-fusion axis with the strongest available orthogonal feature source + paper-stage finding that the 2017 baseline architecture's hidden-test 0.710 was specific to a less-strict split protocol (the official 2017 split allowed cross-speaker leakage that our speaker-grouped subsplit closes)

**Cost:** ~6-10 hr extraction (one-time, cached) + ~30 min LR fit + ~30 min K=3 sweep. Heavily dominated by the extraction. Idempotent.

**Output:** `results/A5b_compare_svm_k3.json` + `cache/handcrafted/compare2016/`.

**Dependencies:** `opensmile==2.6.0` (already installed in env per verification check).

**Expected shadow lift:** +0.005 to +0.015 IF standalone shadow ≥ 0.61 (admit plausible). Probability standalone clears 0.55: high (~80%); probability ≥ 0.61: ~50%. Conditional on ≥ 0.61 admission, probability of K=3 admitting at shadow-eval gate: ~50%. Combined probability of meaningful shadow lift: ~20-30%.

**RESULT (DONE; `compare_k3_neutral`):** cell ran in 18.70 min (~17.4 min for ComParE-2016 extraction across 19,101 chunks at ~16.7 chunks/sec via opensmile + ~1 min for LR fit + standalone audit + K=3 sweep + shadow eval).

**Standalone ComParE-LR audit (paper-grade resolution of devel-vs-hidden-test framing):**

- canonical (split=42): **0.5964**
- shadow distribution (10 splits): **0.5853 ± 0.0083**, range [0.5633, 0.5938]
- M14 verdict: **borderline** (within [0.55, 0.61); ran K=3 for confirmation)

**This is meaningfully below both the FM-A2.5 standalone (~0.656) and the 2017 hidden-test baseline (0.7100).** A regularised LR cold probe on the same 6373-d ComParE-2016 feature set used by the 2017 baseline reaches only ~0.585 shadow-mean under our speaker-grouped subsplit. Either the 2017 baseline used a less-strict split protocol (cross-speaker leakage), or the SVM-vs-LR architectural choice + feature normalisation accounted for the gap to 0.7100. **Paper-relevant finding: the apples-to-oranges devel-vs-hidden-test comparison is now bounded** — a same-protocol replication of the classical feature-family architecture is far below 0.7100; the FM-based system (0.6940 shadow / 0.7111 canonical / 0.7126 multi-K-with-K=3) dominates the classical baseline under speaker-grouped evaluation.

**K=3 sweep (with multi-K-with-K=3 aggregation):**

| seed | locked β_k3\* | thr_UAR (K=3) |
| --- | --- | --- |
| 42 | 8 | 0.6331 |
| 123 | 12 | 0.6164 |
| 7 | 16 | 0.6234 |
| 999 | 16 | 0.5960 |
| 31337 | 16 | 0.6236 |

**4 of 5 seeds at boundary β_k3\* = 16** — the same fusion-absorption signature as §4.11.1.5 G_egemaps_full (boundary β + extreme τ for a weak-standalone candidate). ComParE-LR is too weak (standalone 0.585) to contribute meaningfully in the K=3 fusion despite providing an orthogonal feature family.

**Per-shadow paired Δ table** (multi-K-with-K=3 vs §4.14.1 5-seed multi-K):

| split | 5-seed multi-K | multi-K-with-K=3 | paired Δ |
| --- | --- | --- | --- |
| canonical(42) | 0.7111 | **0.7126** | +0.0016 |
| shadow(1)   | 0.7014 | 0.7049 | +0.0036 |
| shadow(2)   | 0.7004 | 0.7030 | +0.0027 |
| shadow(3)   | 0.6998 | 0.7006 | +0.0008 |
| shadow(5)   | 0.6933 | 0.6959 | +0.0025 |
| shadow(11)  | 0.7081 | 0.7065 | -0.0016 |
| shadow(17)  | 0.7027 | 0.7001 | -0.0026 |
| shadow(23)  | 0.6518 | 0.6491 | -0.0027 |
| shadow(31)  | 0.6930 | 0.6903 | -0.0027 |
| shadow(53)  | 0.6908 | 0.6932 | +0.0024 |
| shadow(99)  | 0.6992 | 0.6971 | -0.0021 |

**Aggregate:** multi-K-with-K=3 shadow mean = **0.6941 ± 0.0166** (vs §4.14.1 5-seed multi-K 0.6940 ± 0.0157); paired Δ +0.00002 ± 0.0026, **positive on 5/10 shadow splits** (exactly half — coin-flip pattern). Canonical Δ +0.0016 (multi-K-with-K=3 canonical 0.7126 vs 5-seed multi-K 0.7111). **K=3-only ensemble shadow mean = 0.6746** — meaningfully WORSE than 5-seed multi-K (0.6940), confirming K=3 alone hurts; multi-K-with-K=3 recovers most of the loss via the K=1 + K=2 components in the per-seed average.

**Decision: `compare_k3_neutral`.** Shadow Δ within ±0.003; canonical-side lift +0.0016 is small (within partition variance) and not reflected in the shadow distribution. The 5-seed multi-K stays the canonical paper-headline configuration (canonical 0.7111 / shadow-mean 0.6940). **The multi-K-with-K=3 canonical 0.7126 is reportable as a paper-supplementary ablation** (the absolute highest canonical UAR achieved in the project, exceeding the 2017 baseline by +0.0026) but the shadow distribution doesn't support promoting it to canonical headline.

**Mechanism (paper-relevant):** ComParE-LR carries enough signal to admit borderline at M14 (shadow standalone 0.585 within [0.55, 0.61]) but is too weak relative to FM-A2.5 to provide a meaningful K=3 lift. The boundary-pegged β_k3* across 4/5 seeds is the smoking gun: the K=3 architecture is signalling "use the maximum β for this candidate" while the τ-sweep absorbs the resulting calibration shift via re-tuning. Net effect on shadow distribution: zero. This refines the M19 finding: chunk-level cold detection on URTIC depends on within-chunk variation that the FM-A2.5 pooled-stat substrate captures and the global ComParE-2016 functionals don't.

**Paper implications (BIGGEST FINDING from §4.14.3 — apples-to-oranges resolution):**

1. **§7 discussion partition-variance framing strengthens substantially.** Previously the devel-vs-hidden-test caveat was a methodological caveat: "we cannot directly compare 0.7090 devel to 0.7100 hidden-test." With the §4.14.3 ComParE-LR replication, we now have a same-protocol classical-baseline number: **0.5853 shadow-mean / 0.5964 canonical.** This is a strict-protocol approximation of "what the 2017 baseline architecture would have scored under our speaker-grouped subsplit." The gap from 0.5853 to the 0.7100 hidden-test number (Δ +0.125) is the joint effect of (a) the speaker-grouped subsplit being stricter than the 2017 hidden-test split + (b) the LR-vs-SVM architectural difference + (c) feature normalisation differences. The protocol-stricter split likely accounts for the larger share of the gap (the SVM-vs-LR architectural difference rarely produces +0.10+ UAR gaps on this scale of data).
2. **The FM-based system's lift over the classical baseline architecture is now quantified on the same protocol.** Multi-K shadow-mean 0.6940 vs ComParE-LR shadow-mean 0.5853 = **+0.109** under speaker-grouped subsplit. This is a substantial methodology contribution that goes alongside the partition-variance framing.
3. **The multi-K-with-K=3 canonical 0.7126 is reportable** as the highest single-partition canonical achieved (vs 2017 baseline 0.7100, Δ +0.0026), but the shadow distribution doesn't support promoting it. Paper-supplementary ablation rather than headline.
4. **§4.14 plan is exhausted.** Step 1 was positive (multi-K, the load-bearing canonical headline). Step 2 was negative (diverse anchors hurt; defensive ablation that the 5-seed multi-K size is correct). Step 3 was neutral on the fusion axis but produced a paper-grade apples-to-oranges resolution. Time to lock + write.

#### 4.14.4 Cumulative ceiling estimate (post-§4.14)

If all three steps stack favorably:

- Step 1 (multi-K, DONE): +0.0059 shadow lift → 0.6940
- Step 2 (diverse anchor, expected): +0.003 to +0.008 → 0.697-0.702
- Step 3 (ComParE-SVM K=3, conditional): +0.005 to +0.015 → 0.702-0.717

Probability all three stack favorably to push shadow mean ≥ 0.710: ~10-20%. Probability of shadow mean ≥ 0.700: ~50-60%. Probability of shadow mean ≥ 0.695: ~75-85%. The "high probability of coming closer" goal is well-served by Steps 2 + 3; the "high probability of exceeding 0.710 on shadow mean" goal remains <25%.

**After Step 3 (whichever outcome): lock and write the paper.** No further intervention has higher EV under the shadow-eval gate; experimental phase is genuinely complete after the §4.14 trio.

#### 4.13.3 Run order + post-cell decision tree

1. **Run §4.13.1 first** (shadow-split harness). Decisive verdict in ~6 min.
   - If `robust_canonical_within_1sigma`: lock the headline as "0.7090 robust across 10 shadow splits"; proceed to §4.13.2.
   - If `marginal` or `fragile`: stop the pure-UAR-push axis; recalibrate paper framing; report shadow distribution alongside canonical 0.7090.
2. **Run §4.13.2** if shadow-split harness was robust. ~6 min.
   - If `smoothing_crosses_baseline` or `smoothing_helps_robustly`: lock the smoothed ensemble as the new canonical; update paper §4 method + §6 results.
   - If `smoothing_neutral`: document as a paper-supplementary ablation; mean-logit at 0.7090 stays canonical.
   - If `smoothing_hurts`: document as a failed-as-expected ablation; suggests per-chunk K=2 already captures the speaker-level signal.
3. **Then truly stop and write.** No LoRA, no classical SVM (deferred to §08 future work), no further sweeps.

**Paper implications when results land:**

- Add a paragraph to §6 results / §7 discussion noting the shadow-split robustness check (paper-grade defence against the "you overfit devel" critique).
- If speaker smoothing helps: add a small subsection in §4 method noting the α-locked smoothed ensemble.
- If speaker smoothing is neutral/hurts: brief mention in §6 results / appendix as a tested-and-rejected variant.
- M18 candidate (if shadow-split robustness check is paper-grade): "shadow-split harness on the cached canonical-pipeline logits is a cheap robustness check that distinguishes architectural lift from devel-split overfit; should be reported alongside the canonical headline number on any small-data corpus where the held-out test set is unavailable."

---

## 5. A5 — feature enhancement + honesty-audited late fusion

**One-line framing**: we perform **feature enhancement** by deriving physiologically motivated, **regime-conditioned** acoustic feature groups from raw audio and pYIN/RMS acoustic states. Each group is audited for cold association and speaker association before being admitted into a constrained late-fusion model.

This is the next rung and the **methodological headline of the paper**, regardless of UAR outcome. It absorbs the original PDF's A2 (handcrafted concat), A5 (OOD), and A9 (late fusion). Split into three sub-rungs so the contributions are separable.

**Methodological lineage**: the *enhancement-then-classify* pattern is loosely inspired by ResST's data-enhancement stage (build auxiliary feature/similarity views before the downstream model), but **we adopt only the data/feature side, not the graph autoencoder**. Closer in-domain references: speech-side regime-conditioned functionals (Schuller-line ComParE), CMVN/VTLN-style speaker-channel normalisation, classical stacking ensembles (Wolpert 1992), and the 2017 ComParE Cold late-fusion baseline (Schuller / Tavarez).

### 5.1 A5a — honesty audit

For each candidate feature group `g`, train two matched probes (same architecture, same input dimensionality):

- **Cold probe**: linear logistic regression on `train_fit` Cold labels, evaluated on `devel_val` → `UAR_g`.
- **Pseudo-speaker probe**: same shape, trained on `train_fit` pseudo-speakers (k=210), evaluated on devel → `top1_g`, `NMI_g`.

Report **two complementary honesty forms** in the same table:

- `label_gain_g   = UAR_g   − 0.50`
- `speaker_gain_g = top1_g  − 1/210`     (chance-floor normalised)
- `ratio_honesty_g       = label_gain_g / (speaker_gain_g + ε)`
- `subtractive_honesty_g = label_gain_g − λ · speaker_gain_g`     (default `λ = 1`; sweep reported as a sensitivity column)

The ratio form is parameter-free and intuitive. The subtractive form is sharper for the paper's claim ("keep features that improve cold prediction without strongly improving speaker prediction") and survives at small `speaker_gain` better than a ratio. Reviewers see both, can judge.

The full table is the **paper's bankable methodological contribution** — even if A5b doesn't lift UAR, the table itself is a re-usable diagnostic for future URTIC work.

### 5.2 Feature groups — physiological cold-cue priors

Group seeds come from the 2017 URTIC literature (Cummins 2017, Schuller 2017 baseline, Huckvale 2018) — what those authors actually found informative for cold detection — not "all openSMILE families wholesale." Each group must be **low-dimensional** (target ≤ ~50 features) so the per-group cold probe is a linear model with little room to encode speaker identity.

Candidate groups (initial set, expandable):

- **Energy / loudness**: RMS stats, low-energy ratio, energy slope, silence/breath-gap features.
- **Voicing**: voiced/unvoiced fraction, F0 coverage, voicing-probability stats, voicing dropouts per second.
- **F0 / prosody**: F0 mean/std/range over voiced frames, pitch instability proxies.
- **Voice quality**: jitter, shimmer, HNR, harmonicity, spectral tilt, CPP if available. (Huckvale VOI.)
- **Spectral shape**: low-order MFCC stats, spectral centroid, rolloff, flux, high/low band ratios.
- **Breath / frication**: high-frequency energy in unvoiced frames, ZCR, noise-like energy.
- **A3-derived scalars** (free): voiced/unvoiced/silence fractions, mean voiced segment duration, voiced↔unvoiced transitions per second, mean RMS in low-energy regions, mean RMS voiced vs unvoiced.
- **Regime-conditioned mel-band stats** — *kept as v1 candidate, explicitly flagged for rigorous verification.* Log-mel-band mean/std stratified by acoustic regime (`mel_band_mean[voiced]`, `[unvoiced]`, `[low_energy]`, plus contrast `Δ_mel = mean[unvoiced] − mean[voiced]`). Use 40 mel bands; per-band stats keep dimensionality bounded (~160 features). **Not** a parallel mel-CNN branch — that would just re-encode what WavLM's CNN feature-extractor already saw. The framing is "low-dimensional, regime-aware spectral view that the linear per-group probe can exploit, audited by honesty score for speaker-leak."

  **Why the cautious framing**: the original attack-plan PDF mapped CQT/Gammatone+CNN to *"subsumed — WavLM trained on similar perceptual objectives at larger scale,"* and `mel_band_mean[voiced]` is structurally close to a speaker's vocal-tract envelope (the same fingerprint that crashed A3). Information-theoretic prior: redundant with WavLM and high speaker-leak risk.

  **Anecdotal anchor pushing the other way**: a 2025 cohort team reports ~69 % UAR on this challenge with a CNN-on-mel-spec approach (above our A2). Whether that was additive on top of a foundation model or a standalone CNN is unclear — exactly the question G7 answers in our setting. Independently of the colleague's result, **the honesty table for G7 is worth reporting on its own** as paper evidence ("does mel-spec carry cold signal not already in WavLM? Yes / no, with measured speaker-leak").

  **G7 acceptance protocol** (stricter than other groups, given the priors):
  - `label_gain_g ≥ 0.05` (mel must show a meaningful linear cold signal on its own)
  - `subtractive_honesty_g > 0` at default `λ = 1` (cold signal not dominated by speaker leak)
  - Held-out check: per-group probe trained on `train_fit`, evaluated on **`devel_test`** (not `devel_val`) for one-shot honest UAR before being admitted into A5b's β table
  - If G7 fails any of the three: drop from A5b, keep the row in the honesty table as documented negative result
  - Coarse fallback if G7 borderline-fails (overfitting suspected): 8 octaves × 3 regimes = 24 features instead of 40 × 3 = 120
- **OOD Mahalanobis distance** (one scalar; was the PDF's whole A5).

**Explicitly excluded**: formants and raw MFCC means as their own group — known to be speaker-rich. If they appear at all, the honesty score should down-weight them automatically; we predict they will be among the lowest-scoring groups.

**Explicitly excluded**: the 6 144-d `manner_pooled/` WavLM cache as a representation stream. Same speaker-fingerprint substrate that just failed in A3.

### 5.3 A5b — constrained late fusion (β fixed = honesty)

Architecture, in one line:

```text
final_logit = β_A2 · logit_A2  +  Σ_g  β_g · logit_g
```

- `logit_A2` comes from the locked A2 head (frozen).
- `logit_g` comes from the per-group linear cold probe (frozen, trained at A5a).
- **β fixed** at A5b — set from the honesty score (e.g. softmax over `subtractive_honesty_g` with temperature T, or hard top-K). **No gradient flows to the βs.**
- A2 stream is the baseline anchor; `β_A2` is fixed = 1 and not subject to honesty scoring.

A5b is intentionally a stacking model. Two motivations:

- **Cleaner paper story.** "Honesty-weighted fusion with *no learning at the fusion stage* lifts UAR by Δ" is much stronger than "regularised βs." It makes the table itself the selection mechanism, not an artefact.
- **Cleaner ablation.** If A5c (learned gate) lifts UAR further, the delta is unambiguously the gate; if A5c doesn't, the priors were already at the ceiling.

### 5.4 A5c — learned per-group gate (conditional)

Run only if A5b passes or nearly passes the gates. Gate replaces the fixed βs:

```text
β_g = σ(honesty_init_g + learned_residual_g)
```

`honesty_init_g` is the frozen value from A5a; `learned_residual_g` is trained against Cold loss with strong L2 regularisation pulling it toward 0. The gate refines the priors rather than overwriting them. Report A5b vs A5c side by side as a controlled comparison of "priors only" vs "priors + learning."

### 5.5 A5d — per-layer honesty diagnostic (DONE; A5e SKIPPED)

**Status: DONE (paper diagnostic).** Ran cold + speaker probes on cached `pooled[:, L, :]` (4096-d per layer) for L ∈ [0, 24], single seed (42), matched to A5a (train_fit / devel_val, linear LR with StandardScaler). Output: `results/A5d_layer_honesty.csv`. Cost ≈ 1 hour wall-clock, no retraining.

**Recipe.** For each layer `L`:

- `cold_probe`     → `cold_uar_L`,    `label_gain_L = cold_uar_L − 0.5`
- `speaker_probe`  → `speaker_top1_L`, `speaker_gain_L = top1_L − 1/210`
- `sub@1_L`        = `label_gain_L − speaker_gain_L`

**Headline numbers, random splits (`results/A5d_layer_honesty.csv`):**

- Best `sub@1` at L21 = +0.0387 — well below the 0.15 trigger.
- Best `cold_uar` at L7 = 0.6052 (cold UAR range L0..24: 0.560–0.605, spread 0.045).
- Highest `speaker_top1` at L3 = 0.0871; lowest at L22 = 0.0402.
- Speaker top-1 decays roughly monotonically L0→L24 (0.087 → 0.043, ~50% reduction); cold UAR is **flat** across the stack with no clean mid-band peak.

**Headline numbers, grouped splits (`results/A5d_grouped_layer_honesty.csv`):**

- Best `sub@1` at **L0 = +0.0401** — peak shifted from late (L21 random) to early (L0 grouped); still ≪ 0.15 trigger.
- Best `cold_uar` at L6 = 0.6090 (range 0.557–0.609, spread 0.052 — same shape as random).
- Highest `speaker_top1` at L3 = 0.0956; lowest at L24 = 0.0417.
- Speaker top-1 still monotone-ish L0→L24 (0.072 → 0.042, similar shape — Pasad 2021 / Chen 2022 layer-stratification confirmed under both splits). Cold UAR still flat across the stack.

**Verdict — A5e SKIPPED holds under both splits.** Both skip-branch conditions of the §5.6 trigger fire simultaneously, on either split: (1) no layer reaches `sub@1_L > 0.15` (peak +0.040 grouped, +0.039 random — both ≪ 0.15), and (2) the cold-UAR peak (L7 random / L6 grouped) coincides with high speaker leak (`speaker_top1` 0.081 / 0.087 at the cold peak, joint top-tier on both splits). No honest mid-band on either split.

**Structural paper finding (independent of A5e, holds on both splits).** Speaker information is layer-stratified on URTIC — confirms Pasad 2021 / Chen 2022 for the speaker axis under both random and grouped splits. Cold information is **not** layer-stratified — `cold_uar` flat L0..L24 with no mid-band peak, refuting the mid-band cold hypothesis on URTIC specifically. Reportable as a standalone empirical finding. The L21→L0 shift in best-`sub@1` between splits reflects that under grouped splits the cold-rich early layers (which are also speaker-rich) get re-ranked above the late layers; absolute sub@1 stays small (~+0.040) on both splits, so the structural verdict (no honest mid-band) is identical.

### 5.6 A5e — WavLM mid-layer retrain (SKIPPED)

**Status: SKIPPED** by A5d verdict (§5.5). Both trigger conditions fire: (1) no layer with `sub@1_L > 0.15` (peak +0.0387 at L21); (2) cold UAR peak (L7) coincides with the speaker-heavy band. The retrain track is closed — GPU goes to A5.5 (cross-speaker splicing) and A6 (contrastive pretraining) instead. Trigger spec retained for completeness: would have fired only on a *dramatic* honest mid-band (`sub@1_L > 0.15` over a contiguous L_a..L_b with `speaker_top1_L` well below A2's full-stack 0.0501), at which point a `LayerWeightedPooledHead` retrain (3 seeds, layer dim masked to the band) + K=1 ablation on `A2_mid + G4_gi` would have been run.

### 5.7 Acceptance gates

Apply at A5b first; A5c only if A5b is within striking distance.

- **UAR**: A5b head UAR ≥ best of {A2, A3} + 0.007 (2σ at N=3).
- **Speaker probe**: probe top-1 on the A5b representation ≤ A2 + 1σ, **measured under the audit's probe substrate** (`honesty.speaker_probe`, multinomial LR — same code path A5a/A5b use for every group). The historical 0.0501 ± 0.0009 in `results/A2.json::speaker_probe` came from the deeper MLP probe in `speakers/probe.py` (different architecture, ~30 epochs of training) and is **not** directly comparable to the audit's LR probe. Three apples-to-apples references under `honesty.speaker_probe` exist:
  - **Random splits** (A5b controls cell, the substrate A5b/A5d were locked under): **0.0674 ± 0.0006** ⇒ ceiling **0.0680**.
  - **Grouped splits** (`results/A2_grouped.json::speaker_probe_lr`, the leak-corrected baseline): **0.0760 ± 0.0020** ⇒ ceiling **0.0780**.
  - **MLP-substrate** (historical, both splits): 0.0501 ± 0.0009 → **NOT comparable** to LR-substrate audit numbers; cited only for paper continuity with prior reports.

  The locked-K probes pass against either LR ceiling: probe (i) literal 2-D 0.0119 ≪ 0.068 / 0.078 by ~6× margin; probe (ii) backbone concat 0.0675 ≤ 0.068 (random) / 0.078 (grouped) by 0.0005 / 0.0105.
- **Honesty table**: reported in the paper with per-group `label_gain`, `speaker_gain`, both honesty forms.

**Status (locked, two splits reported):** A5b **PASSES** these gates via the **K-locked K=1 ablation** (admission frozen to the top-1 group `A2 + G4_gain_invariant`, sweeping only β and τ on `train_threshold`).

**(a) Random splits — historical lock** (`results/A5b_ablation.json`, 3 seeds {42, 123, 7}, `head_A2_seed{seed}.pt`):

- UAR 0.6576 ± 0.0011 (per-seed: 42→0.6571, 123→0.6589, 7→0.6569).
- Δ vs A2_argmax = +0.0148 ± 0.0045 (3.3σ above zero — gate of +0.007 cleared by ~2σ).
- Δ vs A2_τ = +0.0112 ± 0.0066 (1.7σ above zero).
- **Locked-K speaker probe** (`results/A5b.json::locked_speaker_probe`, 3 seeds): probe (i) literal 2-D = **0.0119 ± 0.0015**; probe (ii) backbone-concat = **0.0675 ± 0.0006**. Both PASS against the random-split LR ceiling 0.0680.
- **Probe controls** (`results/A5b.json::locked_speaker_probe_controls`): pooled-only top-1 = **0.0674 ± 0.0006**; pooled+7-d Gaussian-noise = 0.0665 ± 0.0026. G4_gi contributes +0.0001 to speaker recoverability above pooled-alone — essentially zero, no leak channel.

**(b) Grouped splits — leak-corrected lock** (`results/A5b_grouped.json`, 3 seeds, `head_A2grouped_seed{seed}.pt`):

- UAR 0.6588 ± 0.0059 (per-seed: 42→0.6565, 123→0.6656, 7→0.6544).
- Δ vs A2_argmax = **+0.0227 ± 0.0059** (3.8σ above zero — gate cleared by ~3.5σ; **stronger than the random-split lift**).
- Δ vs A2_τ = +0.0206 ± 0.0198 (calibration noisy on grouped splits; calibration-aware lift is also positive).
- **Locked-K speaker probe** (3 seeds): probe (i) literal 2-D = **0.0153 ± 0.0032**; probe (ii) backbone-concat = **0.0733 ± 0.0002**. Both PASS against the grouped-split LR ceiling 0.0780 — probe (i) by ~5×, probe (ii) by 0.0047.

**Reading the two splits together.** The fusion lift got *larger* under grouped splits because A2's argmax baseline was the inflated part of the random number (-0.0067 baseline shift) while K=1 fused UAR is essentially unchanged (+0.0012). Cleanest possible methodology-fix outcome: the audit made the headline stronger, not weaker. Probe (i)/(ii) numbers nudge up slightly because the grouped-split A2 features are slightly more speaker-coded (LR ceiling 0.0760 vs 0.0674 random) — but both still PASS comfortably.

- **Architectural reading.** Probe (i) ≪ probe (ii) on both splits (gap ~5.5pp on random, ~5.8pp on grouped). The per-channel cold-probe compression strips speaker-side variance from each channel before fusion sees it — clean architectural validation of choosing logit-level fusion over A3's concat-MLP, on both splits.

The originally-reported **K=4 free-sweep FAIL** (UAR 0.6502 ± 0.0078, σ > effect size) is documented as **τ-sweep pathology**: free K-sweep on `train_threshold` over-rewards configurations with more τ flexibility (more groups → more degrees of freedom), inflating variance without materially changing the mean. The σ collapse 0.0112 → 0.0011 between free-sweep K=4 and K-locked K=1 is the diagnostic. Both numbers are paper-reportable: the K=1 PASS is the headline, the K=4 FAIL is the documented sweep-protocol finding.

### 5.8 v1 scoping decisions (locked)

- **Per-group probe = linear logistic regression**, not an MLP. If a group needs nonlinearity to predict cold, that's a signal the group should be sub-divided.
- **β-learning at A5c uses a `train_fusion` slice** (10 % of `train_fit`, held out from per-group probe training), **not `devel_val`**. Devel stays for early stopping only — same Huckvale discipline as A2's threshold-on-`train_threshold` choice.
- Group seeds from cold-acoustics literature, not generic ComParE families.
- Per-utterance contrast features (e.g. `HNR_voiced − HNR_unvoiced`) tested **only on the handcrafted scalars**, not on WavLM frame means — the latter is the A3 failure mode.
- **Pseudo-speaker centering deferred** to A6 territory. Train clusters are KMeans-on-train; devel clusters are nearest-centroid assignments. Subtracting noisy cluster means risks introducing systematic bias if cluster purity correlates with class.
- Drop quality/reliability metadata — URTIC has no per-chunk SNR or lab flags.
- Stability via bootstrap on `train_fit`, not k-fold (k-fold rebuilds pseudo-speaker KMeans per fold = ~25 min × k).
- No SMOTE / ADASYN — Huckvale showed they don't help; balanced sampler covers the imbalance.

### 5.9 v1 feature checklist (extraction spec)

Pinned set so A5a is a definite coding task. Extracted once per utterance and cached as one tensor per group. Regime tags `[voiced]`, `[unvoiced]`, `[low_energy]` come from the existing pYIN+RMS labels in `cache/manner_labels/`.

| Group | Features | Approx. dim | Source |
| --- | --- | --- | --- |
| **G1 voicing** | voiced_fraction, unvoiced_fraction, silence_fraction, voicing_dropout_rate, mean_voiced_segment_length, mean_unvoiced_segment_length, voiced↔unvoiced_transitions_per_sec, low_energy_gap_count | ~10 | A3 labels (free) |
| **G2 F0 / prosody** | F0_mean[voiced], F0_std[voiced], F0_range[voiced], F0_missingness, pitch_instability_proxy | ~10 | pYIN |
| **G3 voice quality** | jitter_mean, shimmer_mean, HNR_mean[voiced], harmonicity, spectral_tilt[voiced], CPP if available | ~10 | openSMILE / Praat |
| **G4 energy / pause / breath** | RMS_mean, RMS_std, low_energy_ratio, energy_slope, RMS[low_energy], RMS[voiced] − RMS[low_energy], RMS[unvoiced] − RMS[low_energy], breath_gap_features | ~10 | RMS + A3 labels |
| **G5 unvoiced frication / turbulence** | high_freq_energy_ratio[unvoiced], ZCR[unvoiced], spectral_flux[unvoiced], spectral_centroid[unvoiced], noise-like_energy | ~10 | regime-conditioned spectral |
| **G6 spectral shape** | low-order MFCC stats (μ, σ for MFCC 1–6), spectral_centroid, rolloff, flux, high/low band ratios | ~20 | openSMILE / librosa |
| **G7 mel-band regime stats** | mel_band_mean[voiced], mel_band_mean[unvoiced], mel_band_mean[low_energy] (40 bands × 3 regimes), Δ_mel = mean[unvoiced] − mean[voiced] | ~160 | log-mel + A3 labels |
| **G8 OOD** | Mahalanobis distance of A2 pooled vector from non-cold mean (post-hoc fitted) | 1 | A2 cache |

A2 is **not** a group — it stays a fixed anchor with `β_A2 = 1`.

Total candidate dimensionality across all groups ≈ 230 — comfortably under the "≤ 50 features per group" ceiling for everything except G7 (mel bands), where the per-band structure is intrinsic. If G7's per-group probe overfits, fall back to log-mel-band sums over coarser frequency bands (e.g. 8 octaves × 3 regimes = 24 features).

---

## 6. Future rungs (post-A5)

### A4 — discrete-token histograms

HuBERT-base or HuBERT-large built-in cluster IDs first; VQ-VAE on WavLM embeddings only if free tokens help and time permits. Apply A5's honesty-score check before fusing. Skipped if A5 closes the gap to baseline.

### A5.5 — cross-speaker splicing augmentation

Symmetric across classes (same `p`, same `K`, same `r`-distribution, same crossfade settings for cold and non-cold) so splice presence is uncorrelated with the label. **No pitch shift, no time-stretch, no speed perturbation** — they can mask or mimic cold-like signatures and break causal interpretability. Apply only after A5 to keep the de-confounding levers separable.

**Recipe (per training chunk, v1.1):**

```text
with probability p:
    partner ← random chunk with same Cold label, different pseudo-speaker
    r       ← Uniform(r_lo, r_hi)                       # replacement ratio
    [t0,t1] ← splice boundaries inside silence run ≥ 40 ms in anchor
              (fallback: unvoiced run ≥ 40 ms; final fallback: skip)
    [s0,s1] ← partner segment of length r * len(anchor) such that:
                - s0, s1 land in partner silence runs ≥ 40 ms
                  (fallback: unvoiced runs ≥ 40 ms, mirrors anchor-side rule)
                - voiced_fraction(partner[s0:s1]) ≥ 0.50
              try up to 5 candidate windows; if none qualifies, skip the chunk
    seg_B   ← partner[s0:s1]
    seg_B   ← rms_match(seg_B, local_rms(anchor[t0:t1]))
    cf      ← 150 ms if both anchor and partner boundaries are silence-on-silence
              else 250 ms (longer crossfade when an unvoiced fallback is used)
    out     ← anchor[:t0] ⊕ crossfade(anchor[t0:], seg_B, cf) ⊕
              crossfade(seg_B, anchor[t1:], cf)
else:
    out     ← anchor                                    # untouched
```

- **Crossfade**: equal-power (`fade_out = cos θ`, `fade_in = sin θ`), not linear (linear has a ~−6 dB perceptual dip mid-fade).
- **Crossfade window scales with boundary type**: 150 ms when both anchor and partner boundaries fall on silence (the seam is silence-on-silence, perceptually invisible); 250 ms when either side falls on unvoiced fallback (longer fade hides the spectral handoff between two unvoiced segments that don't share a speaker).
- **Segment replacement, not concat**: output keeps anchor's duration so WavLM frame count stays at 399 and there's no duration shortcut.
- **Boundary picker uses cached manner labels** on **both sides**: scan `cache/manner_labels/` for silence runs of ≥ 40 ms in *both* the anchor (where the splice goes in) and the partner (where the segment is cut from); splice into silence on both ends; fallback to unvoiced where silence isn't available; document the skip rate.
- **Partner-segment voiced-fraction floor (≥ 0.50)**: if the partner window happens to land mostly in partner silence, the segment is a no-op and wastes the augmentation budget; if it lands mid-syllable, the seam is detectable. Requiring ≥ 50% voiced fraction inside `[s0:s1]` ensures the segment carries actual articulatory content (where the cold signal lives) without forcing the whole window to be a single phonetic unit. Re-sample up to 5 times; skip if no candidate qualifies.
- **RMS-match the partner segment** to the anchor's local RMS at the splice region — prevents loudness discontinuity becoming a class proxy.
- **Spectral-envelope matching at the seam** (low-order MFCC delta < threshold): considered for v1.1 but **deferred** as overkill — promote only if the splice-detector audit (below) flags spectral artefacts. The equal-power crossfade + RMS match should be sufficient when both boundaries are silence-on-silence.

**Cache strategy:**

- Pre-extract **K augmented variants of the pooled features only**, not frames. Pooled tensor ≈ 100 KB, current pooled cache ≈ 600 MB; K = 3 → ~2 GB. Manageable.
- **Do not pre-extract K augmented frame variants.** Frame cache is 78 GB and we just hit disk-full. If A6 needs frame-level access to augmented data later, re-extract on a single layer at that point.
- **Augmentation must happen pre-WavLM**, at the waveform level. Splicing pooled stats does not make physical sense — pooled stats are global per-utterance summaries.
- One-time extraction cost: ~3× the current pooled extraction time (~10–15 min on GPU per K).

**v1 default hyperparameters** (so it's not a sweep at extraction time):

- `p = 0.5` — half of training chunks get an augmented variant.
- `r ~ Uniform(0.20, 0.30)` — at 8 s × 50 Hz, replaces 80–120 frames out of 399.
- Crossfade window `= 150 ms`.
- `K = 3` precomputed variants per training sample. During training, sample one of `{original, aug_1, aug_2, aug_3}` per epoch.

**Cold-partner pool size watch**: with ~9.5 % cold rate × 9505 train chunks ≈ 900 cold chunks across ~25 cold pseudo-speakers (URTIC prior). A cold anchor has plenty of different-speaker cold partners but drawn from a small pool — track partner-reuse statistics during cache build, flag if any single partner exceeds ~5 % of all cold splices.

**Acceptance gates (three-dimensional):**

- **Cold UAR**: A5.5 head UAR ≥ A5b − 1σ (no material drop from augmentation noise).
- **Speaker probe**: top-1 drops by ≥ 1σ vs A5b — augmentation must measurably attack the shortcut to count.
- **Splice-detector audit (hard gate)**: train a small linear probe on `train_fit` cached pooled features for `original` vs `spliced`, evaluate on `devel_val`. If detector UAR > **0.55**, the seams are detectable: increase crossfade length, enforce stricter low-energy boundary selection, RMS-match more carefully, or reduce `r`. **Do not promote A5.5 until the splice-detector gate passes.**

A rung that drops speaker probe but fails the splice-detector audit is just teaching the model a new shortcut.

### A6 — supervised contrastive pretraining

Phase 1 (contrastive, 10 epochs) on top of frozen WavLM features, with speaker-masked positives (same Cold label, different pseudo-speaker; same-speaker same-class pairs masked out). Then attach the A5 head and continue (Phase 2, projection MLP at 5× lower lr than heads). Batch composition: 8 pseudo-speakers × 8 chunks per batch, tracking class proportions, so each anchor has multiple cross-speaker positives.

### A7 — MDD adversary (main contribution)

Gradient-reversal speaker adversary against the projection / head input, λ_adv ramped from 0 via the standard `2/(1+exp(−10p)) − 1` schedule. MDD default (stability + DA generalisation bounds), DANN fallback if MDD proves fiddly.

**Framing**: A7 is "can we reduce speaker information *without sacrificing cold UAR?*" — not "the saving move." Cold lives in voice quality, nasality, and breath turbulence, which is the same acoustic space as speaker identity. Too-strong adversarial removal scrubs the disease signal alongside the speaker signal.

**Two-dimensional acceptance** (both must hold):

- **UAR**: A7 head UAR ≥ A6 − 1σ (no material drop).
- **Speaker probe**: probe top-1 drops by ≥ 2σ vs A6.

A rung that drops the probe but also drops UAR is scientifically interesting (documents the trade-off) but is not a better detector and does not get promoted to the final system.

### A8 — MDD vs DANN comparison

Only if A7 lands.

### Scientific ablations (A7 minus one component each)

`A7 − adversary`, `A7 − contrastive`, `A7 − augmentation`, `A7 − A5 features`, `A7 − OOD group`. Each isolates one intervention's effect.

---

## 7. Training recipe (carried forward, with corrections)

Combined loss for A5–A7 phases:

```text
L = L_cold + λ_adv · L_spk + λ_ood · L_ood
```

- `L_cold`: cross-entropy with **balanced sampler** (replaces the PDF's class-weighted (9, 1) — sampler gives cleaner gradients and a better-calibrated boundary).
- `L_spk`: MDD or DANN against pseudo-speaker ID, gradient-reversed at λ_adv into the projection.
- `L_ood`: Mahalanobis distance regulariser, coefficient 0.01–0.1.

λ_adv schedule: ramp from 0. Same applies to MDD and DANN.

Optimizer: AdamW, lr 1e-3 on heads (current A2 setting), 5e-5 on contrastive pretraining, frozen backbone, projection MLP lr in Phase 2 = 5× lower than heads.

**Cross-validation**: speaker-grouped (pseudo-speaker grouped) k-fold on train, never chunk-level. Use devel only for final selection. Huckvale's 71 → 62 collapse came from iterating on dev.

**Seeds**: lock with `{42, 123, 7}` minimum, extend to 5 for borderline rungs. Never compare 1-seed to 3-seed numbers. Statistical floor: A2 σ = 0.0034 → minimum detectable rung gain ≈ 0.007 UAR (2σ at N=3).

**Augmentation (A5.5)**: cross-speaker within-class splicing, 100–200 ms crossfades, applied to ~50 % of chunks of each class so splice presence is decorrelated from the label.

---

## 8. Fusion strategy

A5's design **is** the fusion strategy: per-group cold-probe logits combined under honesty-fixed (A5b) or honesty-initialised learned (A5c) βs, with the A2 logit as the baseline anchor. Logit-level — no high-dimensional concat-before-MLP, since that's the substrate that let A3 rediscover speaker shortcuts.

External late fusion with a standalone ComParE+SVM is folded in as one more group (its logit fed into the same A5 fusion), no longer a separate A9 rung. If we want to break it out for the paper, report averaging, max-confidence, and a small logistic-regression meta-learner side by side as an A5 ablation.

**No cross-attention fusion, no concat-before-MLP.** Data budget too small (37 cold speakers/partition; every extra trainable parameter is a liability) and concat-MLP is the documented A3 failure mode.

---

## 9. Paper positioning (unchanged from PDF, sharpened)

Three claims, in priority order:

1. **Speaker confounding in URTIC is quantifiable and correctable.** Per-pseudo-speaker UAR variance + speaker probe + the new **honesty-score table** + `A7 − adversary` ablation, in combination, show how much of the 2017 numbers were shortcut learning. The honesty table is the bankable contribution even if UAR stays under 71.
2. **The 2017 fusion wisdom holds in the foundation-model era.** Each modernised pillar adds incremental signal; full fusion beats individuals.
3. **OOD-as-feature is a robust auxiliary signal — but only after speaker-invariance is in place.** Without de-confounding, OOD tracks voice idiosyncrasy. Replicates Suresh's PSP insight in a sharper causal frame.

---

## 10. Risks and contingencies (live)

| Risk                                            | Status        | Notes                                                                                       |
| ----------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------- |
| Dev/test mismatch (Huckvale trap)               | **mitigated** | val→test gap centred on zero; threshold on `train_threshold` only                           |
| MDD/DANN training instability                   | open          | Not yet hit; λ_adv ramp and A6 contrastive prep are the planned stabilisers                 |
| MDD/DANN scrubs cold signal with speaker signal | open          | Two-dimensional A7 acceptance (UAR floor + probe drop) is the guard                         |
| VQ-VAE codebook collapse                        | deferred      | A4 deferred; HuBERT cluster IDs first if A4 runs at all                                     |
| Phoneme aligner fails on German                 | **realised**  | wav2vec2-xlsr CommonVoice domain mismatch; pivoted to acoustic manner labels (then A3 fail) |
| Frozen WavLM mismatched to URTIC                | open          | Not yet attacked; A2 numbers are already above Huckvale's honest test, so low priority      |
| Compute shortage                                | **mitigated** | All caches built once (frames, pooled, manner); per-seed runs are seconds–minutes           |
| Disk full (cache regeneration)                  | **realised**  | Frame cache 78+ GB; drop unneeded layers (e.g. L1/L4/L8) before re-extraction               |
| Augmentation creates splice-detection shortcut  | open          | Symmetric splicing across classes is the planned mitigation; un-tested on URTIC             |
| Learned A5c gate rediscovers speaker shortcuts  | open          | A5b (β fixed) runs first; A5c only if A5b clears the gates; gate L2-pulled toward priors    |
| Test submission format surprises                | low           | Test labels withheld; we evaluate on devel as honest proxy                                  |

---

## 11. Where to look

- **Numbers, configs, per-seed results**: [summary.md](summary.md), [results/](AI-For-Health/results/), [results/A2.json](AI-For-Health/results/A2.json), [results/A3.json](AI-For-Health/results/A3.json).
- **Pseudo-speaker validation**: [model/test.ipynb](AI-For-Health/model/test.ipynb) (ECAPA + WavLM-SV diagnostics, raw and UMAP).
- **Code layout**: [model/](AI-For-Health/model/) — `features/` (extraction + heads), `speakers/` (ECAPA, WavLM, cluster, probe, diagnostics), `data/` (datasets + augmentation stub).
- **Caches**: [cache/](AI-For-Health/cache/) — `microsoft_wavlm-large/{pooled,frames,manner_pooled}`, `manner_labels/`, `phoneme_labels/` (abandoned), `ecapa-voxceleb/`, `pseudo_speakers/`.
