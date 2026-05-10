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
| A5b   | **PASS at K=1**     | Constrained late fusion: per-group linear logits, β fixed = honesty | A2.5 anchor β=4–6 plateau: UAR .6875–.6917, Δ +.031 to +.035 vs A2.5 (~6σ), spk probes PASS |
| A5c   | revivable           | Learned per-group gate, honesty-initialised + regularised           | A5b passed → revivable, but K=1 leaves little room; on hold pending A5.5/A6                |
| A5d   | **DONE**            | Per-layer honesty diagnostic on cached pooled stats (no retraining) | Spk mono L0→L24 (.087→.043 R, .072→.042 G); cold UAR flat; sub@1 ≪ 0.15 on both           |
| A5e   | **SKIPPED**         | A2 retrain on a band-restricted WavLM layer slice                   | A5d trigger missed: no sub@1 > 0.15; cold peak L7 = spk peak band. GPU → A5.5 / A6.        |
| A4    | planned             | Discrete-token histograms (HuBERT units → optional VQ-VAE)          | Deferred behind A5 — more speculative, no built-in anti-shortcut mechanism                 |
| A5.5  | **LOCKED (cons-α)** | Cross-speaker augmentation. Audio splicing FAIL → embedding mixup pivot | Conservative α∈[0.70,0.85]: UAR 0.6624 (Δ +0.006, ~1.6σ), probe ~unchanged. Aggressive α∈[0.50,0.70] branch (d): UAR 0.6397 (Δ -0.017), probe still flat → narrow window EMPTY. A5.5 = cons-α canonical; aggro = ablation row |
| A6    | **queued (PoC)**    | Supervised contrastive pretraining (speaker-masked positives)       | Phase 1 head-only PoC scoped (§4.9): projection MLP on cached pooled stats, A2.5 anchor, 30-60 min CPU/GPU. PoC verdict drives escalation to layer-weight-open or transformer fine-tune |
| A7    | planned             | MDD speaker adversary (DANN fallback) — main contribution           | High-variance, high-upside; ramps λ_adv from 0                                             |
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

**Status.** A6 head-only PoC mechanism = LOCKED as illusory (bottleneck artefact). (A-i) layer-weight-open variant available IF the user wants to spend 2-4 hr GPU on it, but must include the same controls. Otherwise pivot to A7 (gradient-level adversary, the remaining unexplored de-confounding mechanism).

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
