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
| A3    | **REJECTED**        | Manner-aware pooling (pYIN+RMS, 3-cat) two-stream head              | UAR 0.6344 ± 0.0069 (Δ −0.008), probe top-1 +0.005. Both gates failed.                     |
| A5a   | **NEXT**            | Honesty audit over low-dim physiological feature groups             | Compute `label_gain`, `speaker_gain`, ratio + subtractive honesty per group. See § 5.      |
| A5b   | next                | Constrained late fusion: per-group linear logits, β fixed = honesty | No learning at the fusion stage; the table *is* the selection mechanism                    |
| A5c   | next (conditional)  | Learned per-group gate, honesty-initialised + regularised           | Run only if A5b passes or nearly passes the UAR/probe gates                                |
| A4    | planned             | Discrete-token histograms (HuBERT units → optional VQ-VAE)          | Deferred behind A5 — more speculative, no built-in anti-shortcut mechanism                 |
| A5.5  | planned             | Cross-speaker splicing augmentation (symmetric across classes)      | Code stub exists in [data/augmentation.py](AI-For-Health/model/data/augmentation.py)       |
| A6    | planned             | Supervised contrastive pretraining (speaker-masked positives)       | Requires the projection-MLP refactor; pseudo-speakers already cached                       |
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
- Revisit only before A6, where pseudo-speaker labels become *training targets* rather than probe ground truth. Candidate: TitaNet-L or CAM++ (architecturally independent from WavLM).

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

### 5.5 Acceptance gates

Apply at A5b first; A5c only if A5b is within striking distance.

- **UAR**: A5b head UAR ≥ best of {A2, A3} + 0.007 (2σ at N=3).
- **Speaker probe**: probe top-1 on A5b representation ≤ A2 + 1σ (≤ 0.0510).
- **Honesty table**: reported in the paper with per-group `label_gain`, `speaker_gain`, both honesty forms.

### 5.6 v1 scoping decisions (locked)

- **Per-group probe = linear logistic regression**, not an MLP. If a group needs nonlinearity to predict cold, that's a signal the group should be sub-divided.
- **β-learning at A5c uses a `train_fusion` slice** (10 % of `train_fit`, held out from per-group probe training), **not `devel_val`**. Devel stays for early stopping only — same Huckvale discipline as A2's threshold-on-`train_threshold` choice.
- Group seeds from cold-acoustics literature, not generic ComParE families.
- Per-utterance contrast features (e.g. `HNR_voiced − HNR_unvoiced`) tested **only on the handcrafted scalars**, not on WavLM frame means — the latter is the A3 failure mode.
- **Pseudo-speaker centering deferred** to A6 territory. Train clusters are KMeans-on-train; devel clusters are nearest-centroid assignments. Subtracting noisy cluster means risks introducing systematic bias if cluster purity correlates with class.
- Drop quality/reliability metadata — URTIC has no per-chunk SNR or lab flags.
- Stability via bootstrap on `train_fit`, not k-fold (k-fold rebuilds pseudo-speaker KMeans per fold = ~25 min × k).
- No SMOTE / ADASYN — Huckvale showed they don't help; balanced sampler covers the imbalance.

### 5.7 v1 feature checklist (extraction spec)

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

Symmetric across classes (50 % cold + 50 % non-cold) so splice presence is uncorrelated with the label. 100–200 ms crossfades. **No pitch shift, no time-stretch, no speed perturbation** — they can mask or mimic cold-like signatures and break causal interpretability. Apply only after A5 to keep the de-confounding levers separable.

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
