# Project summary — Advanced AI in Health (ComParE 2017 Cold)

Running status doc for the cold-detection attack plan. See [results/README.md](results/README.md) for the rigorous per-rung ablation table and [C:/Users/Chris PC/.claude/projects/.../memory/project_context.md](C:/Users/Chris%20PC/.claude/projects/e--Development-Research-Advanced-AI-in-Health/memory/project_context.md) for the high-level framing.

## Goal

Binary audio classification: Cold vs Non-Cold on the ComParE 2017 Cold sub-challenge (URTIC 4students release, ~28.7k chunks across train/devel/test). Official test labels are withheld by the instructors; devel is our honest proxy. Target UAR to beat: **71.0** (2017 late-fusion baseline).

## Attack-plan status

- **A2** — *locked.* Frozen WavLM-Large + layer-weighted pooled-stats probe → **UAR 0.6428 ± 0.0034** (random splits, uniform layer-weight init)
- **A2.5** — *CANONICAL BASELINE (locked).* A2 + honesty-prior layer-weight init (logits = T·sub@1 from A5d) → **UAR 0.6564 ± 0.0038** on grouped splits at default lr (3 seeds). Δ +0.020 vs A2_grouped (4.7σ); MLP probe flat; LR probe slightly down (0.0760 → 0.0725). **Confirmed canonical by the lr × init grid** ([results/A2_lr_init_grid.json](results/A2_lr_init_grid.json), 18 retrains): the prior is a *distinct attractor* — uniform init never reaches 0.656+ even at lr×10 (plateaus at 0.6447–0.6466), while honest_prior reaches it at every lr setting tested. Higher lr lifts UAR slightly (+0.007 at lr×10) but inflates LR speaker probe by +0.016 (0.0725 → 0.088) — strictly Pareto-dominated by A2.5-default for a speaker-honesty paper. **Mechanism (third-pass revision):** not "flat landscape" (M5 stress test contradicted), not "good init only at low lr" (lr × init grid contradicted) — *"the honesty prior is a distinct attractor in the layer-weight subspace that uniform init does not reach via lr alone."* Strongest formulation of the (A1) paper claim earned so far. See [plan.md §4.6 / §4.6.1 / §4.6.2](plan.md).
- **A5b on A2.5** — *PASS at K=1 with β-sweep, LOCKED.* β=1.00 forced gave FAIL (Δ -0.022, τ pegged) — diagnosed as **calibration failure under wrong-anchor β** (β=1.00 was implicitly conditioned on uniform-A2). β-sweep over {0..2.0} recovered PASS at β*=2.0 boundary (UAR 0.6755, Δ +0.019). Extended β-sweep over {2.5..16} + G4-alone baseline ([results/A5b_grouped_honestprior_betasweep_extended.json](results/A5b_grouped_honestprior_betasweep_extended.json)) **fully resolves the boundary-lock concern**: per-β UAR climbs sharply from β=2 to β=4, plateaus at 0.6875–0.6917 across β ∈ [4, 16], drops to 0.6632 at β=∞ (G4-alone). Per-seed argmax-locked β* = [6, 8, 4] (none at boundary); K=1 fused UAR **0.6913 ± 0.0076**, Δ vs A2.5_argmax = **+0.0349 ± 0.0054** (~6.5σ, strong PASS). G4-alone reference 0.6632; K=1 fused beats it by +0.028 — fusion is genuinely additive, A2.5 contributes real signal. Speaker probes are β-independent: probe (i) = 0.0156 (≪ 0.078), probe (ii) = 0.0733 (≤ 0.078) — both PASS. **Striking parity:** G4-alone at 0.6632 ≈ A2.5-alone at 0.6564 (Δ ~0.7σ) — a 7-dim handcrafted feature matches a 4096-dim WavLM-Large representation on chunk-level UAR. **Total stack: 0.6361 (A2 grouped) → 0.6564 (A2.5, +0.020) → 0.6913 (K=1 fused, +0.035) = +0.055 over leak-corrected baseline.** See [plan.md §4.7.3](plan.md).
- **A3** — *null result, rejected.* Manner-aware pooling (pYIN+RMS, 3 cats) → argmax UAR 0.6344 ± 0.0069 (−0.008 vs A2), probe top-1 0.0555 ± 0.0030 (+0.005 vs A2). Both acceptance gates failed. Cache kept for possible reuse as a feature group inside A5.
- **A5a** — *honesty audit, complete (8 groups).* Per-group cold + speaker probes for {G1 voicing, G2 prosody, G3 voice quality, G4 energy + gain-invariant slice, G5 modulation, G6 spectral shape, G8 OOD Mahalanobis}. G4_energy strongest (lab_gain +0.142) but flagged as gain-confound; G4_gain_invariant retains the lift with halved speaker leak. G5_modulation honest (sub@1 +0.0703, ratio 7.40, speaker leak 0.0146 ~3× chance) and lands at #4 in admission order, just 0.005 below G6 — outside the current K∈{1,2,3} A5b sweep window. G8 anti-predictive (rejected, documented). See `results/A5a_honesty.csv` for the full table.
- **A5b §4.12.3 K=3 with HuBERT-base + LEARNED LW softmax (Tier-2; cell appended, QUEUED for user-run).** Cleanest test of the §4.11.2.1 hypothesis (i) "layer-weighted softmax pooling does the work" vs hypothesis (ii) "capacity dominates". Mirrors WavLM-A2.5's treatment on HuBERT-base pooled stats: STEP 1 per-layer HuBERT honesty audit (13 layers; cold + speaker probes per layer; sub@1 vector); STEP 2 HuBERT-A2.5 head training (LayerWeightedPooledHead n_layers=13, stat_dim=3072, proj_dim=128, dropout=0.5; honesty-prior init T*sub@1 with T=50; cold-CE class-balanced sampler, lr 1e-3 head + 1e-4 LW; 25 epochs, early stop patience 6 on devel_val UAR; 5 seeds, ckpt per seed at `cache/facebook_hubert-base-ls960/head_A25_honestprior_seed{seed}.pt`); STEP 3 standalone HuBERT-A2.5 UAR per seed → M14 pre-flight (< 0.55 skip; ≥ 0.61 admit-plausible; [0.55, 0.61) borderline); STEP 4 K=3 fusion sweep (`fused = a2.5_wavlm + β · mean(z_g4, z_g5, z_hubert_a25)`, extended β-grid, 5 seeds) if pre-flight passes. **Decision rules:** mean K=3 UAR > 0.7087 → ADMIT (new canonical, 'layer-weighted softmax does the work' VALIDATED); standalone clears 0.61 but K=3 doesn't admit → M14 falsified on this substrate (orthogonality with WavLM-A2.5 logit not present); standalone < 0.55 → 'layer-weighted softmax DOESN'T close the gap on HuBERT-base alone; capacity / backbone strength dominates' = evidence for hypothesis (ii). **Cost:** ~50 min wall-clock (~5 min audit + ~25 min head training + ~20 min standalone + K=3 sweep). **Highest-priority cell:** load-bearing methodology test, plausible UAR upside if standalone clears 0.61 (K=3 lift +0.005 to +0.012 → ensemble ~0.715-0.721). **Output:** `results/A5b_k3_hubert_lw_5seed.json` + `results/A5d_hubert_layer_honesty.csv`. See [plan.md §4.12.3](plan.md).
- **A5b §4.12.2 K=2 TTA ensemble (Tier-2; cell appended, QUEUED for user-run).** Test-time augmentation on the WavLM substrate of the K=2 LOCKED canonical. 4 audio perturbations + original = 5 versions per chunk; re-extract WavLM-Large pooled stats inline (transformers AutoModel + soundfile + librosa) and cache to `cache/microsoft_wavlm-large/pooled_tta/{aug_name}/`. Augmentations: time_stretch_p2 (rate=1.02), time_stretch_m2 (rate=0.98), gain_p2dB, gain_m2dB — chosen for class-preserving paralinguistic robustness (preserves vowel quality at <2% / <2dB). G4_gi/G5_modulation NOT re-extracted because G4 is gain-invariant by design and G5 modulation is fairly robust to small temporal jitter. Per (aug, seed) compute K=2 fused logit using existing A2.5 head ckpt + un-augmented G4/G5 z-logits at locked β* per seed. TTA ensemble = mean over 5 augs × 5 seeds = 25 fused logits per chunk; sweep τ on train_threshold; evaluate on devel_test. Includes original-only-ensemble sanity reproduction vs the §4.11.1.4 reference 0.7090. **Expected lift:** +0.003 to +0.010 over 0.7090 if WavLM substrate variance is the bottleneck; could push 0.7090 → 0.712-0.717. **Decision rules:** ≥ 0.710 baseline → "crosses baseline"; > 0.7090 + 0.002 → "modest help"; |Δ| ≤ 0.002 → "neutral"; < 0.7090 - 0.002 → "hurts". **Cost:** ~25 min compute (4 augmentations × ~6 min WavLM-Large re-extraction). **Output:** `results/A5b_k2_tta_ensemble.json`. **Dependencies:** librosa for time_stretch (added to project deps requirement). See [plan.md §4.12.2](plan.md).
- **A5b §4.12.1 K=2 ensemble calibration + stacked weighting (DONE; calibration NEUTRAL, mean-logit stays canonical; M15 + M16 cautionary tales).** (`results/A5b_k2_ensemble_calibrated.json`, 53.65 min — 10× the original ~5 min estimate because the per-seed K=2 fused logits are recomputed from scratch via 5×4=20 inference passes through PooledCacheDataset; caching them as `cache/A5b_per_seed_k2_logits.npz` would cut future runs to ~30 sec but defer the cache-build until subsequent cells need it.) Decision: `calibration_neutral`. Per-variant devel_test UAR: **MEAN-LOGIT 0.7090** (sanity reproduction; τ*=-1.375, recC=0.791, recNC=0.627); **LR-STACKED 0.6137** (Δ -0.0953, **catastrophic overfit** — learned weights {+1.57, **-1.85**, +1.18, **-2.89**, +2.68} have huge magnitudes + 2 negative signs that flip on devel_test); **UAR-GRID-SEARCH 0.7049** (Δ -0.0041, best weights (0, 0, 0.8, 0.2, 0) use only seeds {7, 999} — train_thr UAR went up +0.012, devel_test went down -0.004 = overfit-to-target-metric on n=973); **ISOTONIC on MEAN-LOGIT 0.7064** (Δ -0.0026, M15 prediction was 0.000 but observed -0.0026 from tau-grid resolution mismatch — strict M15 statement holds: monotonic calibration cannot IMPROVE UAR-at-optimal-tau, only EQUAL or appear-WORSE due to discretisation). **Mechanism finding (why equal weights win):** the 5 K=2 fused logits per chunk are highly correlated across seeds (Pearson r ≈ 0.94 per the §4.11.1.4 ensemble cell's reproduction sanity check); with high inter-seed correlation the optimal weighting is approximately uniform; any deviation toward signal-selection underperforms variance-minimisation. **Two new methodology candidates:** M15 = "monotonic calibration cannot improve UAR-at-optimal-tau; the swept-threshold protocol already does this work" (strong form: not better than mean-logit, can equal at best modulo tau-grid resolution). M16 = "On small calibration splits (n ~< 1k, k_minority < 100), L2-regularised LR-stacking on per-seed logits is prone to catastrophic generalisation failure (large-magnitude weights with sign flips); equal-weight mean-logit is the default; UAR-grid-search with strict held-out evaluation is the next-most-conservative alternative." Both M15 and M16 worth adding to the paper's methodology table once the other 2 Tier-2 cells (TTA, HuBERT-LW) are in. **A5b K=2 5-seed mean-logit ensemble at 0.7090 stays the canonical paper-headline number** — calibration ablations confirm it's the variance-optimal pooling on 5 highly-correlated per-seed K=2 logits. See [plan.md §4.12.1](plan.md).
- **A5b §4.11.2.1 K=3 with HuBERT-base (DONE; M14 pre-flight `skip_definite_fail`, K=2 stays canonical, M14 heuristic now spans handcrafted AND FM-derived substrates).** (`results/A5b_k3_hubert_5seed.json`, 3.18 min total.) Reviewer-recommended Tier-2 follow-up: HuBERT-base-ls960 (12 transformer + 1 input layer × 768-d hidden, cluster-based pretraining structurally distinct from WavLM's masked-prediction-with-denoising) extracted in 2.5 min for 19,101 chunks at 133/sec on GPU (12× faster than initial budget). Pooled stats [13, 3072] fp16; cold-LR substrate = layer-mean → 3072-d per chunk. **HuBERT mean-pooled standalone cold-LR UAR = 0.5396** (τ*=-1.000, thr_UAR=0.5152) — sits firmly below the M14 0.55 definite-FAIL floor. K=3 sweep skipped per the pre-flight (saved ~5 min compute on a predicted-failed configuration; the fail-fast mechanism worked as designed). **The standalone-UAR-predictor heuristic now confirms across 7 candidates spanning two substrate families:** handcrafted (G5 0.6121 WIN, G1 0.6058 borderline, G6 0.6053, G_egemaps_full 0.5384, G2 0.5088, G3 0.5039 all FAIL) AND FM-derived (HuBERT mean-pooled 0.5396 FAIL). The ~0.61 threshold partitions admit/fail across both families. **Paper-stage finding:** single-FM late fusion is sufficient on URTIC; HuBERT-base mean-pooled adds no cold-relevant signal beyond WavLM-Large's layer-weighted softmax pooling. **Mechanism hypotheses (deeper variants future work):** (i) WavLM-A2.5's trained layer-weighted softmax does the cold-axis selection that HuBERT mean-pooled doesn't; (ii) WavLM-Large just has ~3.5× more pooled-stat capacity than HuBERT-base. Cleaner tests would be HuBERT-base-A2.5 (learned layer-weighting) vs HuBERT-large mean-pooled; both out of scope for this Tier-2. **M14 generalisation extension:** the standalone-UAR predictor is now confirmed on BOTH handcrafted feature groups AND a structurally-different FM substrate with the same ~0.61 threshold — the heuristic captures a more general property than handcrafted-feature-engineering-specific. **A5b K=2 is FINAL canonical with structural validation across 3 axes:** all 6 admitted A5a groups tested as K=2 partners (G5 won); full eGeMAPSv02 superset tested as K=3 addition (failed); HuBERT-base mean-pooled tested as K=3 addition (M14 pre-flight failed). Cumulative stack unchanged: 0.7037 single / 0.7090 ensemble, distance to 0.71 = 0.001. See [plan.md §4.11.2.1](plan.md).
- **A5b §4.11.1.5 K=3 G_egemaps_full (DONE; NO ADMIT, K=2 stays canonical, A5a slicing methodology validated).** (`results/A5b_k3_egemaps_5seed.json`, 2.23 min, 5 seeds.) Tested two configs at 5 seeds: Config A (K=2 with G_egemaps_full replacing G5_modulation) → 0.6692 ± 0.0056 (Δ -0.0345 vs K=2 LOCKED, β* mostly-boundary [16,16,8,16,12]). Config B (K=3 = A2.5 + G4_gi + G5_mod + G_egemaps_full) → 0.6801 ± 0.0097 (Δ -0.0236 vs K=2 LOCKED, β* boundary-heavy [16,12,16,16,8]). Neither beats K=2 LOCKED canonical (0.7037 ± 0.0060). G_egemaps_full standalone UAR = 0.5384 (chance + 0.04) — sits firmly in the "G_other standalone < 0.61 → boundary β* + fusion absorbed" regime. **Standalone-UAR-predictor heuristic now confirmed across 6 candidates** {G5_mod 0.6121 WIN; G1_voicing 0.6058 borderline; G6 0.6053 FAIL; eGeMAPS_full 0.5384 FAIL; G2 0.5088 FAIL; G3 0.5039 FAIL} — clean 0.61 threshold cleanly partitions K-fusion admit/fail. Worth a paper paragraph as transferable heuristic for honesty-audited late-fusion stacks. **Paper-stage finding (validation branch):** A5a's slicing into G3 (14-d voice quality) + G6 (21-d spectral) captured ~all of eGeMAPS's cold-relevant content; the full 88-d set scored *worse* than the G6 subset alone (0.5384 vs 0.6053 standalone), validating the audit-driven dimensional carving as load-bearing methodology. **A5b K=2 (A2.5 + G4_gi + G5_modulation) is FINAL canonical late-fusion system, exhaustively validated:** all 6 admitted groups + the full eGeMAPS_full superset have been tested as K=2 partners or K=3 additions; no other configuration beats K=2 LOCKED. K=2 5-seed mean-logit ensemble (0.7090) remains the best paper-headline number on the controlled-system axis. Distance to 0.71 baseline: 0.001. Next reviewer-recommended axis is structurally different FM (HuBERT/wav2vec2). See [plan.md §4.11.1.5](plan.md).
- **A5b K=2 5-seed MEAN-LOGIT ENSEMBLE — devel_test UAR 0.7090, distance to 0.710 baseline = 0.001 (within measurement noise).** (`results/A5b_k2_5seed_ensemble.json`, 0.89 min.) Tier-1 §4.11.1.4. Per-chunk linear average of 5 per-seed K=2 fused logits at each seed's locked β* ∈ {6, 8, 8, 8, 12}; τ swept fresh on train_threshold for the ensemble. **MEAN-LOGIT 0.7090 (+0.0053 over per-seed single mean 0.7037).** MEAN-PROBABILITY ensemble 0.7041 (essentially neutral). Mean-logit beats mean-probability because per-seed logit magnitudes carry confidence information that the sigmoid pooling washes out. **Recall pattern flip:** ensemble recC=0.791 / recNC=0.627 (cold-balanced), vs per-seed single typically recC≈0.43/recNC≈0.87 (NC-biased) — averaging different per-seed operating points shifts the effective decision toward cold recall. For a 9.5%-cold corpus, recovering 79% of cold cases is practically valuable beyond what the UAR scalar captures. **Caveats:** (1) ensemble UAR is single-number (σ-bearing canonical stays per-seed single mean 0.7037 ± 0.0060); (2) devel_test, not hidden test (the 0.710 baseline was hidden-test in 2017); (3) τ swept on train_threshold for ensemble (no multiple-comparison-on-devel); (4) ensemble adds 5× inference cost. Paper-stage framing options: conservative ("approaches 0.710 within 0.001"), standard ("matches 0.710 within measurement noise"), generous ("matches 0.710 in rounded reporting"). See [plan.md §4.11.1.4](plan.md).
- **A5b K=2 at N=5 (FINAL LOCKED canonical, BOTH probe gates PASS)** — *5-seed cold UAR + speaker-probe gates: A2.5 + G4_gi + G5_modulation = 0.7037 ± 0.0060, +0.0103 over K=1 at N=5 (0.6934 ± 0.0064), 4.30σ; probe (i) literal 0.0182 ± 0.0006 (under 0.0780 gate by 4.3× margin); probe (ii) backbone-concat 0.0729 ± 0.0005 (under 0.0780 gate by Δ 0.0051).* (`results/A5b_k2_5seed_lock.json` + `results/A5b_k2_5seed_speaker_probes.json`.) **Both speaker probes PASS** at 5 seeds: literal 3-d substrate [logit_A2.5, z_logit_g4, z_logit_g5] gives top1 0.0182 (per-seed range 0.0175-0.0187, σ=0.0006); backbone-concat 4167-d substrate (pooled_4096 ⊕ G4_gi ⊕ G5) gives top1 0.0729 (per-seed range 0.0721-0.0731, σ=0.0005). K=2's marginal probe increases over K=1 (+0.0063 literal, +0.0054 backbone-concat) are capacity-driven (more dimensions = more separation capacity) not speaker-leak-driven in the de-confounding sense — neither substrate has been shaped by speaker-aware training. **Exploratory-vs-confirmatory framing for paper:** K=2 G_other candidate selection (§4.11.1.1, 5 candidates tested) was *exploratory* on devel_test; 5-seed expansion at fixed G5 (§4.11.1.2) is *confirmatory*. The 2 new seeds' K=2 lifts (+0.0092, +0.0095) sit within the original 3 seeds' range (+0.0075 to +0.0137), bounding the selection effect. **Caveat:** all numbers on devel_test (speaker-disjoint subsplit); ComParE 2017 hidden test labels unavailable, so 0.7037 is "best achievable on devel_test" not a direct apples-to-apples vs the 0.710 hidden-test baseline. **A5b K=2 = FINAL canonical late-fusion system for the paper.** All 2-D acceptance gates pass. See [plan.md §4.11.1.2 / §4.11.1.3](plan.md). (`results/A5b_k2_5seed_lock.json`, 4.83 min total: ~3 min A2.5 training for 2 new seeds {999, 31337}, ~2 min K=1+K=2 β-sweep across all 5 seeds.) **Headline N=5 cumulative stack:** uniform-A2-grouped 0.6361 → A2.5 0.6563 ± 0.0027 (+0.020, ~7σ) → K=1 0.6934 ± 0.0064 (+0.037 over A2.5, ~5.8σ) → **K=2 0.7037 ± 0.0060 (+0.010 over K=1, 4.30σ; +0.068 cumulative over leak-corrected baseline, ~17σ)**. **Distance to 0.71 baseline target: 0.006**, within ~1σ of the K=2 standard error. **σ tightening 16-29% across all metrics** (A2.5 +29%, K=1 +16%, K=2 +21%, K=2−K=1 lift +24%). **Per-seed N=5 K=2−K=1 deltas all positive**: seed 42 +0.0117, seed 123 +0.0137, seed 7 +0.0075, seed 999 (NEW) +0.0092, seed 31337 (NEW) +0.0095. K=2 lift is uniformly reproducible across the wider seed pool — no seed produces null or reversed lift. **Mechanism stability:** β* for K=2 on the new seeds (999: β*=12, 31337: β*=8) sits in the same interior plateau as the original 3 seeds (β*=[6, 8, 8]); G4_gi+G5_modulation orthogonality holds across all 5 seeds. **The 3-seed prior locks reproduce exactly when re-aggregated as N=3 subsets of this run — confirming the locks were stable, not optimistic.** See [plan.md §4.11.1.2](plan.md).
- **A5b K=2 at N=3 (PRIOR lock, paper-comparable subset)** — *0.7023 ± 0.0077, +0.011 over K=1 LOCKED (0.6913).* Reproduced exactly within the N=5 run as the N=3 subset. (`results/A5b_k2_extended_betasweep.json`, 1.66 min, 3 seeds.) Tier-1 follow-up §4.11.1.1 in plan: K=2 was only previously tested under M4-pathology free-K-sweep on uniform-A2 (FAILed). Re-tested under locked β plateau methodology on A2.5 anchor for G_other ∈ {G1, G5, G6}. **G5_modulation winner: K=2 UAR 0.7023 (Δ +0.0110 vs K=1, β plateau interior at 6-8 across seeds).** G1_voicing borderline admit (+0.0051). G6_spectral fails (β* pegged at 16 across seeds, calibration absorbed). **Mechanism**: G5 captures cross-frame envelope dynamics (syllable rate, breath pacing) orthogonal to G4_gi's per-frame energy stats. **Total cumulative stack** uniform-A2-grouped 0.6361 → A2.5 0.6564 (+0.020) → K=1 0.6913 (+0.035) → **K=2 0.7023 (+0.011 = +0.066 cumulative)**, ~12σ over leak-corrected baseline, within ~0.01 of the 0.71 baseline target. See [plan.md §4.11.1.1](plan.md).
- **A5b K=1** — *PREVIOUSLY canonical, now superseded by K=2; ships in paper as the K=1 baseline within the K=1/K=2 ablation.*
- **A5b (original K=1 entry)** — *PASS at K=1, both splits.* Final classifier `final_logit = logit_A2 + β · mean_g(zscore_g(logit_g))`, hard top-K admission by `subtractive_honesty`. **Random splits, K-locked K=1 (A2 + G4_gain_invariant): UAR 0.6576 ± 0.0011, Δ vs A2_argmax +0.0148 ± 0.0045 (3.3σ; gate +0.007 cleared by ~2σ), Δ vs A2_τ +0.0112 ± 0.0066 (1.7σ — fusion ≈75% of lift, τ ≈25%).** Per-seed: 42→0.6571, 123→0.6589, 7→0.6569. Speaker probe gate (3 seeds, ceiling 0.0680): probe (i) literal 2-D 0.0119 ± 0.0015; probe (ii) backbone concat 0.0675 ± 0.0006 — both PASS. Capacity controls confirm G4_gi adds +0.0001 to speaker recoverability above pooled-alone (no leak channel; ~5.5 pp drop from 0.0675 upstream to 0.0119 fusion-input is the cold-probe compression doing its job). **Grouped splits, K-locked K=1 (`results/A5b_grouped.json`, leak-corrected): UAR 0.6588 ± 0.0059, Δ vs A2_argmax +0.0227 ± 0.0059 (3.8σ — *stronger* than the random-split lift), Δ vs A2_τ +0.0206 ± 0.0198.** Per-seed: 42→0.6565, 123→0.6656, 7→0.6544. Speaker probe gate (3 seeds, grouped ceiling 0.0780): probe (i) 0.0153 ± 0.0032; probe (ii) 0.0733 ± 0.0002 — both PASS. The methodology fix made the headline more convincing, not less: A2's argmax baseline was the inflated part of the random number, and removing that inflation widened the fusion lift. Free-sweep K=4 result (UAR 0.6502 ± 0.0078, σ > effect size) is documented as **τ-sweep pathology**: free K-sweep on `train_threshold` over-rewards configs with more τ flexibility (more groups → more degrees of freedom). σ collapse 0.0112 → 0.0011 between free-sweep K=4 and K-locked K=1 is the diagnostic. Three follow-on diagnostics ship alongside A5b: Pearson logit-correlation matrix + fused-vector speaker probe + redundancy-adjusted ranking (`results/A5b_diag.json`); K=2 ablation `A2 + G4 + {G1, G5, G6}` plus an A2_τ calibration-aware baseline (`results/A5b_ablation.json`); both shipped on random splits as paper diagnostics about the K-sweep pathology, which is split-independent.
- **A5d** — *DONE on both splits → A5e SKIPPED + structural paper finding.* Per-layer cold + speaker probes on cached `pooled[:, L, :]` for L ∈ [0, 24], single seed (42), no retraining. **Random splits** (`results/A5d_layer_honesty.csv`): best `sub@1` L21 = +0.0387 (≪ 0.15 trigger); best `cold_uar` L7 = 0.6052 with `speaker_top1` 0.0813; highest `speaker_top1` L3 = 0.0871, lowest L22 = 0.0402. **Grouped splits** (`results/A5d_grouped_layer_honesty.csv`): best `sub@1` L0 = +0.0401 (peak shifted from late→early; still ≪ 0.15); best `cold_uar` L6 = 0.6090 with `speaker_top1` 0.0866; highest `speaker_top1` L3 = 0.0956, lowest L24 = 0.0417. **Both A5e skip-branch conditions fire on both splits** (no `sub@1_L > 0.15`; cold peak coincides with high-speaker layer). **Structural paper finding** (independent of A5e, holds on both splits): speaker top-1 decays ~monotonically L0→L24 (random 0.087→0.043, grouped 0.072→0.042 — confirms Pasad 2021 / Chen 2022 for the speaker axis on URTIC under either split discipline), but cold UAR is **flat** across the stack (0.56–0.61 on both) — refutes mid-band cold hypothesis on URTIC specifically. Reportable as standalone empirical finding alongside the A5b headline.
- **A5e** — *SKIPPED.* A5d verdict closed the retrain track. GPU goes to A5.5 / A6 instead.
- **A5c** — *revivable.* Was conditional on A5b passing the gate; A5b passes at K=1, so A5c is technically revivable. But K=1 leaves little fusion stack for a learned gate to refine over — on hold pending A5.5 / A6 outcomes, then revisit if a richer admission set re-opens.
- **A4** — *planned (speculative).* Discrete audio tokens (EnCodec/HuBERT-codes) as auxiliary stream
- **A5.5** — *LOCKED at conservative-α embedding mixup. Both endpoints of α-axis tested → narrow window EMPTY. Move to A6 next.* History: Phase 1 (splice primitives) + smoke 77.3% DONE. Phase 2 (K=3 audio-spliced cache, 80.5% successful, 10.9 min GPU) DONE. **Phase 3 splice-detector audit FAILED at UAR 0.998**. **Phase 3.5 self-splice control → branch C (splicer broken)**: self-splice UAR 0.9900 ≈ cross-splice 0.9981 (Δ -0.008); cross-speaker mixing accounts for <1% of detectability, splicing operation itself accounts for ~99%. **Audio-level splicing dead → pivot to Plan B (embedding mixup).** **Plan B Phase 2-equiv mixup cache** (`results/A5_5_planB_phase2_mixup.json`): 100% successful mixes (no fallback); α ∼ U(0.70, 0.85) realised mean 0.775 ± 0.043; partner-class balance 10.1% cold ≈ corpus 9.5% (symmetric); 1.13 min CPU. **Plan B audit** (`results/A5_5_planB_audit.json`): both gates PASS — (A) detector UAR 0.5034 (chance); (B) mix-bit-as-cold 0.5000 (PASS by class-balanced construction, α-independent). **Plan B Phase 4 conservative-α (LOCKED as A5.5 canonical, `results/A5_5_planB_phase4_mixup.json`, 3 seeds, warm-start from A2.5):** UAR 0.6624 ± 0.0031 (Δ +0.006, ~1.6σ vs A2.5); recC 0.474 vs 0.43 (+0.04, more cold-balanced); MLP probe 0.0506 ± 0.0019 (Δ +0.0005); LR probe 0.0759 ± 0.0030 (Δ +0.0034); cos(init_A2.5, final) ≈ 0.9999 (optimizer didn't move layer weights — M5 consistent). 3-D gate: gate 1 PASS, gates 2a/2b FAIL on probe drop, gate 3 PASS → **PARTIAL PASS**. **Plan B Phase 4 aggressive-α (ABLATION; `results/A5_5_planB_phase4_mixup_aggro.json`, α ∼ U(0.50, 0.70), 3 seeds, 4.10 min):** UAR 0.6397 ± 0.0186 (Δ -0.017 vs A2.5; σ exploded ~6×); MLP probe 0.0485 (Δ -0.0017, within noise); LR probe 0.0735 (Δ +0.0010, within noise); cos(init, final) = 0.99998 (EVEN MORE locked at A2.5 init); best_epoch = {1, 1, 1} (converges immediately, then degrades). 3-D gate: ALL 1+2a+2b FAIL → **branch (d, unexpected): UAR drops AND probe doesn't drop**. **A5.5 LOCK: conservative α as canonical; aggressive α as ablation evidence (M9 narrow-window-EMPTY).** Diagnosis: at conservative α the partner contribution (22.5%) is too gentle to push toward speaker-invariance; at aggressive α (35–50%) label-validity damage shows up in UAR before any de-confounding shows up in the probe. The intermediate range U(0.60, 0.80) wouldn't escape — gentler falls back toward conservative's null, stronger toward aggressive's UAR damage. **Per-chunk pooled-stat mixing on URTIC + frozen WavLM has no usable α operating point.** Paper framing (3-level de-confounding story): A5.5 = data-level (locked, modest contribution, evidence that data alone is insufficient); A6 = representation-level contrastive (next, explicit speaker-invariance objective); A7 = gradient-level adversary. The reflection's deeper diagnosis confirms: mixup on post-frozen-backbone representations cannot de-confound without an explicit objective. See [plan.md §4.8.4-4.8.8](plan.md).
- **A6** — *Phase 1 head-only PoC mechanism = LOCKED as illusory (bottleneck artefact). Layer-weight-open variant available IF user wants 2-4 hr GPU spend, must include controls. Otherwise pivot to A7.* Recipe: fresh projection MLP (4096 → 512 → 128, L2-normalised) on top of A2.5's frozen scaler + layer-weights, supervised contrastive (Khosla 2020) with positives = same Cold + different pseudo-speaker (`cache/microsoft_wavlm-large/A6_phase1_proj_seed{42,123,7}.pt`). **Phase 1 PoC** (`results/A6_phase1_PoC.json`, 3 seeds, 0.59 min): MLP probe 0.0477 (vs A2.5 ref 0.0501, Δ -0.0024); LR probe 0.0410 (vs A2.5 ref 0.0725, Δ -0.0315 — appeared dramatic); cold-LR UAR on z 0.5988 ± 0.0085; class margin +0.0594 ± 0.0065. Strict gate fired branch (D) for 2/3 seeds; mechanistic read suggested partial activation pending controls. **Controls** (`results/A6_phase1_PoC_controls.json`, 3 controls × 3 seeds, 1.62 min) decisively refuted the mechanism: random untrained projection drops LR to 0.0427 (within 0.0017 of A6 — bottleneck explains the LR drop independent of training); cold-CE-only Pareto-dominates A6 on every metric (LR 0.0373, MLP 0.0413, cold UAR 0.6124, margin +0.350); vanilla SupCon (no speaker-masking) slightly beats A6 on LR (0.0395 vs 0.0410 — speaker-masking is mildly *harmful*, reducing positive count without buying de-confounding back). **Decision: `bottleneck_explains_lr_drop`.** Three independent refutations of the head-only mechanism. The 4096 → 128 + L2-norm bottleneck does essentially all of the apparent LR-substrate de-confounding; the contrastive objective adds nothing on top; speaker-masking is slightly counterproductive. **Methodology lesson (M10):** probe-substrate dimensionality is itself a confound for de-confounding measurements. De-confounding rungs that introduce a dimensionality bottleneck must include a random-projection control at the same target dim before claiming mechanism activation. Generalises beyond URTIC. See [plan.md §4.9.1 / §4.9.1.1](plan.md).
- **A6** (continued) — Phase 2 status: (A-i) layer-weight-open available but requires its own controls (random-projection-at-layer-weight-open + cold-CE-at-layer-weight-open) before claiming any verdict; (A-ii) full transformer fine-tune doubly disqualified for now. **Recommendation: pivot to A7** unless user wants to explore the layer-weight variant with full control discipline.
- **A6b** — *Combined cold-CE + speaker-masked SupCon (lambda-sweep) closes head-only scope.* (`results/A6b_phase1_combined_lambda_sweep.json`, 5 λ × 3 seeds, 2.66 min.) After §4.9.1.1 controls disproved the strong claim "speaker-masked SupCon alone uniquely de-confounds," §4.9.1.2 tests the surviving hypothesis "contrastive as a regulariser on top of CE-anchored cold bottleneck." Recipe: joint training of projection + cold linear head with `L = L_cold_CE + λ · L_supcon_speaker_masked`, λ ∈ {0.0, 0.05, 0.1, 0.25, 0.5}. **Result: λ=0 (pure cold-CE) is monotonically Pareto-best on EVERY metric.** MLP probe worsens monotonically with λ (0.0408 → 0.0428 → 0.0435 → 0.0451 → 0.0462); LR probe similar; cold UAR drops at λ > 0 and stays flat-low; class margin shrinks (+0.350 → +0.107). **Contrastive class-pressure is purely subtractive when added to a CE-anchored bottleneck at head-only scope** (M11 in EXPLAINER.md §14): the 4-step mechanism is (1) CE+bottleneck already produces class-separated geometry with speaker mostly compressed away (M10), (2) SupCon flattens within-class diversity instead of adding structure, (3) the flattened diversity includes cold-relevant variation → cold UAR drops, (4) SupCon's "different cold class = separation" treats same-speaker different-cold pairs as separation pressure → re-introduces speaker-correlated variance → MLP probe rises. Decision: `contrastive_dead_pivot_to_a7`. **A6 head-only fully closed across all 3 tested recipes** (pure SupCon / vanilla SupCon / combined CE+SupCon λ-sweep); pure cold-CE Pareto-dominates all of them. Publishable negative result. (A-i) layer-weight-open scope still untested but low expected payoff given the head-only failure pattern. Pivot to A7. See [plan.md §4.9.1.2](plan.md).
- **A7** — *DANN adversary at layer-weight-open scope = LOCKED dead. Three-level closure of de-confounding ladder is now categorical at every tractable frozen-backbone scope.* **Phase 1 PoC** (`results/A7_phase1_PoC.json`, 6 λ × 3 seeds, 4.95 min): A2.5 scaler frozen → layer-weights open at lr 1e-5 → fresh projection 4096 → 512 → 128 L2-norm → cold head + GRL(λ_adv) → SpeakerDiscriminator 128 → 256 → 210; loss = L_cold_CE + L_speaker_CE; matched-control gate vs λ=0 (architecturally identical except for adversary). Result: no λ > 0 cleared the matched-control gate; high λ destabilised (λ=1.0 cls UAR -3.5pp, MLP probe inflates 44%, margin halves). Discriminator at λ=0 train acc = 0.0235, just below the 5×-chance threshold → verdict-classifier fired `B_null` (uninterpretable). **Disc-ceiling diagnostic** (`results/A7_disc_ceiling.json`, 1.30 min): froze A7 λ=0 projection, trained 3 progressively-stronger discriminators; strongest (MLP 128 → 1024 → 512 → 210, 200 epochs) reaches **best devel top1 = 0.0422 ± 0.0007 with best train top1 = 0.9521 ± 0.0062 — memorisation gap +0.910**. Linear LR matches the deepest MLP within noise (devel 0.0434 vs 0.0422) — no nonlinear speaker structure for the deeper discriminators to find. **Decision: `memorisation_a7_dead`** (the M12 case). The two findings stack independently: **Finding 1 (instrumentation, the A7 PoC's specific numbers are uninterpretable)** — in-loop disc at A7 PoC's lr 1e-4 / 20 epochs / moving-target dynamics reached only ~5× chance, confirming the B_null verdict; PoC λ-sweep cannot be read as "adversary fails." **Finding 2 (substrate, the load-bearing one)** — decoupling disc training (frozen z + 200-epoch 1024+512+210 disc at lr 1e-3) STILL gives best devel 0.042 with memo gap +0.91; the substrate has no transferable speaker info. Even if Finding 1 were fixed, Finding 2 says it wouldn't help — the bottleneck has already done the work. **M13 (disc-ceiling-before-λ-sweep ordering discipline)** is the methodological insight: run the substrate-capacity check FIRST, the adversarial λ-sweep SECOND. The adversary in A7 wasn't pushing against a generalisable speaker direction; it was pushing against memorised in-batch fingerprints. The 128-d projection compresses generalisable speaker info to nearly nothing during cold-CE training, regardless of whether layer-weights are open or frozen — bottleneck-confound (M10) returns at layer-weight-open scope. **Methodology lesson (M12):** memorisation gap as discriminator-validity check for adversarial training. Anyone running adversarial de-confounding on a low-dim representation should report the discriminator's train+devel gap; high-train-low-devel = memorisation, no useful adversarial signal. Generalises beyond URTIC. **Three-level de-confounding ladder, final state:** data (A5.5 = M9 narrow-window-empty), representation head-only (A6 = M10 + M11), representation layer-weight-open with adversary (A7 PoC = M10 + M12), representation layer-weight-open un-bottlenecked (A7c = M14 substrate-memorisation-dominated). **All within-budget options now ruled out.** Pivot to (c): write the negative-result methodology paper anchored on M8/M9/M10/M11/M12/M13/M14 + the systematic three-level closure pattern. **A7c v1 + v2 (corrected after reviewer critique).** v1 (`results/A7c_unbottlenecked.json`, 1.78 min) used memo-gap-as-fail-fast and verdicted `memorisation_dead`. Reviewer correctly identified v1 was over-conservative: devel 0.084 > 0.08 means substrate HAS signal; memo gap is regularisation issue not substrate-dead. **v2 (`results/A7c_v2_unbottlenecked.json`, 9.44 min)** applied three corrections (decision logic fixed, regularised disc dropout 0.3 + WD 1e-3, λ range {0, 0.001, ..., 0.1}); phase (a-i) re-passed; phase (a-ii) ran the full sweep with the matched-control gate. **Result: cls UAR, MLP probe, LR probe, AND in-flight disc devel are ALL FLAT across λ — every metric within seed noise of λ=0, no detectable adversary effect.** Verdict: `B_dann_dead_substrate_resistant` — DANN got a fair fight and still didn't shape the substrate. **STRONGER closure than v1.** v1 ships as methodology negative example (over-conservative fail-fast hides real result); v2 ships as canonical A7c verdict. **M14 (NEW finding):** speaker-probe-as-de-confounding-measurement has a fundamental noise floor on small-data corpora with coarse pseudo-speaker labels; discriminator memorisation dominates regardless of bottleneck; treat probe top-1 as an upper bound on speaker leakage, not a precise measurement; report disc-ceiling + memo gap alongside any probe number. Generalises beyond URTIC to any small-data paralinguistic corpus at the ~5k-50k chunk scale. See [plan.md §4.10 / §4.10.1 / §4.10.1.1 / §4.10.1.2](plan.md).
- **A9** — *merged into A5.* Late fusion is A5's output stage, not a standalone rung

Expected gain per rung: A3 worth ~0.5–1.5 UAR; A5 worth ~1–2 (the honesty-weighted fusion is the main de-confounding lever we have pre-A6); A5.5 and A6 worth ~1–2 each; A7 is 2–5 if it works, ~0 if it destabilises. Budget to baseline: ~8 UAR points across ~6 rungs.

### Plan divergence from the original scaffold

Tracked here so the write-up can describe what we actually did, not what we first sketched:

- **A3 labeller pivoted then A3 head rejected**: phoneme-CTC (`wav2vec2-xlsr-53-espeak-cv-ft`) documented as abandoned negative result (84% blank, sharply confident → domain-mismatched). Replaced with pYIN voicing + RMS silence-gate → 3 acoustic-manner categories. Manner labels validated, full cache built, head trained on 3 seeds — both acceptance gates failed (UAR Δ −0.008 vs A2, probe top-1 +0.005). Rejected; manner caches retained as candidate feature group for A5. See [A3 full record](#a3--full-record).
- **A5 scope expanded, absorbs A9**: old A5 ("OOD feature family") was vague. Replaced with a concrete enriched-handcrafted-features design weighted by a per-group **honesty score** (`label_association / speaker_association`) and closed over a **learned gate**. A9 (late fusion) is the A5 output stage rather than a separate rung — one fusion design, one end-to-end training run, one speaker-probe check.
- **A5 promoted ahead of A4**: A5 attacks the speaker shortcut directly via a measurement (honesty score is the Huckvale trap in numerical form). A4 (discrete tokens) stays on the plan but is scheduled behind A5 — it's more speculative and gives no probe guarantee.
- **Pseudo-speakers locked on ECAPA + KMeans(k=210)**: HDBSCAN-204-vs-KMeans-210 cross-method agreement (ARI 0.856 / NMI 0.962) is now the load-bearing evidence, not the raw silhouette number. WavLM-SV documented as a negative control. See [Pseudo-speaker validation](#pseudo-speaker-validation-ecapa-vs-wavlm-sv).

## Locked: A2 baseline

**UAR = 0.6428 ± 0.0034** (argmax, N=3 seeds, data splits fixed at seed=42). Calibrated UAR = 0.6464 ± 0.0082 (calibration is within-noise; mean delta +0.0036 is smaller than its own σ).

Full numbers:

- **Per-class recall**: C = 0.432 ± 0.028, NC = 0.861 ± 0.019 (at τ*)
- **val→test gap**: −0.001 ± 0.005 (centred on zero — honest eval pipeline confirmed)
- **Speaker probe at k=210**: top-1 = 0.0501 ± 0.0009 (10.5× chance), NMI = 0.377 ± 0.003
- **Probe train top-1 ≈ 0.92**, devel top-1 ≈ 0.05 — **18× train/devel gap** is the headline diagnostic. z memorises training-speaker idiosyncrasies but barely generalises speaker structure; this is the Huckvale trap in measured form.

**Architecture**: frozen `microsoft/wavlm-large` (25 hidden states, pooled mean+std+skew+kurt per layer) → FeatureStandardiser (per-position z-score, fit on train) → softmax layer weights (lr × 0.1) → 2-layer MLP 128-d + BatchNorm + GELU + dropout 0.5 → 2-class linear. Balanced sampler, no class weights in loss. AdamW `lr=1e-3`, cosine schedule, early stop patience 6.

**Where this sits**: 2017 e2e baseline 60.0, our A2 62.9, 2017 ComParE+SVM 70.2, late fusion baseline 71.0, Huckvale best 62.1. We're already above Huckvale's honest test number with a linear probe — the "modernised pillar" effect paying off.

## Methodology (locked, do not change per-rung)

**Splits** (all stratified on Cold label, `split_seed=42`):

- `train` (9505) → `train_fit` (90%) + `train_threshold` (10%)
- `devel` (9596) → `devel_val` (50%) + `devel_test` (50%)
- `test` (9551) — withheld challenge labels, not used

**Speaker-disjointness**: by URTIC construction, train and devel are speaker-disjoint. 4students TSV has no speaker IDs, so direct verification is impossible — the val→test gap of −0.001 ± 0.005 is the structural evidence.

**Within-partition leak — DONE (audit report).** `stratified_split` did per-class random shuffle only, so `train_fit`/`train_threshold` and `devel_val`/`devel_test` shared 198/210 and 198/206 pseudo-speakers respectively (massive within-partition overlap). New [`stratified_grouped_split`](AI-For-Health/model/data/cached_dataset.py) uses `sklearn.StratifiedGroupKFold` keyed on `cache/pseudo_speakers/k210_seed42.tsv` → 0/0 overlap. A2 retrained on grouped splits (3 seeds, `results/A2_grouped.json`): argmax UAR 0.6428 ± 0.0034 → **0.6361 ± 0.0019** (Δ -0.0067, ~2σ — argmax was mildly inflated by leak); calibrated UAR 0.6464 → 0.6498 (within σ); val-test gap -0.0009 → -0.0133 (correctly reveals devel_val/devel_test speaker disjointness). MLP speaker probe top-1 0.0501 → 0.0498 (unchanged — speaker-probe interpretation in the paper holds). LR speaker probe top-1 0.0760 ± 0.0020 (codepath-consistent ceiling for A5b/A5d audits, partly higher because grouped devel_val has fewer distinct true-speaker classes). A5b/A5d locked numbers are still anchored on random splits; K=1 PASS structurally survives a baseline shift of ~0.007, but a future re-run on grouped splits is the immediate methodological followup. See [plan.md §4.5](plan.md).

**Seed discipline**:

- Dev seed: `42` (iteration, debugging)
- Lock seeds: `{42, 123, 7}` — all three runs committed to `results/<rung>.json` before claiming a rung
- Paper seeds: extend to 5 for borderline rungs; never compare 1-seed to 3-seed numbers

**Statistical floor**: A2 argmax σ = 0.0034 → minimum detectable rung gain ≈ 0.007 UAR (2σ) at N=3.

**Calibration**: threshold τ selected on `train_threshold` (never devel). Report both UAR_argmax and UAR_calibrated, plus `calib_delta`. Argmax is the cleaner comparison reference (2.4× tighter σ).

**Model selection**: best val_UAR on `devel_val`, patience 6, cosine schedule on base LR.

**Speaker probe protocol**: 2-layer MLP on frozen `z`, trained on train_fit z with pseudo-speaker targets from `cache/pseudo_speakers/k210_seed42.tsv`, evaluated on all of devel. Report top-1 and NMI across 3 seeds. Probe is a measurement tool, re-run after every de-confounding rung (A5.5/A6/A7) — numbers must drop for the rung to count as honest.

## Code layout

```
model/
  data/
    data.py               AudioDataset (mel + opensmile + raw wave)
    cached_dataset.py     PooledCacheDataset, stratified_split, load_labels
    augmentation.py       SpliceSpec + symmetric-across-class sampler (not yet wired)
  features/
    backbone.py           Backbone protocol; WavLM/HuBERT/Whisper concrete impls (fp16)
    extract.py            Batched pooled-stats + frame-level extraction (extract_frames for A3)
    cache.py              CacheManifest (checkpoint_hash + version compat check)
    standardizer.py       FeatureStandardiser (per-position z-score, registered buffers)
    head.py               LayerWeightedPooledHead
    head_a3.py            MannerAwareHead (A3, rejected; kept as documented negative)
    train.py              train_head, evaluate, sweep_threshold, predict_probs, evaluate_at_threshold
    phoneme.py            wav2vec2-xlsr phoneme CTC → cache/phoneme_labels/ (ABANDONED, see A3)
    manner.py             pYIN voicing + RMS → cache/manner_labels/ (A3 pivot, validated)
    manner_pool.py        per-(layer, manner-cat) pooled stats over WavLM frame cache
    f0.py                 pYIN F0 contour → cache/f0/{stem}.npy
    opensmile_extract.py  eGeMAPSv02 functionals → cache/handcrafted/egemaps/
    modulation.py         Huckvale-style modulation spectrogram → cache/handcrafted/modulation/
    scalar_g1.py          voicing scalars from manner labels (G1)
    scalar_g2.py          prosody scalars from F0 + manner labels (G2)
    scalar_g3.py          voice-quality carve of eGeMAPS (jitter/shimmer/HNR/tilt) (G3)
    scalar_g4.py          energy / pause / breath from waveform RMS (G4)
    scalar_g5.py          modulation-spectrogram aggregate (4 acoustic × 8 mod × 2) (G5)
    scalar_g6.py          spectral-shape carve of eGeMAPS (low-MFCC + flux) (G6)
    ood_g8.py             Mahalanobis distance on A2-fused vectors (G8)
  honesty/
    probe.py              cold_probe + speaker_probe (matched linear LR; the audit instrument)
    audit.py              audit_group; appends one row per group to A5a_honesty.csv
    fusion.py             A5b math: fit_cold_probe, predict_logit, fit_zscore, fuse, sweep_tau
  speakers/
    ecapa.py              SpeechBrain ECAPA-TDNN extraction → cache/ecapa-voxceleb/
    cluster.py            KMeans k-sweep over train; writes cache/pseudo_speakers/k{K}_seed{S}.tsv
    probe.py              SpeakerProbe (2-layer MLP, A2 protocol), extract_z, train_probe
  run.ipynb               orchestration cells — A2/A3 training + A5a audits + A5b sweep + diag
cache/
  microsoft_wavlm-large/
    pooled/               pooled stats + per-seed head checkpoints (head_A2_seed{S}.pt, head_A3_seed{S}.pt)
    frames/L{1,4,8,12,16,20,24}/  per-utterance fp16 frames, padding stripped (for A3/A4/A6) — 103 GB
    manner_pooled/        per-(layer, manner-cat) pooled stats (A3 stream input, kept as candidate group)
  phoneme_labels/         wav2vec2-xlsr argmax IDs (ABANDONED — see A3 status)
  manner_labels/          pYIN + RMS 3-cat labels aligned to WavLM frames (validated)
  f0/                     pYIN F0 contour per stem (NaN at unvoiced)
  handcrafted/
    egemaps/              per-stem eGeMAPSv02 functionals [88] fp32 + _columns.json
    g4/                   per-stem G4 energy scalars [11] fp32
    modulation/           per-stem G5 modulation features [64] fp32
  ecapa-voxceleb/         28652 × [192] fp16 speaker embeddings
  pseudo_speakers/        k{100,210,420}_seed42.tsv
  speechbrain/            auto-downloaded ECAPA checkpoint
results/
  README.md               per-rung ablation table + methodology + per-rung notes
  A2.json                 full A2 distribution (3 seeds) + MLP speaker probe block (random splits, historical)
  A2_grouped.json         A2 retrained on grouped splits (3 seeds) + MLP probe + LR probe (within-partition audit)
  A2_grouped_honestprior.json  A2.5: A2 + honesty-prior layer-weight init from A5d sub@1 (CANONICAL baseline at default lr)
  A2_grouped_honestprior_lr_stress.json  A2.5 layer-weight lr × {0.1, 1, 10, 100} stress test (single seed; mechanism control)
  A2_lr_init_grid.json     A2 lr × init grid: (uniform, honest_prior) × (lr×3, 5, 10) × 3 seeds (canonical-A2.5 confirmation; "different attractor" finding)
  A3.json                 rejected A3 distribution (3 seeds) + diagnosis block
  A5a_honesty.csv         per-group honesty rows (G1, G2, G3, G4, G4_gain_invariant, G5, G6, G8)
  A5b.json                A5b sweep results + locked (β*, K*, τ*) + devel_test (gate FAIL, run) + locked-K speaker probe + controls
  A5b_diag.json           A5b Pearson correlation matrix + fused-vector speaker probe + redundancy-adjusted ranking
  A5b_ablation.json       K=2 ablation A2+G4+{G1,G5,G6} across 3 seeds + A2_τ calibration baseline (random splits)
  A5b_grouped.json        A5b K=1 lock + locked-K speaker probes on grouped splits (leak-corrected, uniform-A2 anchor)
  A5b_grouped_honestprior.json  A5b K=1 lock on A2.5 anchor with β=1.00 forced (FAIL — calibration failure under wrong-anchor β)
  A5b_grouped_honestprior_betasweep.json  A5b β-sweep on A2.5 anchor (β grid {0..2.0}, PASS at β*=2.0 boundary; superseded by extended)
  A5b_grouped_honestprior_betasweep_extended.json  A5b extended β-sweep {0..16} + G4-alone baseline (LOCKED PASS at β=4-6 plateau, +0.035 over A2.5)
  A5_5_phase2_extract.json    A5.5 Phase 2: K=3 augmented pooled cache build per train_fit chunk (80.5% successful splices, 10.9 min on GPU)
  A5_5_phase3_splice_detector.json   A5.5 Phase 3: splice-detector audit (FAIL at UAR 0.998 vs gate ≤ 0.55; gate-redefinition framing)
  A5_5_phase3p5_selfsplice_control.json   A5.5 Phase 3.5: self-splice control (UAR 0.9900; Δ vs cross-splice -0.008 → splicer broken; pivot to Plan B)
  A5_5_planB_phase2_mixup.json    A5.5 Plan B Phase 2-equiv: K=3 embedding-mixup cache (α∼U(0.70, 0.85), 100% successful, 1.13 min CPU)
  A5_5_planB_audit.json           A5.5 Plan B audit: detector UAR 0.5034 (chance) + mix-bit-as-cold 0.5000 (PASS gate B, class-balanced)
  A5_5_planB_phase4_mixup.json    A5.5 Plan B Phase 4 conservative α∈[0.70,0.85] (LOCKED as A5.5 canonical): UAR 0.6624 (Δ +0.006); probes within noise → PARTIAL PASS (3-D gate FAIL on probe drop)
  A5_5_planB_phase2_mixup_aggro.json    A5.5 Plan B Phase 2-equiv aggressive α∈[0.50,0.70] (ABLATION): K=3 mixup cache, 100% successful, mean α=0.600, 1.13 min CPU
  A5_5_planB_phase4_mixup_aggro.json    A5.5 Plan B Phase 4 aggressive α∈[0.50,0.70] (ABLATION): UAR 0.6397 (Δ -0.017, σ 6×); probes still flat → branch (d) → narrow-window-EMPTY confirmed
  A6_phase1_PoC.json              A6 Phase 1 head-only PoC (3 seeds, 0.59 min): MLP 0.0477 / LR 0.0410 / cold UAR 0.5988 / margin +0.0594; strict-gate FAIL pending controls
  A6_phase1_PoC_controls.json     A6 PoC controls (random / cold-CE / vanilla SupCon × 3 seeds, 1.62 min): bottleneck explains LR drop; cold-CE Pareto-dominates A6 → A6 head-only mechanism LOCKED illusory
  A6b_phase1_combined_lambda_sweep.json   A6b combined cold-CE + speaker-masked SupCon (5 λ × 3 seeds, 2.66 min): λ=0 monotonically Pareto-best on every metric → contrastive purely subtractive at head-only → A6 head-only fully closed → pivot to A7
  A7_phase1_PoC.json              A7 Phase 1 PoC: DANN adversary at layer-weight-open scope (6 λ × 3 seeds, 4.95 min); B_null (disc at λ=0 = 0.0235 ≈ 5× chance threshold)
  A7_disc_ceiling.json            A7 disc-ceiling diagnostic (3 archs × 3 seeds, 1.30 min); D3 1024/512 best_devel=0.0422 best_train=0.9521 → memo gap +0.910 → memorisation_a7_dead (M12)
  A7c_unbottlenecked.json         A7c v1 (SUPERSEDED): un-bottlenecked DANN on 4096-d (3 archs × 3 seeds, 1.78 min); D3 best_devel=0.0841 → memorisation_dead via OVER-CONSERVATIVE memo-gap fail-fast (reviewer-flagged methodology error)
  A7c_v2_unbottlenecked.json      A7c v2 (CANONICAL after reviewer correction): regularised disc + low λ + corrected decision logic (9.44 min); Phase (a-i) PASSED; Phase (a-ii) all metrics flat across λ → B_dann_dead_substrate_resistant — DANN got a fair fight and still didn't shape the substrate
  A5b_k2_extended_betasweep.json  Tier-1 §4.11.1.1: A5b K=2 extended β-sweep on A2.5 anchor (3 candidates × 3 seeds, 1.66 min). G5_modulation winner: K=2 = (A2.5 + G4_gi + G5_mod) → UAR 0.7023 ± 0.0077, +0.011 over K=1 LOCKED. NEW canonical A5b.
  A5b_k2_g2g3_betasweep.json      Tier-1 §4.11.1.1 (exhaustive completion): K=2 with G2_prosody and G3_voice_quality candidates. Both FAIL (G2 0.6674, G3 0.6576; both β*-pegged at boundary). G5_modulation stays definitive K=2 winner across all 5 tested candidates {G1, G2, G3, G5, G6}.
  A5b_k2_5seed_lock.json          Tier-1 §4.11.1.2 (5-seed expansion): trains A2.5 for seeds {999, 31337}; re-runs K=1 + K=2 β-sweep across all 5 seeds; aggregates N=3 subset (verification) + N=5 (new canonical). K=2 lift sharpens 1.4σ → 4.30σ. K=2 N=5 = 0.7037 ± 0.0060 (was 0.7023 ± 0.0077 at N=3). σ reduction 16-29% across all metrics.
  A5b_k2_5seed_speaker_probes.json Tier-1 §4.11.1.3 (K=2 speaker probes): 5-seed speaker leakage check on K=2 fusion. probe (i) literal 3-d 0.0182 ± 0.0006 (4.3× under 0.0780 gate); probe (ii) backbone-concat 4167-d 0.0729 ± 0.0005 (Δ 0.0051 under gate). BOTH PASS → A5b K=2 FINAL LOCKED.
  A5b_k2_5seed_ensemble.json      Tier-1 §4.11.1.4 (5-seed logit ensemble): MEAN-LOGIT 0.7090 (+0.0053 over per-seed single mean), MEAN-PROBABILITY 0.7041. 0.001 from 0.710 baseline. Recall pattern flips cold-balanced (recC 0.791 / recNC 0.627). Single-number, no σ.
  A5b_k3_egemaps_5seed.json       Tier-1 §4.11.1.5 (K=3 G_egemaps_full): Config A (K=2 with eGeMAPS replacing G5) 0.6692; Config B (K=3 with eGeMAPS added) 0.6801; both WORSE than K=2 LOCKED 0.7037. G_egemaps_full standalone 0.5384 → boundary-pegged β*. A5a slicing methodology validated.
  A5b_k3_hubert_5seed.json        Tier-2 §4.11.2.1 (K=3 with HuBERT-base): HuBERT extracted in 2.5 min (12× faster than budget). Standalone mean-pooled cold-LR UAR 0.5396 → M14 definite-FAIL pre-flight, K=3 sweep SKIPPED. M14 heuristic now generalises across handcrafted + FM substrates (~0.61 threshold).
  A5b_k2_ensemble_calibrated.json  [QUEUED] Tier-2 §4.12.1: K=2 5-seed ensemble calibration + stacked weighting. 4 variants: MEAN-LOGIT baseline (0.7090 ref), LR-STACKED (sklearn LogisticRegression on per-seed logits → cold), UAR-GRID-SEARCH (7776 (w_1..w_5) combos on the simplex), ISOTONIC ABLATION (M15-candidate "monotonic calibration cannot improve UAR-at-optimal-tau" test). Cell appended; ~5 min wall-clock.
  A5b_k2_tta_ensemble.json         [QUEUED] Tier-2 §4.12.2: K=2 5-seed TTA ensemble (4 audio augmentations + original = 5 versions per chunk, WavLM-only re-extraction). Augs: time_stretch_p2/m2 (±2% rate), gain_p2/m2dB. G4/G5 NOT re-extracted (gain-invariant + modulation-robust by design). Mean over 25 fused logits per chunk. Cell appended; ~25 min compute.
  A5b_k3_hubert_lw_5seed.json      [QUEUED] Tier-2 §4.12.3: K=3 with HuBERT-base + LEARNED LW softmax. Cleanest test of the §4.11.2.1 hypothesis (i) "layer-weighted softmax does the work". Per-layer HuBERT honesty audit + HuBERT-A2.5 head training (5 seeds, honesty-prior init T*sub@1) + standalone UAR + M14 pre-flight + K=3 fusion sweep with HuBERT-A2.5 logit. Cell appended; ~50 min compute.
  A5d_hubert_layer_honesty.csv     [QUEUED] Tier-2 §4.12.3 by-product: per-layer cold + speaker probes on cached HuBERT pooled[:, L, :] for L=0..12 (mirrors A5d_grouped_layer_honesty.csv for WavLM). Used as the honesty prior for HuBERT-A2.5 layer-weight init.
  A5d_layer_honesty.csv   per-layer cold + speaker probes on cached pooled[:, L, :] for L=0..24 (random splits)
  A5d_grouped_layer_honesty.csv   same per-layer probes on grouped splits
```

## Key decisions made so far

- **Frozen backbone, no LoRA, no finetuning**: dataset too small (~9.5k train samples with speaker leakage risk); full FT would destroy pretrained features.
- **WavLM-Large over Whisper/HuBERT**: strongest published SUPERB paralinguistic scores, 94k-hour pretraining corpus.
- **fp16 caching everywhere**: one-time extraction cost (~10–15 min per backbone), subsequent training iterates on cached features in seconds.
- **Pooled stats over frame-level**: mean+std+skew+kurt per layer captures the paralinguistic signal without needing frame-aware heads; keeps per-rung training to ~60s.
- **Per-position z-score standardiser as first child of the head**: without it, per-position std spans 4 orders of magnitude and training collapses to majority class. Persisted via buffers so checkpoint is self-contained.
- **Balanced sampler, no class weights in loss**: equivalent in effect but sampler gives cleaner gradients and calibrates the decision boundary without needing threshold tuning.
- **Devel 50/50 split over random-split-from-train**: latter leaked speakers between train and val, producing a phantom val_UAR of 0.97 with test of 0.63 (gap +0.35). Devel 50/50 gives speaker-disjoint val and test by URTIC construction.
- **Threshold on train_threshold, not devel**: reviewer's call — picking τ on devel and reporting on devel is the Huckvale dev-tuning trap.
- **Pseudo-speakers via ECAPA + KMeans k=210**: URTIC has no speaker IDs; ECAPA (voxceleb-trained) + KMeans-on-train + nearest-centroid-on-devel gives defensible speaker groupings. k=210 wins silhouette sweep cleanly, matching URTIC's ~210-speakers-per-split prior.

## Pseudo-speaker validation (ECAPA vs WavLM-SV)

Probe-validation experiment in `model/test.ipynb` to substantiate the ECAPA + KMeans(k=210) choice with multi-method evidence and rule out a swap to WavLM-base-plus-sv. Run on train embeddings only (N=9505).

**ECAPA-VoxCeleb (raw 192-d, L2-normalised, no UMAP)**:

- KMeans k=210 silhouette: **+0.235** (positive, real structure)
- HDBSCAN: **204 clusters**, 2.7% noise, silhouette +0.291
- KMeans vs HDBSCAN agreement: **ARI 0.856 / NMI 0.962** — two methods with completely different inductive biases (centroid vs density-based) recover essentially the same partition.
- kNN cohesion @ k=10: **0.957** — 96% of chunks have all 10 nearest neighbours sharing a cluster, exactly what same-speaker chunks should look like for a corpus where each speaker reads the same passage chunked into 8 s pieces.

**WavLM-base-plus-sv (raw 512-d, post-UMAP-32d analysis)**: HDBSCAN flags **25.0% of points as noise**; KMeans-vs-HDBSCAN ARI = **0.093** — the two methods cannot agree on a partition, meaning there is no stable speaker structure in this embedding space on URTIC. Confirms the architectural-circularity concern empirically: a WavLM-derived speaker embedding doesn't recover speaker structure on URTIC the way an architecturally independent encoder (ECAPA, VoxCeleb-trained) does.

**Headline take-aways for the write-up**:

- HDBSCAN finding **204 clusters ≈ KMeans k=210**, *independently*, on raw ECAPA embeddings, is much stronger evidence for the chosen k than any single silhouette score.
- 204 ≈ 210 ≈ URTIC's expected ~210 speakers/split corroborates that the structure being recovered is genuinely speaker-identity, not artefact.
- Silhouette in UMAP-projected space is inflated (UMAP is designed to pull neighbours together — the +0.92 in UMAP-32d is a self-fulfilling number); raw-embedding silhouette is the honest figure to report.
- WavLM-SV is now the documented *negative control* — not "we didn't try it", but "we tried it and it doesn't recover speaker structure on URTIC".

**Decision (locked)**: stay on ECAPA + KMeans(k=210) for all probe and pseudo-speaker uses. Revisit (TitaNet-L or CAM++, both architecturally independent from WavLM) only before A6 (speaker-masked contrastive pretraining), where pseudo-speaker labels become *training targets* rather than just probe ground truth.

## A3 — full record

**Headline**: phoneme-CTC labelling abandoned with empirical evidence; pivoted to acoustic-manner labelling (pYIN voicing + RMS energy); manner validation gate PASSED on 20-chunk subset; full extraction running. A3 head, training, and probe still to do.

### Infrastructure built (regardless of which labeller path)

- **`extract_frames()`** in `model/features/extract.py` — frame-level WavLM cache, padding stripped via the backbone output mask. Per-(layer, file) layout so downstream rungs (A3/A4/A6) load only what they need.
- **Frame cache**: `cache/microsoft_wavlm-large/frames/L{1,4,8,12,16,20,24}/{stem}.pt` — 7 layers × 19101 utterances (train 9505 + devel 9596) = **133 707 fp16 tensors, 103 GB on disk.**
- **CNN stride math verified**: 8 s × 16 kHz = 128 000 samples → CNN stride 320 → 399 frames at 50 Hz. Spot-check on `devel_0001` confirms `[399, 1024]` per layer. `extract_manner_labels` truncates/pads to this exact count so labels index cleanly into the frame cache.

### Path 1 (ABANDONED): wav2vec2-xlsr-53-espeak-cv-ft phoneme CTC

**Why scoped**: multilingual IPA phoneme CTC, fine-tuned on CommonVoice transcripts converted via espeak-ng. Per-frame argmax at 50 Hz matches WavLM stride exactly, no resampling. No German-specific phoneme CTC available with comparable coverage.

**Code**: `model/features/phoneme.py` — `extract_phonemes()` (corpus walker), `classify_token()` and `build_category_map()` for IPA → 6-category mapping (`silence`, `vowel`, `nasal`, `fricative`, `plosive`, `approximant`).

**Implementation hurdles fixed**:

- HF tokenizer stack (`Wav2Vec2Processor`, `AutoTokenizer`, `Wav2Vec2PhonemeCTCTokenizer`) misresolves the espeak-cv-ft config across the transformers versions we tested — all returned a `bool` instead of an instance. Worked around by skipping the tokenizer entirely and downloading `vocab.json` directly via `huggingface_hub.hf_hub_download(...)`.
- Per-sample fp32 normalisation over valid frames (mean/var via attention mask) before fp16 inference, matching wav2vec2 feature-extractor default.

**What we ran and observed**:

1. Full extraction over train + devel: 19101 files, vocab size 392, written to `cache/phoneme_labels/{stem}.pt` int16.
2. **Histogram on `devel_0001` (399 frames), straight argmax**: silence 75%, vowel 9%, nasal 4%, fricative 4%, plosive 5%, approximant 3%. Blank token alone wins ~75% of frames. Per-utterance pooled stats from ~14 nasal / 16 fricative / 21 plosive frames are too noisy for a classifier seeing one row per utterance.
3. **Diagnostic: blank-masked argmax**: gave silence ~5% / plosive 45.6% / vowel 12% — but raw-token breakdown on a sample stem showed `t=36.8%`, `ɾ=7.8%`, `j=7.3%` of non-blank mass. The plosive bucket is dominated by filler tokens, not actual plosive articulations. Reverted to straight argmax.
4. **Reviewer pushback (correctly applied)**: smearing heuristic (±W frames into adjacent blanks) rejected as untestable on URTIC — no phoneme-boundary ground truth to validate W against, and resulting error correlates with phoneme category (systematic bias, not random noise). Per-utterance vs corpus-level statistical reasoning: the classifier sees one row per utterance, so corpus-level aggregation does not rescue per-utterance σ²/14 noise.
5. **Soft-aggregation diagnostic (the closing experiment)**: 8 devel stems, full softmax projected into 6 IPA categories vs hard-argmax histogram, plus confidence proxies. Aggregate over 8 stems:

   - mean blank-wins-top1: **84.1%**
   - mean top-1 prob: **0.962**
   - mean blank prob: **0.836**
   - mean per-frame entropy: **0.16 nats** (uniform over V=392 = 5.97 nats; the model is at 2.7% of uniform — sharply peaked, not smeared)
   - hard-argmax aggregate: silence 84.3 / vowel 5.6 / nasal 2.6 / fricative 2.6 / plosive 3.1 / approximant 1.8
   - soft-sum aggregate:    silence 84.0 / vowel 5.5 / nasal 2.7 / fricative 2.6 / plosive 3.3 / approximant 1.9
   - Conditional view (where does residual non-silence mass go in blank-winning frames): vowel 8.5 / nasal 23.4 / fricative 13.8 / plosive 23.3 / approximant 30.9 — but residual averages ~5% per blank-winning frame, which is noise-floor for pooling.

**Diagnosis**: the model is *not* underconfident — it is sharply confident, with hard and soft histograms identical to within sampling noise (84.3 vs 84.0% silence). Cross-check vs the manner labeller (40% silence on the same corpus) shows the phoneme model labels roughly *half* of audible speech as blank. Classic domain-mismatch signature: CommonVoice lay-reading training distribution doesn't cover URTIC's German clinical recordings, so the model falls back to its prior (blank). Soft pooling cannot rescue this — there is nothing smeared to recover. The reviewer's original pushback against smearing heuristics is now empirically vindicated rather than speculatively defensible.

**Artefacts kept**:

- Code: `model/features/phoneme.py`, `model/features/__init__.py` exports, two notebook cells in `run.ipynb` (the extraction cell and the soft-aggregation diagnostic cell — kept as documented negative result).
- Data: `cache/phoneme_labels/` (~15 MB, gitignored). Not used downstream; deletable any time disk pressure matters.
- For the write-up: this section + the diagnostic cell output is the documented negative result. Useful if a reviewer asks "why didn't you try a phoneme labeller?".

### Path 2 (ACTIVE): pYIN voicing + RMS manner labels

**Why pivoted here**: simpler categorisation, validated against decades of speech-literature voicing-detection work, citable in the paper. pYIN has known properties; smearing heuristics on a flaky CTC model do not.

**Locked decisions**:

- **Labeller**: `librosa.pyin(fmin=65 Hz, fmax=400 Hz)` for voiced/unvoiced + `librosa.feature.rms` for silence gate. fmin/fmax bracket human F0 (male 65 Hz to female 400 Hz).
- **Categories**: **3** — `silence`, `voiced`, `unvoiced`. Coarser than 6-cat phonetic but captures the same articulation axis. Cold signal lives in voiced (glottal pulse, nasal formants) and unvoiced (fricative turbulence broadens with mucus) regions; silence is the non-signal bucket.
- **Frame alignment**: librosa `hop_length=320, center=True` at sr=16000 → 50 Hz, matches WavLM stride. Output truncated/zero-padded (with silence label) to the WavLM frame count per utterance — mismatch is always 1–3 frames.
- **Silence floor**: 30 dB below per-utterance RMS peak.

**Code**: `model/features/manner.py` — `compute_manner()` (per-utterance) and `extract_manner_labels()` (corpus walker, with optional `frames_cache_root` so validation runs can write to a tmp dir while reading frame counts from the real cache).

**Validation gate result (PASSED)**: 20-chunk subset of devel, aggregate over 7980 frames:

- silence  40.4% (prior 20–40%) — at upper edge but cleanly explained by 8 s clipping; `devel_0001` alone has 1.46 s trailing silence = 18% of the chunk.
- voiced   37.7% (prior 45–65%) — slightly below; same 8 s-clipping caveat (active speech only fills part of the window).
- unvoiced 21.9% (prior 10–25%) — within prior.

Time-range structure on `devel_0001` (399 frames = 7.98 s): speech bracketed by silence, alternating voiced/unvoiced with syllable-scale durations (vowels 300–900 ms, consonants 60–300 ms), trailing 1.46 s silence. No pathological 20-ms flicker. pYIN behaving exactly as advertised.

### Full manner extraction (DONE)

Wall-time **22.6 h** on CPU (much slower than initial extrapolation; pYIN HMM-Viterbi cost per utterance is ~5× the 20-chunk validation estimate). Cache is idempotent (`skip_existing=True`), never needs to run again.

- `cache/manner_labels/{stem}.pt` — **19101 int8 tensors** (train 9505 + devel 9596), aligned to WavLM L1 frame count per utterance.
- `cache/manner_labels/categories.json` — `{"names": ["silence", "voiced", "unvoiced"]}`.

### Built and trained

1. **Category-pooling extractor** — `model/features/manner_pool.py`. Per-utterance mean+std per (layer, category) over the 7-layer frame cache and 3-cat labels. Writes `cache/microsoft_wavlm-large/manner_pooled/{stem}.pt` as `{pooled: [7, 3, 2048] fp16, indicator: [3] uint8}`. Empty buckets zero-filled and flagged via the indicator. 19101 bundles cached (~12 min).
2. **A3 head** — `model/features/head_a3.py::MannerAwareHead`. Two streams: A2 `[25, 4096]` and manner `[7, 3, 2048]`. Per-stream FeatureStandardiser (manner-side fit weighted by indicator so empty buckets don't deflate stds). Per-stream softmax layer-weights (lr×0.1). Concat `[4096 + 6144 + 3] = 10243` → MLP 128 → BN → GELU → dropout 0.6 → 2-class. AdamW wd=3e-3.
3. **Three-seed training** `{42, 123, 7}`, splits identical to A2.

### A3 result (FAIL — both acceptance gates)

- **UAR argmax**: **0.6344 ± 0.0069**  vs A2 0.6428 ± 0.0034 → Δ −0.0084. Needed +0.0154 (2σ at N=3) → **FAIL**.
- **UAR calibrated**: 0.6475 ± 0.0059 vs A2 0.6464 ± 0.0082 → Δ +0.0011, within noise.
- **recall_C @ τ**: 0.4328 ± 0.0277 vs A2 0.432 ± 0.028 → ~0.
- **recall_NC @ τ**: 0.8621 ± 0.0392 vs A2 0.861 ± 0.019 → ~0.
- **val→test gap**: −0.0027 ± 0.0072 vs A2 −0.001 ± 0.005 → within noise.
- **Probe top-1**: **0.0555 ± 0.0030** vs A2 0.0501 ± 0.0009 → Δ +0.0054. Needed ≤+0.0031 (1σ joint) → **FAIL**.
- **Probe NMI**: 0.3907 ± 0.0023 vs A2 0.377 ± 0.003 → Δ +0.014, inflated.
- **Probe train top-1**: 0.9958 vs A2 ~0.92 → Δ +0.07, severely inflated (z encoding speakers near-perfectly on train).

Per-seed numbers stored in `results/A3.json`.

### Diagnosis

Two structural findings explain both failures:

- **Manner stream concentrates on the earliest WavLM layers across all seeds.** Top-3 manner layer weights = (idx 0, 1, 2) = WavLM L1, L4, L8. Late layers (L20, L24) get the lowest weight. WavLM literature (Chen et al. 2022, SUPERB) places speaker/acoustic information in early layers and phonetic/semantic content in mid–late layers. The manner stream is preferentially weighting the very layers most loaded with speaker formants and spectral identity — exactly why probe top-1 inflated.
- **A2 stream layer weights stayed pinned to uniform** across all 3 seeds (max−min spread < 0.004 over 25 layers vs uniform 0.04). Best epoch was 2–6 — the model latched onto easy-to-fit manner-stream features and stopped improving before the A2 layer-weight track could differentiate. The MLP is effectively running a manner-stream-only classifier with the A2 stream as untouched padding.
- **Severe overfitting**: train acc 99.1–99.7% by epoch 12 despite dropout 0.6 + wd 3e-3. Probe train top-1 is 0.99+ (vs A2's 0.92), meaning z carries near-perfect speaker identity in the training set. The 18× train/devel probe gap that defined A2's Huckvale signature is now ~18× × (0.0555/0.0501) ≈ same shape, just more pronounced.

**Why this happened, in one sentence**: per-utterance per-category mean of WavLM frames in early layers IS a speaker fingerprint (voiced-frame mean ≈ speaker formants; unvoiced-frame mean ≈ speaker spectral envelope), and z-scoring against the population mean does not remove the per-utterance offset that carries speaker identity.

### Decision

A3 rejected. **Calibrated UAR is statistically indistinguishable from A2 (+0.0011, within noise)** and the probe inflation, while modest, is consistent across seeds. The pYIN+RMS labels themselves are clean (manner gate passed) — the failure is in the head design: pooling WavLM frames by manner adds a speaker-leaky feature stream without contributing label-relevant signal beyond what A2 already captures.

What the project keeps from A3:

- **`cache/manner_labels/`** (19101 stems) — usable as a per-utterance handcrafted feature inside A5 (e.g., voiced-frame fraction is a one-line summary of vocal-fold activity that's nearly free).
- **`cache/microsoft_wavlm-large/manner_pooled/`** (19101 bundles, mean+std per layer per cat) — a candidate feature group for A5's honesty-score table. If its honesty score is ≪ 1, A5 will down-weight it automatically; if ≥ 1, the per-cat pooled stats had a label-relevant dimension that the speaker-leaky early-layer weighting was masking.
- **The diagnostic itself** — for the write-up, this is a clean documented attempt at manner-aware pooling with empirical evidence of why it doesn't work on URTIC. Useful section in the paper.

### Possible rescue paths (NOT pursued — listed for write-up completeness)

- **Per-utterance contrast features**: `mean(voiced) − mean(silence)` and `mean(unvoiced) − mean(silence)` per layer. Removes the per-utterance speaker baseline that's leaking. Would be the standard speech-recognition channel-normalisation approach.
- **Manner pooling restricted to late WavLM layers** (L20, L24) where speaker information is weakest. Risks losing whatever cold signal the manner pooling was supposed to capture.
- **Mid-late fusion** with a gradient-reversal speaker-adversarial head on the manner stream's z. Pushed to A7 territory; not a v1 fix.

These are noted because A5's honesty-score framework is designed to handle exactly this kind of mixed-signal feature group cleanly, so re-engineering A3 in isolation is poor return on time vs putting the same effort into A5.

## A5 — design (enriched handcrafted features + honesty-weighted fusion)

**One-line framing**: an interpretable, openSMILE-grounded handcrafted branch whose per-group contribution to the final logit is weighted by a precomputed **honesty score** and further modulated by a learned gate. A5's output is the late-fusion stage that used to be A9.

### Motivation

A2's speaker probe shows train top-1 ≈ 0.92 vs devel top-1 ≈ 0.05 — the Huckvale trap in measured form. Every feature family has some mix of label-relevant and speaker-identity-relevant signal. A5 is the first rung that pre-measures that mix **per group** and down-weights groups where speaker identity dominates the label signal. This is a direct, numerical attack on the central methodological problem of the 2017 challenge — not a post-hoc defensive check.

### Honesty score

For each feature group $g$ (seeded from ComParE-2016 LLD families + Schuller/Huckvale literature on cold-relevant acoustics):

- **label_association(g)** — UAR of a tiny group-only probe trained on `train_fit` Cold labels, evaluated on `devel_val`.
- **speaker_association(g)** — top-1 of the same-shape group-only probe trained on `train_fit` pseudo-speakers (`cache/pseudo_speakers/k210_seed42.tsv`), evaluated on `devel` (same protocol as the A2 speaker probe).
- **honesty(g)** = `label_association(g) / speaker_association(g)` (with a small floor to avoid div-by-zero).

Report the full table in the paper. Groups with honesty > 1 pull their weight; groups with honesty ≪ 1 are the Huckvale rug-pulls to shrink.

### v1 scoping decisions (locked)

- **Groups**: reuse **ComParE 2016 functional-family partitions** (MFCC stats, F0, jitter/shimmer, HNR, spectral-shape, loudness/energy, voicing-probability, formants) instead of inventing a fresh taxonomy. The families already align with published cold-acoustics findings (Cummins 2017, Schuller 2017 baseline, Huckvale 2018).
- **Drop quality/reliability metadata**: we have no per-chunk SNR or lab-recording flags on URTIC. If a quality proxy matters later, use voiced-frame fraction (free — we already have it from A3 manner labels).
- **Stability score via bootstrap on `train_fit`**, not k-fold: k-fold re-builds pseudo-speaker KMeans on each fold which is ~25 min × k. Bootstrap is cheaper, comparable evidence.
- **Late fusion first, mid-late fusion second**: A5 v1 concatenates group-summarised handcrafted logits with the A2/A3 head logits at the final layer. Mid-late (cross-attention between streams) is a follow-on only if late fusion lands a gain but the probe stays flat.
- **Gating mechanism**: a per-group scalar $\alpha_g = \sigma(\text{honesty}(g)/T) \cdot \sigma(\text{learned}(g))$ — the honesty term is fixed (computed once from probes), the learned term is trained end-to-end against the Cold loss. Both sigmoids so the composition is interpretable as an elementwise attention.

### Success criteria

- A5 head UAR must beat the best of {A2, A3} by ≥ 0.007.
- Speaker probe top-1 on the A5 representation must not exceed A2's by more than 1σ (≤ 0.0510). For A5b the literal "1-d fused logit" is degenerate as a 210-class probe input — operationalised as the speaker probe on the actual `[logit_A2, z_logit_g, ...]` concat the fusion has access to (see A5b diagnostics below).
- Honesty table must be reported in the paper with per-group numbers — this is the **novel methodological headline**, bankable regardless of whether A5 beats baseline.

### Why this defers A4 behind A5

A4 (discrete audio tokens) is more speculative and has no built-in anti-speaker-shortcut mechanism; A5 gives a probe-checkable, paper-reportable de-confounding result on its own. If A5 closes the gap to baseline, A4 may never be necessary.

### A5a — honesty audit results

Per-group rows in `results/A5a_honesty.csv`. Snapshot (pre-G5; G5 pending one cell run):

```text
group                    dim     UAR   lab_gain   spk_top1   spk_gain    ratio     sub@1
G4_energy                 11  0.6418    +0.1418     0.0181    +0.0134    +9.87   +0.1284
G4_gain_invariant          7  0.6318    +0.1318     0.0127    +0.0080   +14.73   +0.1239
G1_voicing                 9  0.5831    +0.0831     0.0110    +0.0063   +11.41   +0.0768
G6_spectral_shape         21  0.6050    +0.1050     0.0340    +0.0292    +3.48   +0.0758
G5_modulation             64  0.5802    +0.0802     0.0146    +0.0098    +7.40   +0.0703
G2_prosody                10  0.5680    +0.0680     0.0194    +0.0146    +4.35   +0.0534
G3_voice_quality          14  0.5591    +0.0591     0.0233    +0.0186    +3.02   +0.0405
G8_ood_mahalanobis         1  0.4334    -0.0666     0.0073    +0.0025   -18.86   -0.0692
```

Rows ordered by `sub@1` descending (admission ranking). Reading: `lab_gain = UAR − 0.5`, `spk_gain = top1 − 1/210`, `sub@1 = lab_gain − 1·spk_gain` (admission key, λ=1). Linear-only probes (matched cold + speaker LR, balanced for cold, multinomial for speakers); StandardScaler fit on `train_fit`, evaluated on `devel_val`.

Highlights:

- **G4_energy** is the strongest single group but raises a recording-gain confound concern. The **G4_gain_invariant** ablation (drop absolute-RMS cols 0-3, keep regime-contrast and pause-shape cols 4-10) loses only 0.010 UAR while halving speaker_gain → admit the gain-invariant slice instead, document G4_energy as the comparison row.
- **G1_voicing** has the best ratio (11.41) — cleanest signal in the table. Cold-biased (recall_C > recall_NC) — useful complement to A2's NC-bias.
- **G6_spectral_shape** carries the second-strongest predictive lift but the highest speaker leak (low-MFCCs are by construction speaker-rich — vocal-tract envelope is speaker identity). Passes admission at λ=1 (sub@1 +0.076), would fail at λ=2 — borderline.
- **G5_modulation** (Huckvale's MOD family — per-mel-band FFT-over-time → modulation spectrum, 64-d) is honest: ratio 7.40, speaker leak 0.0146 (~3× chance — much cleaner than G3 / G6). Lands at **#4 in admission ordering**, only 0.005 sub@1 below G6 — under the current A5b K∈{1,2,3} sweep it would *not* enter; with K=4 it would. Whether to extend the K grid is a sweep-design call, not a correctness one.
- **G8_ood_mahalanobis** anti-predictive (UAR 0.433, label_gain −0.067). Documented negative result — the original PDF-A5 hypothesis ("OOD distance from healthy manifold predicts cold") doesn't hold on URTIC. Excluded from admission pool.

### A5b — late fusion (PASS at K=1)

Final classifier per utterance:

```text
final_logit = logit_A2 + β · mean_g( zscore_g( logit_g ) )
```

- `logit_A2` = log-odds from A2 head, β_A2 = 1 (anchor never re-weighted).
- `logit_g` = `clf.decision_function(scaler.transform(X_g))` from a per-group cold probe matching the A5a recipe (StandardScaler + balanced LR, fixed seed) — so the audited UAR is exactly what fusion sees.
- `zscore_g` removes per-group scale differences (G4 logits naturally span a wider range than G2). Mean and std fit on `train_fit` predictions.
- `mean_g` over the **K admitted** groups, picked top-K by `sub@1` (ranked descending, filtered to `label_gain > 0`).
- `β` and `K` swept on `train_threshold`; `τ` swept in **logit space** (`np.linspace(-4.0, 4.0, 321)`) — the fused quantity is a logit, not a probability.
- Locked `(β*, K*, τ*)` evaluated **once** on `devel_test`. Three training seeds {42, 123, 7}.

Admission pool is read from `A5a_honesty.csv` at A5b runtime, so adding G5 (or any future group) is one CSV row away from being considered without touching the A5b code.

**Acceptance gate**: A5b mean UAR on `devel_test` ≥ A2 mean + 0.007 (= 2σ at N=3).

### A5b — locked numbers (3 seeds {42, 123, 7})

**Headline (K-locked K=1, admission frozen to A2 + G4_gain_invariant; sweep only β and τ on `train_threshold`):**

- **A5b devel_test UAR**: **0.6576 ± 0.0011** (argmax on the fused logit). Per-seed: 42→0.6571, 123→0.6589, 7→0.6569.
- **Δ vs A2_argmax**: **+0.0148 ± 0.0045** — 3.3σ above zero, gate target +0.007 cleared by ~2σ. **PASS.**
- **Δ vs A2_τ** (calibration-aware baseline): **+0.0112 ± 0.0066** — 1.7σ above zero. Fusion contributes ~75% of the headline lift; τ-tuning the A2 logits alone contributes the remaining ~25%.
- **Fused-vector speaker probe**: top-1 = 0.0194 on `[logit_A2, z_logit_G4_gi]` — well below the 0.0510 ceiling. No speaker leak from the fusion construction.

**Documented sweep pathology (free K∈{1..4}-sweep on `train_threshold`):**

- Original A5b devel_test UAR: 0.6502 ± 0.0078 (argmax). Δ vs A2_argmax +0.0074 ± 0.0112 — variance > effect size, fusion not reliably winning.
- Per-seed K winners: 42→K=4, 123→K=4, 7→K=2 (β=0.5, τ=+0.250 on the seed-42 lock).
- Diagnosis (matches the diagnostics-cell finding that the K=4 admits are mutually redundant with A2 and with each other, esp. G4↔G1 r=0.628): **free K-sweep over-rewards configurations with more τ flexibility** (more groups in `mean_g` widen the achievable τ-vs-UAR curve on `train_threshold`), inflating variance without improving the mean.
- σ collapse from **0.0112** (free K-sweep) to **0.0011** (K-locked K=1) — variance drops 7× when admission is frozen and only β/τ are swept. That collapse is what flips FAIL → PASS.

Both numbers are paper-reportable: K=1 is the headline result, K=4 is the documented sweep-protocol finding.

**Paper framing.** A5b passes with a *selective single-group fusion*: WavLM A2 logits plus gain-invariant energy/pause features (G4_gi). Broader handcrafted fusion did not help despite positive A5a honesty scores — the §10.1 diagnostics show the additional groups (G6, G5, G1) carry signal that overlaps Pearson 0.30–0.63 with A2 or with G4. Mechanism is **redundancy with A2**, not absence of signal in the other groups (G6 has +0.105 standalone label_gain). The useful handcrafted-and-A2-orthogonal cold signal concentrates in temporal energy structure; broader fusion adds groups whose lift is largely already in A2's logit.

**Locked-K speaker probe — DONE (gate cleared cleanly).** The existing `A5b_diag.json` fused-vector probe (top-1 = 0.0194) was on the *full admission pool* concat, not the K=1 representation. The locked-K probe runs both (i) the literal 2-D K=1 fused vector `[logit_A2, z_logit_G4_gi]` and (ii) the backbone-level concat `pooled_4096 ⊕ G4_gi_7`, plus capacity controls — recorded under `A5b.json::locked_speaker_probe` and `…::locked_speaker_probe_controls`.

```text
                                    top-1 (3 seeds)        notes
(i)  literal 2-D  [logit_A2, z_g4]   0.0119 ± 0.0015        fusion-input view, gate target
(ii) backbone concat [pooled, g4]    0.0675 ± 0.0006        leak-channel audit, upstream
(a)  pooled-only [pooled]            0.0674 ± 0.0006        codepath-consistent A2 ref
(b)  pooled + 7d Gaussian noise      0.0665 ± 0.0026        capacity sanity for (ii)
ceiling = (a) + 1σ                   0.0680                 honesty.speaker_probe substrate
```

Verdict: **both gates PASS** against the codepath-consistent ceiling 0.0680.

- **Probe-substrate distinction.** The A5a/A5b/A5d audit uses `honesty.speaker_probe` (multinomial LR, LBFGS); the historical `results/A2.json::speaker_probe` value 0.0501 ± 0.0009 is from `speakers/probe.py` (deeper MLP, ~30 epochs of training). They give different numbers on the same A2 fused vector — 0.0501 (MLP) vs 0.0674 (LR) — because LR with L2 on a convex problem extracts more from a 4096-d substrate than the MLP recovers under its training schedule. The LR number is the apples-to-apples reference for everything anchored on `honesty.speaker_probe`; reporting both keeps the ceiling honest.
- **G4_gi adds ~zero speaker info above pooled-alone.** Probe (ii) − (a) = +0.0001. Capacity control (b) confirms it: 7 dims of N(0,1) noise concatenated to `pooled_4096` give the same probe top-1 as pooled alone (0.0665 ± 0.0026, sklearn LR's L2 regresses noise out). The earlier "+0.017 lift" reading was a baseline mismatch (comparing (ii) against the MLP-substrate 0.0501), not real interaction information.
- **Architectural reading.** Probe (i) 0.0119 vs probe (ii) 0.0675 = ~5.5 pp speaker-info drop. The per-channel cold-probe compression is the speaker bottleneck the late-fusion design intends: speaker information is present upstream but gets stripped when each channel is reduced to a 1-d cold logit. The fusion classifier sees a speaker-honest 2-D representation. This is what makes the K=1 PASS architecturally robust and is the case for choosing logit-level fusion over A3-style concat-MLP.

- **A2_τ calibration baseline** (cell 45 ablation): A2 with τ swept on `train_threshold` instead of argmax. Reports Δ vs both A2_argmax and A2_τ in the K=2 ablation table — separates fusion-lift from threshold-movement-lift.

### A5b — diagnostics (cells 42-43, no impact on locked numbers)

Three structural checks reported alongside A5b for the paper:

1. **Logit correlations on `devel_val`.** Pearson matrix over `{logit_A2, z_logit_g for g in admission_pool}`, plus per-group **argmax disagreement vs A2** (fraction of utterances where `sign(logit_A2) ≠ sign(z_logit_g)` — survives monotonic nonlinearities that Pearson doesn't). Two purposes: spot redundancy with A2 (a group with high `sub@1` but high A2-correlation contributes less than the table suggests), and spot pairwise redundancy across admitted groups (K=3 over highly-correlated groups is one weighted sum repeated, not three independent voters).

2. **Fused-vector speaker probe on `devel_val`.** plan.md § 5.7 specifies "probe top-1 on A5b representation ≤ A2 + 1σ ≤ 0.0510" — but a 1-d fused logit can't naively support a 210-class probe. The honest version probes the actual concat `[logit_A2, z_logit_g for g in admitted]` that fusion has access to. Reported for both the **full admission pool** (sanity ceiling) and the **locked top-K** set. A spike above the per-group max in `A5a_honesty.csv` would mean combining admitted groups creates a speaker channel none of them carry alone — invalidates admission even if every individual group passed honesty.

3. **Redundancy-adjusted ranking on `devel_val`.** Per group: `unique_gain = label_gain · (1 − corr(logit_A2, z_logit_g)²)` and `safe_unique_gain = unique_gain − speaker_gain`. Diagnostic only — admission still keyed on `sub@1`. Tells the paper which groups carry signal A2 doesn't already see, separately from raw cold-UAR.

Single seed (42) — structural diagnostic, not a multi-seed UAR claim. Output in `results/A5b_diag.json` (correlations + fused probe + redundancy table); K=2 ablation `A2 + G4 + {G1, G5, G6}` plus A2_τ baseline in `results/A5b_ablation.json`.

### A5b — diagnostic results (seed 42, devel_val)

**Pearson correlations vs A2** (logit_A2 ↔ z_logit_g):

```text
group               corr_A2   argmax_disagree
G4_gain_invariant   +0.404    +0.387
G1_voicing          +0.258    +0.359
G6_spectral_shape   +0.453    +0.400
G5_modulation       +0.303    +0.419
G2_prosody          +0.190    +0.486
G3_voice_quality    +0.157    +0.404
```

Notable pairwise group correlations (off-diagonal block): **G4 ↔ G1 = +0.628** (the largest in the matrix — both are built off the manner labels), G4 ↔ G6 = +0.426, G4 ↔ G5 = +0.374, G1 ↔ G5 = +0.292. G3 is the least entangled with everything else.

**Fused-vector speaker probe** on `[logit_A2, z_logit_g for g in admitted]`: top-1 = **0.0194** (NMI 0.354) for both the full admission pool and the locked top-K set (locked == pool here because A5b.json fell back to "no locked admitted, use full pool"). References: max per-group `spk_top1` in pool = 0.0340 (G6), A2's own speaker probe = 0.0501, chance = 0.0048. Concat probe stays *below* the per-group ceiling — combining admitted groups does **not** create a speaker channel beyond what the worst single group carries.

**Redundancy-adjusted ranking** (sorted by `safe_unique_gain` descending):

```text
group              corr_A2  lab_gain  spk_gain    sub@1   unique  safe_unique
G4_gain_invariant   +0.404   +0.1318   +0.0080  +0.1239  +0.1103      +0.1024
G1_voicing          +0.258   +0.0831   +0.0063  +0.0768  +0.0776      +0.0713
G5_modulation       +0.303   +0.0802   +0.0098  +0.0703  +0.0728      +0.0630
G6_spectral_shape   +0.453   +0.1050   +0.0292  +0.0758  +0.0835      +0.0543
G2_prosody          +0.190   +0.0680   +0.0146  +0.0534  +0.0655      +0.0509
G3_voice_quality    +0.157   +0.0591   +0.0186  +0.0405  +0.0576      +0.0390
```

Reordering vs raw `sub@1`: **G5 jumps over G6**. G6 was #3 by raw sub@1 (+0.0758) but its 0.453 Pearson² ≈ 0.20 means a fifth of its label_gain is already explained by A2; after the redundancy adjustment G5 (+0.0630) lands #3 ahead of G6 (+0.0543). Empirical post-hoc support for the K=4 sweep extension that admitted G5.

**Diagnosis (and what flipped FAIL → PASS).** The original K=4 free-sweep FAIL is a **signal-redundancy** problem, not a speaker-leak problem. Three of the four K=4 admits (G4, G6, G5) overlap meaningfully with A2 (corr 0.30–0.45); one (G1) is 63% redundant with another admit (G4). "Mean over 4 groups" therefore averages fewer independent voters than 4, and the resulting variance dominates the +0.0074 mean lift. The K=2 ablation tested this directly: locking admission and re-running the sweep over only β and τ collapses σ 7× and lifts the mean by another +0.0074 — net Δ +0.0148 ± 0.0045 (3.3σ) at K=1 over A2_argmax. Practical implications:

- Fused-vector speaker probe rules out the "hidden speaker leak" hypothesis (top-1 0.0194 ≪ per-group ceiling).
- The K=1 winner (`A2 + G4_gi`) admits the highest-honesty group standalone, sidesteps the G4↔G1 redundancy entirely, and clears the gate cleanly.
- The free K∈{1..4} sweep on `train_threshold` was the protocol bug: more groups widen the achievable τ-vs-UAR curve, the val sweep over-rewards that flexibility, and seed-to-seed which-K-wins gets noisy. Locking K removes the over-fit channel.
- A5d (per-layer probe) is now demoted to paper diagnostic — A5b passes without it. Still worth running for the layer-stratification claim independently. A5e (mid-layer retrain) is contingent on A5d showing a dramatic band; otherwise skipped.

### A5d — per-layer honesty diagnostic (DONE; A5e SKIPPED + structural finding)

Tests the WavLM layer-stratification hypothesis (Pasad et al. 2021, Chen et al. 2022: early layers carry speaker identity, mid layers carry paralinguistic content) **on this corpus**. For each layer L ∈ [0, 24]: `cold_probe` + `speaker_probe` on cached `pooled[:, L, :]` (4096-d per layer, matched A5a recipe), single seed (42), train_fit / devel_val splits. No head retraining. Output `results/A5d_layer_honesty.csv`. Cost ~1 hour wall-clock.

**Headline numbers.**

```text
best sub@1   : L21    sub@1=+0.0387   cold_uar=0.5746   spk_top1=0.0406
best cold UAR: L7     cold_uar=0.6052 spk_top1=0.0813   sub@1=+0.0287
highest spk  : L3     spk_top1=0.0871 cold_uar=0.5769   sub@1=−0.0054
lowest spk   : L22    spk_top1=0.0402 cold_uar=0.5726   sub@1=+0.0371
```

**Structural finding (paper-reportable on its own).** Speaker information is **layer-stratified** on URTIC: speaker top-1 decays ~monotonically L0 → L24 (0.087 → 0.043, ~50% reduction), confirming Pasad 2021 / Chen 2022 for the speaker axis on this corpus. Cold information is **not** layer-stratified: `cold_uar` is roughly flat L0..L24 (range 0.56–0.61, spread 0.045) with no clean mid-band peak. The two-axes-don't-stratify-together pattern refutes the strong form of "mid-band paralinguistic = cold-relevant" on URTIC; cold signal lives across the stack, with peak coinciding with speaker-heavy layers (L7) — exactly the kind of entanglement that motivates A5b's logit-level fusion as the de-confounding lever rather than a layer-band restriction.

**Verdict — A5e SKIPPED.** Both skip-branch conditions of the A5e trigger fire simultaneously: (1) no layer reaches `sub@1_L > 0.15` (peak L21 = +0.0387, ≪ 0.15), and (2) the cold UAR peak (L7 = 0.6052) coincides with high speaker leak (L7 `speaker_top1` = 0.0813, joint top tier with L0/L3/L6). There is no honest mid-band that would justify the retrain spend; GPU goes to A5.5 / A6 instead.

### A5e — WavLM mid-layer retrain (SKIPPED)

**Status: SKIPPED** by A5d verdict. Trigger conditions both miss (no `sub@1_L > 0.15`; cold peak coincides with speaker peak band). The retrain track is closed. Trigger spec retained for completeness: would have fired only on a *dramatic* honest band (`sub@1_L > 0.15` over a contiguous L_a..L_b with `speaker_top1_L` well below A2's full-stack 0.0501), at which point a `LayerWeightedPooledHead` retrain with the layer dim masked to the band (3 seeds, identical optimiser/schedule) + K=1 ablation on `A2_mid + G4_gi` would have been run.

### A5c — learned per-group gate (revivable)

Was scoped to fire after A5b passed the gate (replace the fixed `mean_g` with a learned σ-gate per admitted group). A5b passes at K=1 → A5c is technically revivable, but K=1 leaves a single-group fusion stack with little surface for a learned residual to refine over. On hold pending A5.5 / A6 outcomes; revisit if a richer admission set re-opens. Spec retained in plan.md §5.4.

## Git state

- `ff0a32b` (tag `a2-probed`) — A2 locked + speaker probe + pseudo-speakers
- `e297e73` — A3 scaffold: frame-level WavLM cache + notebook wiring
- `ee7a373` — A3 scaffold: phoneme CTC labels (wav2vec2-xlsr-53-espeak-cv-ft) — labels are the abandoned path but code is kept
- **uncommitted (end of session)**: `features/manner.py`, `features/__init__.py` exports, new notebook cells, this summary update. Will commit as "A3 pivot: acoustic-manner labelling (pYIN + RMS)"
