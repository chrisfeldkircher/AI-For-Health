# Project summary — Advanced AI in Health (ComParE 2017 Cold)

Running status doc for the cold-detection attack plan. See [results/README.md](results/README.md) for the rigorous per-rung ablation table and [C:/Users/Chris PC/.claude/projects/.../memory/project_context.md](C:/Users/Chris%20PC/.claude/projects/e--Development-Research-Advanced-AI-in-Health/memory/project_context.md) for the high-level framing.

## Goal

Binary audio classification: Cold vs Non-Cold on the ComParE 2017 Cold sub-challenge (URTIC 4students release, ~28.7k chunks across train/devel/test). Official test labels are withheld by the instructors; devel is our honest proxy. Target UAR to beat: **71.0** (2017 late-fusion baseline).

## Attack-plan status

- **A2** — *locked.* Frozen WavLM-Large + layer-weighted pooled-stats probe → **UAR 0.6428 ± 0.0034**
- **A3** — *null result, rejected.* Manner-aware pooling (pYIN+RMS, 3 cats) → argmax UAR 0.6344 ± 0.0069 (−0.008 vs A2), probe top-1 0.0555 ± 0.0030 (+0.005 vs A2). Both acceptance gates failed. Cache kept for possible reuse as a feature group inside A5.
- **A5a** — *honesty audit, complete (G5 row pending one cell run).* Per-group cold + speaker probes for {G1 voicing, G2 prosody, G3 voice quality, G4 energy + gain-invariant slice, G6 spectral shape, G8 OOD Mahalanobis}. G4_energy strongest (lab_gain +0.142) but flagged as gain-confound; G4_gain_invariant retains the lift with halved speaker leak. G8 anti-predictive (rejected, documented). G5 modulation (Huckvale's MOD family) added at end of A5a; cell wired, extraction pending. See `results/A5a_honesty.csv` for the full table.
- **A5b** — *constrained late fusion, wired up; pending run.* Final classifier `final_logit = logit_A2 + β · mean_g(zscore_g(logit_g))`, hard top-K admission by `subtractive_honesty`. Sweep β ∈ {0.25, 0.5, 1.0}, K ∈ {1, 2, 3} on `train_threshold`; lock once on `devel_test`. Two diagnostic cells (Pearson logit-correlation matrix + fused-vector speaker probe) follow A5b for paper figures, no impact on locked numbers.
- **A4** — *planned (speculative).* Discrete audio tokens (EnCodec/HuBERT-codes) as auxiliary stream
- **A5.5** — *planned.* Augmentation — directly attacks training-speaker shortcut
- **A6** — *planned.* Contrastive pretraining (speaker-masked loss)
- **A7** — *planned.* MDD adversarial head — highest-variance, highest-upside bet
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

**Within-partition leak — TODO**: `stratified_split` does per-class random shuffle only; `train_fit`/`train_threshold` and `devel_val`/`devel_test` therefore share pseudo-speakers. The leak is mild (val→test gap is centred on zero, so the load-bearing disjointness is the cross-partition one) but real for early stopping on `devel_val` and threshold τ on `train_threshold`. Fix: replace with `stratified_grouped_split` using pseudo-speaker IDs from `cache/pseudo_speakers/k210_seed42.tsv` as the group key. After patching, re-run A2 (3 seeds, ~10 min total — pooled features cached) and re-run the speaker probe on `devel_test`; if top-1 stays at ≈ 0.05 the leak was minor and our reported numbers are honest, if it drops further we found and closed an internal leak. Either outcome is paper-reportable. See [plan.md §4.5](plan.md).

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
  A2.json                 full A2 distribution (3 seeds) + speaker probe block
  A3.json                 rejected A3 distribution (3 seeds) + diagnosis block
  A5a_honesty.csv         per-group honesty rows (G1, G2, G3, G4, G4_gain_invariant, G5, G6, G8)
  A5b.json                A5b sweep results + locked (β*, K*, τ*) + devel_test (after running)
  A5b_diag.json           A5b correlation matrix + fused-vector speaker probe (after running)
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
G6_spectral_shape         21  0.6050    +0.1050     0.0340    +0.0292    +3.48   +0.0758
G1_voicing                 9  0.5831    +0.0831     0.0110    +0.0063   +11.41   +0.0768
G2_prosody                10  0.5680    +0.0680     0.0194    +0.0146    +4.35   +0.0534
G3_voice_quality          14  0.5591    +0.0591     0.0233    +0.0186    +3.02   +0.0405
G8_ood_mahalanobis         1  0.4334    -0.0666     0.0073    +0.0025   -18.86   -0.0692
G5_modulation             64    pending — cell 39 in run.ipynb (Huckvale MOD family)
```

Reading: `lab_gain = UAR − 0.5`, `spk_gain = top1 − 1/210`, `sub@1 = lab_gain − 1·spk_gain` (admission key, λ=1). Linear-only probes (matched cold + speaker LR, balanced for cold, multinomial for speakers); StandardScaler fit on `train_fit`, evaluated on `devel_val`.

Highlights:

- **G4_energy** is the strongest single group but raises a recording-gain confound concern. The **G4_gain_invariant** ablation (drop absolute-RMS cols 0-3, keep regime-contrast and pause-shape cols 4-10) loses only 0.010 UAR while halving speaker_gain → admit the gain-invariant slice instead, document G4_energy as the comparison row.
- **G1_voicing** has the best ratio (11.41) — cleanest signal in the table. Cold-biased (recall_C > recall_NC) — useful complement to A2's NC-bias.
- **G6_spectral_shape** carries the second-strongest predictive lift but the highest speaker leak (low-MFCCs are by construction speaker-rich — vocal-tract envelope is speaker identity). Passes admission at λ=1 (sub@1 +0.076), would fail at λ=2 — borderline.
- **G8_ood_mahalanobis** anti-predictive (UAR 0.433, label_gain −0.067). Documented negative result — the original PDF-A5 hypothesis ("OOD distance from healthy manifold predicts cold") doesn't hold on URTIC. Excluded from admission pool.
- **G5_modulation** added late as Huckvale's MOD family from the 2017 ComParE write-ups: per-mel-band FFT-over-time → modulation spectrum, aggregated to 4 acoustic super-bands × 8 log-spaced modulation bands × {mean, std} = 64-d. Captures syllable-rate (3-8 Hz) and slow-envelope (<2 Hz) dynamics that no other group sees. Cell wired, ~5 min CPU on full corpus, audit row pending.

### A5b — late fusion (wired, pending run)

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

### A5b — diagnostics (cells 42-43, no impact on locked numbers)

Two structural checks reported alongside A5b for the paper:

1. **Logit correlations on `devel_val`.** Pearson matrix over `{logit_A2, z_logit_g for g in admission_pool}`, plus per-group **argmax disagreement vs A2** (fraction of utterances where `sign(logit_A2) ≠ sign(z_logit_g)` — survives monotonic nonlinearities that Pearson doesn't). Two purposes: spot redundancy with A2 (a group with high `sub@1` but high A2-correlation contributes less than the table suggests), and spot pairwise redundancy across admitted groups (K=3 over highly-correlated groups is one weighted sum repeated, not three independent voters).

2. **Fused-vector speaker probe on `devel_val`.** plan.md § 5.5 specifies "probe top-1 on A5b representation ≤ A2 + 1σ ≤ 0.0510" — but a 1-d fused logit can't naively support a 210-class probe. The honest version probes the actual concat `[logit_A2, z_logit_g for g in admitted]` that fusion has access to. Reported for both the **full admission pool** (sanity ceiling) and the **locked top-K** set. A spike above the per-group max in `A5a_honesty.csv` would mean combining admitted groups creates a speaker channel none of them carry alone — invalidates admission even if every individual group passed honesty.

Single seed (42) — structural diagnostic, not a multi-seed UAR claim. Output in `results/A5b_diag.json`.

## Git state

- `ff0a32b` (tag `a2-probed`) — A2 locked + speaker probe + pseudo-speakers
- `e297e73` — A3 scaffold: frame-level WavLM cache + notebook wiring
- `ee7a373` — A3 scaffold: phoneme CTC labels (wav2vec2-xlsr-53-espeak-cv-ft) — labels are the abandoned path but code is kept
- **uncommitted (end of session)**: `features/manner.py`, `features/__init__.py` exports, new notebook cells, this summary update. Will commit as "A3 pivot: acoustic-manner labelling (pYIN + RMS)"
