# Project summary — Advanced AI in Health (ComParE 2017 Cold)

Running status doc for the cold-detection attack plan. See [results/README.md](results/README.md) for the rigorous per-rung ablation table and [C:/Users/Chris PC/.claude/projects/.../memory/project_context.md](C:/Users/Chris%20PC/.claude/projects/e--Development-Research-Advanced-AI-in-Health/memory/project_context.md) for the high-level framing.

## Goal

Binary audio classification: Cold vs Non-Cold on the ComParE 2017 Cold sub-challenge (URTIC 4students release, ~28.7k chunks across train/devel/test). Official test labels are withheld by the instructors; devel is our honest proxy. Target UAR to beat: **71.0** (2017 late-fusion baseline).

## Attack-plan status

| Rung | Status       | Headline                                                                                 |
|------|--------------|------------------------------------------------------------------------------------------|
| A2   | **locked**   | Frozen WavLM-Large + layer-weighted pooled-stats probe → **UAR 0.6428 ± 0.0034**         |
| A3   | in progress  | Manner-aware pooling (pYIN+RMS, 3 cats) — phoneme-CTC pivoted, manner gate passed        |
| A4   | planned      | Discrete audio tokens as auxiliary feature stream                                        |
| A5   | planned      | OOD feature family                                                                       |
| A5.5 | planned      | Augmentation — directly attacks training-speaker shortcut                                |
| A6   | planned      | Contrastive pretraining (speaker-masked loss)                                            |
| A7   | planned      | MDD adversarial head — highest-variance, highest-upside bet                              |
| A9   | planned      | Late fusion with ComParE + SVM                                                           |

Expected gain per rung: A3–A5 worth ~0.5–1.5 UAR each; A5.5 and A6 worth ~1–2 each; A7 is 2–5 if it works, ~0 if it destabilises. Budget to baseline: ~8 UAR points across ~6 rungs.

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
    train.py              train_head, evaluate, sweep_threshold, evaluate_at_threshold
    phoneme.py            wav2vec2-xlsr phoneme CTC → cache/phoneme_labels/ (ABANDONED, see A3)
    manner.py             pYIN voicing + RMS → cache/manner_labels/ (A3 pivot, pending validation)
  speakers/
    ecapa.py              SpeechBrain ECAPA-TDNN extraction → cache/ecapa-voxceleb/
    cluster.py            KMeans k-sweep over train; writes cache/pseudo_speakers/k{K}_seed{S}.tsv
    probe.py              SpeakerProbe (2-layer MLP), extract_z, train_probe
  run.ipynb               orchestration cells — training, calibration, probe
cache/
  microsoft_wavlm-large/
    pooled/               pooled stats + per-seed head checkpoints (head_A2_seed{S}.pt)
    frames/L{1,4,8,12,16,20,24}/  per-utterance fp16 frames, padding stripped (for A3/A4/A6) — 103 GB
  phoneme_labels/         wav2vec2-xlsr argmax IDs (ABANDONED — see A3 status)
  manner_labels/          pYIN + RMS 3-cat labels aligned to WavLM frames (pending validation)
  ecapa-voxceleb/         28652 × [192] fp16 speaker embeddings
  pseudo_speakers/        k{100,210,420}_seed42.tsv
  speechbrain/            auto-downloaded ECAPA checkpoint
results/
  README.md               per-rung ablation table + methodology + per-rung notes
  A2.json                 full A2 distribution (3 seeds) + speaker probe block
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

### Remaining for A3

1. **Category-pooling extractor** — reads `frames/L{N}/{stem}.pt` + `manner_labels/{stem}.pt`, writes `cache/microsoft_wavlm-large/manner_pooled/{stem}.pt` as `{pooled: [7 layers, 3 cats, 2*1024] fp16, indicator: [3] uint8}` (mean+std only; 3rd/4th moments too noisy on the smallest bucket per utterance).
2. **A3 head** — two branches: A2 pooled-stats (25×4096 → softmax layer-mix → 4096) concat A3 manner-pooled (7×3×2048 → softmax layer-mix shared across cats → 3×2048 = 6144) concat indicator (3). FeatureStandardiser → MLP 128 → BN → GELU → dropout 0.6 → 2-class. Weight decay bumped to 3e-3 given capacity jump.
3. **Train on lock seeds** `{42, 123, 7}`, **re-run speaker probe** — top-1 must not increase vs A2 (manner pooling mustn't smuggle in extra speaker structure).
4. Add A3 row to `results/README.md` + write `results/A3.json`.

**Acceptance**: A3 must beat A2's argmax UAR (0.6428 ± 0.0034) by ≥ 0.007 (2σ at N=3) AND speaker probe top-1 must not exceed A2's 0.0501 ± 0.0009 by more than 1σ. If the head beats A2 but the probe inflates, A3 is rejected as a confound (manner pooling smuggling speaker info). If neither: document as null result, keep `manner_pooled/` cache for possible late fusion, move to A4.

## Git state

- `ff0a32b` (tag `a2-probed`) — A2 locked + speaker probe + pseudo-speakers
- `e297e73` — A3 scaffold: frame-level WavLM cache + notebook wiring
- `ee7a373` — A3 scaffold: phoneme CTC labels (wav2vec2-xlsr-53-espeak-cv-ft) — labels are the abandoned path but code is kept
- **uncommitted (end of session)**: `features/manner.py`, `features/__init__.py` exports, new notebook cells, this summary update. Will commit as "A3 pivot: acoustic-manner labelling (pYIN + RMS)"
