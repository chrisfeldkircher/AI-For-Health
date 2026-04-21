# Project summary — Advanced AI in Health (ComParE 2017 Cold)

Running status doc for the cold-detection attack plan. See [results/README.md](results/README.md) for the rigorous per-rung ablation table and [C:/Users/Chris PC/.claude/projects/.../memory/project_context.md](C:/Users/Chris%20PC/.claude/projects/e--Development-Research-Advanced-AI-in-Health/memory/project_context.md) for the high-level framing.

## Goal

Binary audio classification: Cold vs Non-Cold on the ComParE 2017 Cold sub-challenge (URTIC 4students release, ~28.7k chunks across train/devel/test). Official test labels are withheld by the instructors; devel is our honest proxy. Target UAR to beat: **71.0** (2017 late-fusion baseline).

## Attack-plan status

| Rung | Status       | Headline                                                                 |
|------|--------------|--------------------------------------------------------------------------|
| A2   | **locked**   | Frozen WavLM-Large + layer-weighted pooled-stats probe → **UAR 0.6428 ± 0.0034** |
| A3   | next         | Phoneme-aware pooling (scoping: charsiu aligner, 5 phonetic categories)  |
| A4   | planned      | Discrete audio tokens as auxiliary feature stream                        |
| A5   | planned      | OOD feature family                                                       |
| A5.5 | planned      | Augmentation — directly attacks training-speaker shortcut                |
| A6   | planned      | Contrastive pretraining (speaker-masked loss)                            |
| A7   | planned      | MDD adversarial head — highest-variance, highest-upside bet              |
| A9   | planned      | Late fusion with ComParE + SVM                                           |

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
    extract.py            Batched pooled-stats extraction with masked pooling
    cache.py              CacheManifest (checkpoint_hash + version compat check)
    standardizer.py       FeatureStandardiser (per-position z-score, registered buffers)
    head.py               LayerWeightedPooledHead
    train.py              train_head, evaluate, sweep_threshold, evaluate_at_threshold
  speakers/
    ecapa.py              SpeechBrain ECAPA-TDNN extraction → cache/ecapa-voxceleb/
    cluster.py            KMeans k-sweep over train; writes cache/pseudo_speakers/k{K}_seed{S}.tsv
    probe.py              SpeakerProbe (2-layer MLP), extract_z, train_probe
  run.ipynb               orchestration cells — training, calibration, probe
cache/
  microsoft_wavlm-large/  pooled stats + per-seed head checkpoints (head_A2_seed{S}.pt)
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

## Open decisions (for A3)

1. **Phoneme aligner**: leaning `charsiu_forced_aligner` over MFA — Python-only, MFA-quality, no transcripts required. Open to MFA if you want the gold-standard.
2. **Pooling granularity**: leaning **5 phonetic categories** (nasals, fricatives, plosives, vowels, silence) — respects Wagner's fricative/nasal finding without going so fine we get empty buckets on 2–8s chunks.
3. **Version control**: repo is not git-initialised. Want me to `git init` + commit + tag `a2-locked`, or are you handling externally?

## Immediate next step

Awaiting go-ahead on the A3 scoping. Once decisions (1) and (2) are locked, I'll write `model/features/phoneme_align.py` + pooling head extension + notebook cell, run on the 3 lock seeds, and add the A3 row.
