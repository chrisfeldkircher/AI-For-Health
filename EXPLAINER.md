# EXPLAINER — Cold detection from voice, end to end

A walking tour of what we built, why each step is the right one, and what each feature physically means. Written for someone comfortable with ML basics but new to audio / speech.

> **Audience.** You know what a logistic regression is, what cross-entropy loss is, what overfitting is. You're rusty on signal-processing terminology like *MFCC*, *F0*, *jitter*, *spectral tilt*, etc. — every piece of audio jargon is defined before it's used.

The companion files for raw numbers and decisions:
- [plan.md](plan.md) — the canonical attack plan (what we're trying, in what order, with status flags)
- [summary.md](summary.md) — the live status doc with locked numbers
- [results/](results/) — per-rung JSON dumps and the A5a honesty CSV
- [model/run.ipynb](model/run.ipynb) — the actual runnable cells

---

## 1. The task

**ComParE 2017 Cold sub-challenge.** Binary classification: *does this 8-second speech clip come from a person with a cold or not?* The dataset is **URTIC** ("Upper Respiratory Tract Infection Corpus") — German speakers reading the children's story *Die Buttergeschichte* into a headset microphone in a controlled recording booth. We have ~28k 8-second chunks split into `train` / `devel` / `test`. The official `test` labels are withheld by the challenge organisers, so we use `devel` as our honest proxy.

**Metric: UAR (Unweighted Average Recall).** Average of recall on each class:

```text
UAR = 0.5 * (recall_cold + recall_non_cold)
```

UAR is used instead of accuracy because the dataset is imbalanced (~9.5% cold, ~90.5% non-cold). A model that always predicts "non-cold" gets 90% accuracy but UAR = 0.5 (chance). Cold and non-cold contribute equally regardless of how many examples each has.

**Numbers to anchor on:**
- Always predicting non-cold → UAR = **0.500** (chance baseline by definition)
- 2017 ComParE late-fusion baseline → **0.710** (the number to beat)
- Huckvale 2018 reanalysis on honest test → **0.621** (the *honest* number — see § 4 below)
- Our locked A2 baseline → **0.6428 ± 0.003** (already above Huckvale's honest test)

---

## 2. Audio, the 30-second crash course

### Waveform
A *digital audio signal* is a 1-D array of amplitude samples at a fixed sample rate. URTIC is **16 kHz**, so 8 seconds = 128,000 samples. Each sample is roughly the air pressure at the microphone diaphragm at that instant, normalised to `[-1, 1]`.

### Frames
Audio models almost never look at raw samples directly. We chop the waveform into short overlapping windows ("frames"), e.g. 25 ms long, every 20 ms ("stride" / "hop"). Each frame becomes one column in a 2-D *time-frequency* representation. For our work the convention is **50 frames per second** (hop = 320 samples at 16 kHz = 20 ms).

### Spectrum and spectrogram
For each frame we run an **FFT** (fast Fourier transform) which decomposes the waveform into its constituent sinusoids — gives an array of energy at each frequency. A *spectrogram* is the matrix of those spectra stacked over time. The y-axis is frequency, x-axis is time, colour is energy.

### Mel spectrogram
Human hearing is logarithmic in frequency — we discriminate finely at low frequencies, coarsely at high ones. The **mel scale** is a frequency warping that mimics this. A *mel spectrogram* applies a triangular filterbank on the spectrogram, summing energy into ~40 perceptually-spaced bands. It's the canonical input to almost every speech model since the 1990s.

### MFCC
**Mel-frequency cepstral coefficients.** Take a log-mel spectrogram, run a discrete cosine transform on each frame's mel vector — this decorrelates the bands and packs most of the spectral *shape* information into the first few coefficients. Low-order MFCCs (1–6) describe the broad spectral envelope: how energy is distributed across frequencies. Higher-order MFCCs describe finer structure (often related to pitch harmonics).

### F0 (fundamental frequency, "pitch")
When the vocal folds vibrate (during *voiced* speech like vowels), they do so at some rate — typically 80–250 Hz for adults, depending on speaker, intonation, mood. That rate is the **fundamental frequency, F0**, perceived as pitch. During *unvoiced* speech (like /s/, /f/, /sh/) the vocal folds don't vibrate and F0 is undefined. Algorithms like **pYIN** (probabilistic YIN) estimate F0 per frame and also tell you whether each frame is voiced.

### Voicing manner
A high-level categorical label per frame:
- **silence** — energy below a threshold (the speaker isn't producing sound)
- **voiced** — vocal folds vibrating (vowels, nasals like /m/, voiced consonants like /b/, /d/, /z/)
- **unvoiced** — turbulent airflow with no vocal-fold vibration (/s/, /f/, /sh/, /t/, /k/)

Cold affects voiced and unvoiced frames differently (vocal folds inflame; mucus changes airflow turbulence), so conditioning features on this label is one of our main moves.

### Jitter and shimmer
Two voice-quality measures of how *regular* vocal-fold vibration is:
- **Jitter** — frame-to-frame variability in F0 (the period of the vibration). High jitter means the pitch wobbles. Healthy voice: <0.5%. Pathological voice: >2%.
- **Shimmer** — frame-to-frame variability in *amplitude*. High shimmer means each glottal cycle has a different loudness. Often correlates with breathy voice.

Both rise during illness — colds, laryngitis, fatigue all introduce micro-perturbations into vocal-fold mechanics.

### HNR
**Harmonics-to-noise ratio.** During voicing, the spectrum has a clean harmonic stack (peaks at F0, 2·F0, 3·F0, ...) plus background broadband noise. HNR is the ratio of energy in the harmonics to energy in the noise floor, in dB. Healthy adult voice: 15–25 dB. Cold/hoarse voice: lower (often <10 dB) — more noise relative to harmonic structure.

### Spectral tilt / alpha ratio / Hammarberg index
The slope of the spectrum from low to high frequencies. Healthy modal voice has more energy at low frequencies (rolling off ~12 dB/octave). *Pressed* or *tense* voice has a flatter spectrum (more high-frequency energy). Cold often pushes voice towards a more pressed quality. Operationalised as:
- **alpha ratio** — energy ratio between two bands (e.g. 50–1000 Hz vs 1–5 kHz)
- **Hammarberg index** — peak-to-peak energy ratio between low and high bands
- **slope V0-500, V500-1500** — linear fit of dB-vs-frequency in a band, voiced frames only

### Spectral flux
How much the spectrum *changes* between consecutive frames. High flux = articulation is active (transitions, plosives). Low flux = sustained vowel or silence. Often elevated in faster speech.

### RMS (root-mean-square energy)
Per-frame loudness:
```text
RMS_t = sqrt( (1/N) * sum( x_t[i]^2 for i in window ) )
```
Reported either linearly or in dB (= 20·log10(RMS) + offset). Talking louder → higher RMS. Background silence → low RMS. Doesn't tell you what *kind* of sound it is, just *how much* of it.

### Modulation spectrum
A spectrogram tells you *what frequencies* are present at each instant. A **modulation spectrum** tells you *how fast each frequency is amplitude-modulating over time*. Concretely: take one row of a log-mel spectrogram (the energy in a single mel band as a function of time), treat it as its own 1-D signal, and run an FFT over time. The result is a per-band spectrum whose x-axis is *modulation rate in Hz* — how many times per second that band's amplitude wobbles up and down.

Why it's interesting: speech has natural rhythms at characteristic modulation rates. **<2 Hz** is the breath-group / phrase scale; **3-8 Hz** is the syllable rate (the canonical "speech rhythm"); **10-20 Hz** is the fine glottal / articulatory perturbation scale (close to the regime where jitter and shimmer live). Cold disturbs all three: congestion slows syllable rate, breathiness changes phrase pacing, vocal-fold inflammation perturbs the high-rate band. None of the per-frame static features (MFCC, jitter, HNR) capture these *cross-frame* dynamics — they describe single frames or differences between adjacent frames, not the full envelope FFT. That's the gap G5 fills.

---

## 3. Foundation models for audio (WavLM)

The 2017 ComParE baseline used 6,373 hand-engineered features (the **ComParE feature set**) plus an SVM. That was state-of-the-art in 2017. In 2026 we have **self-supervised foundation models** trained on tens of thousands of hours of speech, which learn rich representations end-to-end without any labels.

**WavLM-Large** is one of these. It's a 316M-parameter Transformer trained on 94k hours of unlabelled speech with a denoising-and-prediction objective. Once trained, it takes a raw waveform and produces a stack of 25 frame-level hidden states (one per Transformer layer) at 50 Hz. Each hidden state is a 1024-d vector per frame. Different layers encode different things — early layers capture phonetic structure, late layers capture semantic / paralinguistic content.

We use WavLM **frozen** — no fine-tuning. The classifier on top is small. This is partly compute discipline (URTIC has only ~9k training chunks, fine-tuning a 316M model would overfit immediately) and partly a methodological choice (so the comparison "what do hand-crafted features add on top of WavLM?" has a clean control).

---

## 4. The shortcut-learning problem (the Huckvale trap)

This is the methodologically important part. The 2017 baseline reported UAR 71.0 on devel and the headline number became 71.0. **Huckvale 2018 reproduced the same pipeline and reported 62.1 on the held-out test set** — a 9-point drop. What happened?

**Speakers.** URTIC has ~210 speakers per partition. Each speaker contributes many chunks. The corpus organisers split train/devel/test so the **speakers are disjoint** between partitions, but in practice the *recording conditions* drift between partitions and so does each speaker's voice timbre, mic gain, and reading style. A model that learns "this voice = cold" on `train` (where some speakers are coldy and some aren't) cannot generalise to a speaker it has never seen, because what it actually learned is "Speaker_42's voice has these formants" — not the disease.

The 2017 number was inflated by **shortcut learning on speaker identity**. Iterating on the devel set picked hyperparameters that exploited devel-speaker idiosyncrasies. The honest number — what you get when you only ever look at the test set once — is much lower.

**Our methodological discipline:**
- Threshold τ is selected on `train_threshold` (10% of train, never seen during training and never the same as devel).
- Model selection is on `devel_val` (50% of devel) early-stopping signal; *never* on `devel_test`.
- `devel_test` (the other 50% of devel) is the one-shot honest UAR estimate. We look at it once per locked rung.
- We track the **val→test gap** every rung. If val and test disagree systematically, we're tuning on val.
- We measure **speaker leakage** explicitly via a probe (see § 6).

This protocol is what gives our A2 number (0.6428) credibility — it survives the Huckvale check.

---

## 5. The split layout (read this once, refer back forever)

```text
train (9505)  ──┬─ train_fit       (90%, 8554)   ←  fit heads + per-group probes
                └─ train_threshold (10%,  951)   ←  pick τ; pick (β, K) for A5b

devel (9596)  ──┬─ devel_val       (50%, 4798)   ←  early stopping; honesty audit eval
                └─ devel_test      (50%, 4798)   ←  one-shot honest UAR per rung

test  (9551)                                     ←  withheld; not used
```

The `stratified_split` is per-class, so the cold rate stays ~9.5% in every slice. `split_seed=42` is locked across every rung — same files in every partition, every time.

---

## 6. Pseudo-speakers and the speaker probe

**Problem.** URTIC's 4students release strips speaker IDs. We can't directly check whether the model is shortcutting on identity.

**Solution.** Build pseudo-speaker IDs by clustering speaker embeddings.

1. **ECAPA-TDNN**, an off-the-shelf model trained on VoxCeleb for speaker verification, produces a 192-d embedding per utterance. Two utterances from the same speaker land close in this space; different speakers land far apart. ECAPA was *designed* to be invariant to content and channel — it captures voice identity, period.
2. **KMeans with k=210** (≈ the known per-partition speaker count) on those embeddings → 210 cluster IDs. We treat each cluster as a "pseudo-speaker."
3. **Cross-validation.** HDBSCAN finds 204 clusters independently; KMeans-vs-HDBSCAN agreement is ARI 0.856 / NMI 0.962. Two independent clustering algorithms recover essentially the same groups → the clusters are real speaker structure, not silhouette artefacts.
4. **Negative control.** WavLM-base-plus-sv (a different speaker model) couldn't recover speaker structure on URTIC — KMeans-vs-HDBSCAN ARI = 0.093. This rules out architectural circularity (the speaker structure isn't an artefact of clustering ECAPA representations *and then* probing with ECAPA-derived structure).

**The speaker probe.** Given any feature vector for an utterance, train a logistic regression to predict its pseudo-speaker ID (210-class). On `devel_val` the probe sees chunks from speakers it never saw during fitting (since `train` and `devel` are speaker-disjoint), so high top-1 accuracy means the *features themselves* carry generalisable speaker identity — not just memorised mappings.

- **Chance** = 1/210 = 0.00476
- **A2's WavLM features** → top-1 = 0.0501 (~10× chance) — measurable speaker leak
- **A3 manner-pooled features** → top-1 = 0.0555 — *more* leakage, hence rejected
- A speaker probe trained on the *same* features as the cold classifier tells you how much of the cold model's prediction could plausibly be speaker shortcut

---

## 7. The path so far

### A1 (folded into A2)
The original PDF plan started with mean-pool over WavLM frames. Mean-pool is a strict subset of layer-weighted pooled-stats, so we skipped it.

### A2 — locked, UAR 0.6428 ± 0.003
Frozen WavLM-Large, 25 layers, per-layer pooled stats (mean + std + skew + kurt over the time axis = 4 × 1024 = 4096-d per layer). Then:
1. **FeatureStandardiser** — z-score each of the 25 × 4096 features on `train_fit`. Critical: raw pooled stats span ~4 orders of magnitude in std, which collapses the head to majority-class.
2. **Softmax layer weights** — learn 25 weights, softmax-normalised; sum the layers into one 4096-d vector.
3. **2-layer MLP** (4096 → 128 → 128 → 2 logits) with BatchNorm, GELU, dropout 0.5.
4. Balanced sampler in the DataLoader (each batch sees roughly equal cold and non-cold).

This is the **anchor** for everything downstream. β_A2 = 1 in late fusion; we never touch this number once it's locked.

### A3 — null result, rejected
Tried two ideas:
1. **Phoneme-CTC labels** via `wav2vec2-xlsr-53-espeak-cv-ft`. 84% of frames came back as `<blank>` because the model was trained on CommonVoice (clean read speech, IPA targets) and URTIC is recording-booth German with different phoneme inventory. **Documented as abandoned negative result.**
2. **Manner labels** (silence / voiced / unvoiced) from pYIN + RMS. Labels themselves are clean, but the *head* — pool WavLM features per category, concat into 6144-d — failed both gates: argmax UAR −0.008 vs A2, speaker-probe top1 +0.005 vs A2. The high-dim concat let an MLP rediscover speaker shortcuts. **Rejected as a standalone stream.** Manner labels survived as conditioning information for A5 features.

### A5 — current rung
Where we are now. Three sub-rungs:
- **A5a** — *honesty audit* (done). Build per-group features, audit each one for cold predictivity *and* speaker leakage.
- **A5b** — *constrained late fusion* (just wired up). Combine A2 with admitted groups under fixed weights.
- **A5c** — *learned gate* (conditional on A5b passing).

---

## 8. Feature groups (the meat of A5)

For each group `g` we measure:
- `UAR_g` — how well a linear logistic regression on group `g` alone predicts cold on `devel_val`
- `top1_g` — how well the same linear architecture predicts pseudo-speaker on `devel_val`
- `label_gain_g = UAR_g − 0.5` (UAR over chance)
- `speaker_gain_g = top1_g − 1/210` (top-1 over chance)
- `subtractive_honesty_g = label_gain_g − λ · speaker_gain_g` (default λ = 1)

A group gets admitted into A5b if `subtractive_honesty > 0` and `label_gain > 0`. Linear-only by design — if a group needs nonlinearity to predict cold, that's a signal the group should be sub-divided rather than blessed with hidden capacity (which is exactly how A3 found speaker shortcuts).

### G1 — Voicing scalars (free, ~9 dims)

**What:** Statistics computed over the cached manner labels (silence / voiced / unvoiced per frame at 50 Hz).

**Features:** `voiced_fraction`, `unvoiced_fraction`, `silence_fraction`, `voicing_dropout_per_sec` (count of voiced→unvoiced transitions), mean segment lengths for each category, `long_silence_rate_per_sec` (silence runs ≥ 200 ms).

**Physical meaning:** A cold subject often has more pauses (sniffling, throat-clearing breaks) and more voicing dropouts (vocal folds fail to maintain a stable vibration, dropping into a breathy or unvoiced region mid-syllable). Speech rhythm is altered.

**Extraction:** Pure post-processing over `cache/manner_labels/*.pt`. Zero waveform reads. Free.

**A5a result:** UAR 0.5831, label_gain +0.0831. Speaker leak very low (top1 0.011, ~2× chance). **Highest honesty ratio (11.4)** — the cleanest signal in the audit. Cold-biased: recall_C = 0.642 > recall_NC = 0.524, so it's pulling toward predicting cold on the margin (useful complement to A2 which under-recalls cold).

### G2 — F0 / prosody (~10 dims)

**What:** Per-utterance statistics of the F0 contour.

**Features:** `f0_mean_hz`, `f0_std_hz`, percentiles (p10, p90), `f0_range_hz` (p90−p10, robust spread), `f0_log_mean_st` (mean in semitones-from-100Hz, perceptually meaningful), `f0_jitter_local` (mean |ΔF0| / mean F0), `f0_voiced_fraction`, `f0_missingness_in_voiced` (frames the manner labeller called voiced but pYIN couldn't track F0 — a sanity / quality check), `f0_voiced_run_count_per_sec`.

**Physical meaning:** Cold can *raise* F0 (vocal-fold inflammation makes the folds stiffer → higher pitch) or *lower* it (mucus loading → heavier folds → lower pitch). F0 *variability* increases — pitch-tracking algorithms see more wobble. Voiced run length shortens.

**Extraction:** Two-step. First, run `librosa.pyin` (probabilistic YIN — a pitch tracker) on each waveform with the same hop length (320 samples = 20 ms) as the manner labels, so frame indices align. Save the F0 contour to `cache/f0/{stem}.npy` (NaN where unvoiced). Second, derive scalars from the cached contour. Total CPU: ~11.5 hours one-time on the full corpus.

**A5a result:** UAR 0.5680, label_gain +0.0680. Higher speaker leak than G1 (top1 0.0194, ~4× chance — F0 distribution is somewhat speaker-identifying), but `subtractive@1 = +0.053 > 0` so it passes admission. Also cold-biased.

### G3 — Voice quality (~14 dims)

**What:** The classical clinical voice-quality measures, computed by openSMILE.

**Features:** `jitterLocal`, `shimmerLocaldB`, `HNRdBACF` (mean + std of each), and four spectral-tilt measures restricted to voiced frames: `alphaRatioV`, `hammarbergIndexV`, `slopeV0-500`, `slopeV500-1500`. All are eGeMAPSv02 functionals.

**Physical meaning:** This is the *textbook* feature group for voice pathology. Cold inflames the vocal folds, increases mucus, alters airflow. The result: jitter and shimmer rise (more cycle-to-cycle perturbation), HNR drops (more noise relative to harmonic structure), spectral tilt flattens (voice gets pressed). Schuller 2017, Cummins 2017, Huckvale 2018 all flag this as the most physiologically motivated group.

**Extraction:** `opensmile-python` running the eGeMAPSv02 functionals config — 88 features per utterance. Cache once at `cache/handcrafted/egemaps/{stem}.npy` plus column-name list at `_columns.json`. We then *carve* the 14 voice-quality columns by name prefix. Carving is a runtime slice, not a re-extraction — if we change which columns count as G3 we don't need to re-run openSMILE.

**A5a result:** UAR 0.5591, label_gain +0.0591 (the weakest passing group). Speaker leak is moderate (top1 0.0233, ~5× chance). NC-biased: recall_NC = 0.611 > recall_C = 0.507. Honesty `subtractive@1 = +0.041 > 0` — admitted, but on the margin.

### G4 — Energy / pause / breath (~11 dims)

**What:** RMS-based loudness statistics, both global and conditioned on manner regime.

**Features (cols 0–3, absolute amplitude):** `rms_lin_mean`, `rms_lin_std`, `rms_db_mean`, `rms_db_std`. **(cols 4–10, gain-invariant):** `low_energy_ratio` (fraction of frames in the bottom 10% of RMS), `energy_slope_db_per_sec` (linear fit of dB-vs-time), three regime-contrast scalars (`rms_db_voiced − rms_db_silence`, etc. — these *cancel* any constant gain offset), `long_pause_per_sec`, `median_silence_seg_sec`.

**Physical meaning:** Two distinct signals. (1) Absolute amplitude — a cold patient may speak more quietly, or louder due to compensation. (2) Energy *contrast* between voiced and silence — illness changes how dramatically a speaker modulates loudness.

**Extraction:** One-shot RMS pass over the waveform. Cached per-stem at `cache/handcrafted/g4/{stem}.npy`. Tens of minutes total.

**A5a result:** UAR 0.6418, label_gain +0.1418 — **the strongest single group.**

**The amplitude-confound concern, and what we did about it.** Cols 0–3 (raw absolute amplitude) raise a red flag: if recordings are made with different mic gain across speakers or sessions, "louder = cold" might really be "louder = certain mic". To check, we ran a **gain-invariant ablation**: drop cols 0–3, keep cols 4–10. New UAR 0.6318 (only 0.010 drop) and speaker_gain halves from +0.013 to +0.008. Verdict: G4 is *genuinely physiological*, not a recording-gain artefact. The cleaner slice (`G4_gain_invariant`) is what we admit into A5b — same essential signal, less confound.

### G6 — Spectral shape (~21 dims)

**What:** Low-order MFCCs and spectral flux, in three regime variants.

**Features:** `spectralFlux_*` (no-suffix = whole signal, V = voiced only, UV = unvoiced only), `mfcc1` through `mfcc4` (mean and std-norm of each), and the same three regime variants. Total 21 columns from eGeMAPSv02.

**Physical meaning:** MFCCs describe the broad shape of the spectrum. Cold thickens secretions on the vocal-tract walls and changes nasal coupling, both of which subtly shift the spectral envelope. Spectral flux measures articulation rate — how quickly the spectrum changes frame-to-frame; cold often slows articulation.

**Caveat:** Low-order MFCCs are also classical *speaker* features. They encode vocal-tract shape (how long your vocal tract is, how your tongue resting position curves the resonances). So we expect this group to leak speaker information, and the audit confirms it: speaker_gain = +0.029, the highest of any predictive group (~7× chance top-1). Still, label_gain = +0.105 is the second-strongest, and `subtractive@1 = +0.076 > 0`, so it passes admission at λ=1. It would *fail* admission at λ=2 (= +0.047, still positive, but margin thin).

**Extraction:** Same eGeMAPSv02 cache as G3, just a different name-prefix carve.

**A5a result:** UAR 0.6050, label_gain +0.105. Honesty `subtractive@1 = +0.076`. NC-biased.

### G8 — OOD Mahalanobis (1 dim)

**What:** *One scalar per utterance* — the squared Mahalanobis distance from the utterance's A2-fused vector to the centroid of the non-cold training distribution.

**Idea:** The original A5 (in the PDF plan) was *just* this feature. The reasoning: a model trained mostly on healthy speech learns a "normal" manifold; cold speech is an outlier. Distance from the manifold should correlate with cold.

**Extraction:**
1. Load the trained A2 head.
2. Run forward up to (but not through) the classifier — give us 4096-d fused vectors per utterance.
3. Fit a **LedoitWolf shrinkage covariance** on `train_fit` non-cold vectors (LedoitWolf handles the rank-deficient covariance you get when N < d, common when "non-cold subset of train_fit" is not huge).
4. For each test utterance, compute `(x − μ)ᵀ Σ⁻¹ (x − μ)` — the Mahalanobis distance. That's the feature.

**A5a result:** **UAR 0.4334 — anti-predictive!** label_gain = −0.067. The distance is *lower* for cold cases on average (cold lives near the middle of the non-cold manifold, in the dimensions A2 cares about). This is a *documented negative result* — we drop G8 from the admission pool but keep the row in the table because "OOD as a cold signal" was the original A5 hypothesis and it's important to publish that it doesn't hold here.

### G5 — Modulation spectrogram (64 dims)

**What:** The Huckvale-2018-style **MOD** family — per-mel-band amplitude-modulation rates aggregated to a small fixed-size descriptor. This was the one feature family from the 2017 challenge literature we hadn't covered with G1-G4 / G6, and it captures something genuinely orthogonal: cross-frame *dynamics*, not per-frame *statics*.

**Features:** 4 acoustic super-bands × 8 modulation bands × {mean, std} = 64.
- Acoustic super-bands: split the 40-band log-mel spectrum into low / mid-low / mid-high / high (10 mels per band).
- Modulation bands: 8 log-spaced bins from 1 Hz to 20 Hz — covers slow envelope (<2 Hz), syllable rate (3-8 Hz), and fine perturbation (10-20 Hz) regimes.
- For each (super-band, modulation-band) cell, take the magnitude of the per-mel-band-FFT-over-time within that frequency bin and report (mean, std) over the mel rows in the super-band.

**Physical meaning:** Cold disturbs amplitude modulation at all three of those rate scales — congestion slows syllable rate (3-8 Hz energy redistributes), breathiness changes phrase pacing (<2 Hz), vocal-fold inflammation perturbs the higher rates. None of G1-G4/G6 see across more than a handful of adjacent frames. G5 is the first group that summarises *full-utterance temporal envelope structure* in the frequency domain.

**Extraction:** For each utterance, compute the log-mel spectrogram (40 bands at 100 Hz frame rate, hop=160 at sr=16k). Subtract per-band mean (remove DC). Hann-window in time. rFFT along time → magnitude. Bin and aggregate. Cached at `cache/handcrafted/modulation/{stem}.npy` as 64-d fp32. ~5 minutes CPU on the full corpus.

**Why aggressive aggregation (64-d, not the full per-band spectrum):** the linear-only probe discipline in plan.md § 5.6 says if a group needs nonlinearity to predict cold it should be sub-divided rather than blessed with hidden capacity. A 40-mel × 16-bin per-band spectrum (640+ dims) gives a linear LR enough surface area to launder speaker identity in the same way the A3 head did. Aggregating to 64-d before the probe sees it forces the predictive signal to live in *low-rank* modulation structure, which is what the literature claims is cold-relevant.

**A5a result:** *(pending — cell 39 in `model/run.ipynb` runs the extraction + audit and appends one row to `results/A5a_honesty.csv`.)* Whether G5 enters the A5b admission pool is a data question — the same `subtractive_honesty > 0` and `label_gain > 0` rules as every other group.

### G7 — not implemented in v1

**G7** (regime-conditioned mel-band stats, ~160 dims) was flagged in plan.md § 5.2 as the highest-risk group — it's structurally close to a vocal-tract envelope, the same fingerprint that crashed A3. Held for a stricter acceptance protocol (`label_gain ≥ 0.05` and one-shot eval on `devel_test` before admission). Not run for v1.

---

## 9. The A5a honesty audit, the actual numbers

Snapshot of `results/A5a_honesty.csv` *(pre-G5 — the G5 row appends once cell 39 in `model/run.ipynb` runs)*:

```text
group                    dim     UAR   lab_gain   spk_top1   spk_gain    ratio     sub@1
G4_energy                 11  0.6418    +0.1418     0.0181    +0.0134    +9.87   +0.1284
G4_gain_invariant          7  0.6318    +0.1318     0.0127    +0.0080   +14.73   +0.1239
G6_spectral_shape         21  0.6050    +0.1050     0.0340    +0.0292    +3.48   +0.0758
G1_voicing                 9  0.5831    +0.0831     0.0110    +0.0063   +11.41   +0.0768
G2_prosody                10  0.5680    +0.0680     0.0194    +0.0146    +4.35   +0.0534
G3_voice_quality          14  0.5591    +0.0591     0.0233    +0.0186    +3.02   +0.0405
G8_ood_mahalanobis         1  0.4334    -0.0666     0.0073    +0.0025   -18.86   -0.0692
G5_modulation             64    pending — extraction + audit cell wired (Huckvale MOD family)
```

**Reading this table:**
- **`UAR`** — cold-probe UAR on `devel_val`. Chance = 0.5.
- **`lab_gain`** = UAR − 0.5. How much the group lifts cold prediction over chance.
- **`spk_top1`** — speaker-probe top-1 accuracy on `devel_val`. Chance = 1/210 = 0.0048.
- **`spk_gain`** = top1 − 1/210. How much the group lifts speaker prediction over chance.
- **`ratio`** = lab_gain / (spk_gain + 1e-3). Honest if >> 0; reports cold lift per unit speaker lift. Parameter-free.
- **`sub@1`** = lab_gain − 1·spk_gain. The subtractive form at λ=1. The admission key for A5b.

**Per-group take:**
- **G4_energy / G4_gain_invariant** are the strongest groups. They're the same waveform information so we can only admit one (admitting both would double-count). The gain-invariant slice keeps almost all the predictive power (UAR drop 0.010) with a much cleaner honesty story (ratio 14.7 vs 9.9, spk_gain halved). **Admitted: G4_gain_invariant.**
- **G6_spectral_shape** is the second-strongest predictor but the highest speaker leaker. MFCCs being speaker-rich is a known property — that's why we measured the leak rather than trusting the predictive lift naïvely.
- **G1_voicing** is the most honest signal in the audit (ratio 11.4). Modest predictive lift but speaker-clean.
- **G2_prosody** and **G3_voice_quality** both pass admission at λ=1 but on the margin.
- **G8_ood** is anti-predictive — *documented* and *dropped*.

---

## 10. Late fusion (A5b)

**Formula:**
```text
final_logit = logit_A2 + β · mean_g( zscore_g( logit_g ) )
```
- `logit_A2` is the binary log-odds from A2's classifier. Frozen, β_A2 = 1.
- `logit_g` is `clf.decision_function(scaler.transform(X_g))` from the per-group cold probe. Frozen at the values audited in A5a.
- `zscore_g` removes per-group scale differences (G4's logit naturally spans a wider range than G2's). Mean and std are fit on `train_fit` so β has the same interpretation across groups.
- `mean_g` averages over the **K admitted groups** (the top-K by `sub@1`).
- `β` is a single scalar — no per-group βs, no learned parameters. Hard top-K admission.

**Sweep:**
- `β ∈ {0.25, 0.5, 1.0}`
- `K ∈ {1, 2, 3}` — admission order: G4_gain_invariant → G1_voicing → G6_spectral_shape (by sub@1)
- **τ** (decision threshold on the fused logit) — also swept on `train_threshold`

All sweeping is on `train_threshold`. The locked `(β*, K*, τ*)` is evaluated **once** on `devel_test`. No tuning on devel. Three training seeds {42, 123, 7} for variance estimation.

**Acceptance gate** (plan.md § 5.5): A5b mean UAR on `devel_test` ≥ A2 mean + 0.007 (= 2σ at N=3, so any improvement smaller than this is statistical noise).

**Why this exact formula?**
- *Logit-level* — not concat-then-MLP. Concat-MLP is the substrate that let A3 rediscover speaker shortcuts. Forcing each group through its own 1-d cold-probe logit *before* fusion means every group has to prove standalone cold utility before it gets any β weight.
- *No learned weights* — the table itself is the selection mechanism. Cleaner paper story ("honesty-weighted fusion with no learning at the fusion stage lifts UAR by Δ") than "regularised βs."
- *Hard top-K rather than soft weighted by honesty* — sharper claim, cleaner ablation. If A5c (learned gate) lifts further, the delta is unambiguously the gate.

### 10.1 Diagnostics that ship next to A5b (do not change locked numbers)

Two short cells run after the A5b sweep. Both feed paper figures, neither touches `(β*, K*, τ*)`.

**(a) Logit correlations + argmax disagreement vs A2.** The honesty audit ranks groups in isolation. It cannot tell you whether two admitted groups are pulling in the same direction (in which case `mean_g(z_g)` over both is just one weighted vote repeated twice — wasted budget) or whether they catch different errors (genuine complementarity). We compute:

- A Pearson matrix on `devel_val` over `{logit_A2, z_logit_g for g in admission_pool}` — the diagonal of redundancy with the A2 anchor (right-most column) and pairwise redundancy across groups (off-diagonal block).
- Per-group **argmax disagreement** = fraction of utterances where `sign(logit_A2) ≠ sign(z_logit_g)`. This is a non-Pearson, ranking-style view that survives monotonic nonlinear relationships (e.g. `z_g = sigmoid(logit_A2 + noise)` would correlate strongly with A2 in Pearson but still disagree at the threshold). Higher disagreement = more independent error patterns.

What this answers: a group with high `sub@1` (admitted) but high A2-correlation (Pearson ~0.6+) and low disagreement (<0.2) is mostly re-saying what A2 already says — its admission is a less effective lever than the table suggests. We expected this for G6 (low MFCCs and WavLM both encode spectral envelope); we already saw it *not* be the case for G4 in earlier checks (Pearson ~0.4, argmax disagreement ~0.35 — real complementarity).

**(b) Fused-vector speaker probe.** plan.md § 5.5 says "probe top-1 on A5b representation ≤ A2 + 1σ ≤ 0.0510." Literally probing the 1-d fused logit against 210 pseudo-speaker classes is degenerate (multinomial LR over one feature can't separate that many classes). The honest version probes the actual `[logit_A2, z_logit_g for g in admitted]` vector — what fusion has access to, dimensionality 1 + K. We report top-1 on `devel_val` for two cuts:

- **Full admission pool** — sanity ceiling. If this is fine, every K we'd consider is fine.
- **Locked top-K admitted** (read from `results/A5b.json`) — the actual gate to report.

Comparison reference: max per-group `spk_top1` from the A5a CSV. If the concat top-1 stays at or below that ceiling, no group combination is creating a *new* speaker channel beyond what its parts already carry. If it spikes (say to 0.07+ vs A2's 0.05 reference), one of the admitted groups is amplifying speaker info when seen *jointly* with another — admission would need to be revisited even though every individual group passed.

Single seed (42) — structural, not a multi-seed UAR claim. Output: `results/A5b_diag.json`.

---

## 11. What's next (post-A5b)

Conditional on A5b passing the gate:
- **A5c** — learned per-group gate `β_g = σ(honesty_init_g + learned_residual)`, residual L2-pulled toward zero. Refines the priors rather than overwriting them.
- **A5.5** — cross-speaker splicing augmentation. Symmetric across classes (so splice presence is decorrelated from the label). Measures the speaker shortcut by trying to attack it with augmentation.
- **A6** — supervised contrastive pretraining with speaker-masked positives.
- **A7** — MDD/DANN gradient-reversal speaker adversary. The biggest-upside, biggest-variance bet.

Each subsequent rung must lower the speaker probe top-1 *without* dropping cold UAR materially. That's the two-dimensional acceptance discipline A2 set up and every rung inherits.

---

## 12. Key file map

```text
model/
  data/
    cached_dataset.py     PooledCacheDataset, stratified_split
    data.py               AudioDataset (raw waveform iterator)
  features/
    backbone.py           WavLM / HuBERT / Whisper interface
    extract.py            Batched pooled-stats extraction
    head.py               LayerWeightedPooledHead (A2's head architecture)
    head_a3.py            MannerAwareHead (rejected; kept as documented negative)
    manner.py             pYIN + RMS → 3-cat acoustic manner labels
    f0.py                 pYIN F0 contour, cached at cache/f0/
    opensmile_extract.py  eGeMAPSv02 functionals, cached at cache/handcrafted/egemaps/
    modulation.py         per-mel-band FFT-over-time → cache/handcrafted/modulation/
    scalar_g1.py          voicing scalars from manner labels (G1)
    scalar_g2.py          prosody scalars from F0 + manner labels (G2)
    scalar_g3.py          voice-quality carving of eGeMAPS (G3)
    scalar_g4.py          energy / pause / breath from waveform RMS (G4)
    scalar_g5.py          modulation-spectrogram aggregate, 64-d (G5)
    scalar_g6.py          spectral-shape carving of eGeMAPS (G6)
    ood_g8.py             Mahalanobis distance on A2-fused vectors (G8)
    train.py              train_head, sweep_threshold, predict_probs, evaluate
  honesty/
    probe.py              cold_probe + speaker_probe (matched linear LR)
    audit.py              audit_group; appends one CSV row per group
    fusion.py             A5b math (fit_cold_probe, predict_logit, zscore, fuse, sweep_tau)
  speakers/
    ecapa.py              ECAPA-TDNN embedding extraction
    cluster.py            KMeans pseudo-speaker assignment, load_pseudo_speakers
    probe.py              MLP speaker probe used in A2/A3 evaluation

cache/
  microsoft_wavlm-large/
    pooled/               per-stem WavLM pooled stats [25, 4096] fp16
    frames/L*/            per-stem WavLM frame hidden states (subset of layers)
    head_A2_seed*.pt      A2 head checkpoints (3 seeds)
    head_A3_seed*.pt      A3 head checkpoints (rejected)
  manner_labels/          per-stem 3-cat manner labels [T] int8
  f0/                     per-stem pYIN F0 contour [T] fp32 (NaN at unvoiced)
  handcrafted/
    egemaps/              per-stem eGeMAPSv02 functionals [88] fp32 + _columns.json
    g4/                   per-stem G4 energy scalars [11] fp32
    modulation/           per-stem G5 modulation features [64] fp32
  pseudo_speakers/
    k210_seed42.tsv       pseudo-speaker assignments

results/
  A2.json                 locked A2 metrics (3 seeds)
  A3.json                 rejected A3 metrics (3 seeds)
  A5a_honesty.csv         per-group honesty rows (G1, G2, G3, G4, G4_gain_invariant, G5, G6, G8)
  A5b.json                A5b late-fusion sweep + locked devel_test eval (after running)
  A5b_diag.json           A5b correlation matrix + fused-vector speaker probe (after running)
```

---

## 13. Decision diary (the "why" line by line)

In rough chronological order:

- **Frozen WavLM, not fine-tuned.** 9k training chunks + 316M parameters = guaranteed overfit. Frozen also gives a clean comparison axis for handcrafted features.
- **Layer-weighted pooled-stats over mean-pool.** Different WavLM layers encode different things; let the model learn which layers matter rather than throwing away that structure. Mean-pool is a special case (uniform weights, mean-only).
- **FeatureStandardiser.** Without it, raw pooled stds span 4 orders of magnitude and the head collapses to majority-class. Discovered the hard way during A2 debugging.
- **Balanced sampler over class-weighted loss.** Cleaner gradients, better-calibrated boundary; Huckvale showed this is preferable on URTIC.
- **Phoneme-CTC abandoned.** XLSR-CommonVoice domain-mismatched on URTIC's recording-booth German. Documented as negative result.
- **Manner labels via pYIN+RMS.** Replaced phonemes with a 3-cat acoustic regime label that's domain-robust and aligns frame-for-frame with the WavLM cache.
- **A3 head rejected.** Concat-then-MLP rediscovered speaker shortcut. Manner labels survived as conditioning info for A5.
- **Pseudo-speakers from ECAPA + KMeans, not WavLM-SV.** Architectural-circularity concern + WavLM-SV negative-control failure on URTIC.
- **Honesty audit — linear probes only.** If a feature group needs nonlinearity to predict cold, that's a signal it should be sub-divided, not blessed with hidden capacity.
- **eGeMAPSv02 over ComParE_2016.** 88 features vs 6,373 — fits comfortably under per-group dimensionality ceilings and the set is curated/physiologically motivated, not kitchen-sink.
- **G3/G6 carved by name prefix at runtime.** If we want to re-carve we don't re-run openSMILE.
- **F0 in a separate extractor, not folded into manner.py.** Manner cache is validated and committed across 19k chunks; mutating it risks breaking a working caching invariant.
- **G4 gain-invariant ablation.** Concern that absolute RMS is recording-gain confound. Ablation showed cols 4–10 alone retain almost all the lift with halved speaker leak. Admit the gain-invariant slice into A5b.
- **G8 dropped from admission pool.** Anti-predictive on devel_val. Documented negative — the original A5 hypothesis from the PDF doesn't hold here.
- **A5b: hard top-K, fixed β, fused-logit τ on train_threshold.** Cleanest paper story; sharpest ablation against A5c if we run it.
- **G5 (modulation spectrogram) added late.** Cross-comparison with the 2017 ComParE submissions surfaced Huckvale's MOD family as the one feature class we hadn't covered with G1-G4/G6. Modulation rate captures cross-frame envelope dynamics (syllable rate, breath pacing) that none of the per-frame static features see. Aggressively aggregated to 64-d so the linear-only probe can't launder speaker identity through capacity (same discipline as G3/G6).
- **A5b diagnostics added (correlations + fused-vector speaker probe).** `sub@1` measures honesty in isolation; it doesn't see redundancy with A2 or with other admitted groups, so K ≥ 2 over highly-correlated groups looks beneficial in the table but does little in fusion. A 4×4 / 5×5 Pearson matrix + per-group argmax-disagreement-vs-A2 makes that visible. Separately, plan.md § 5.5's "speaker probe top-1 ≤ A2 + 1σ" gate is degenerate on a 1-d fused logit — the fused-vector concat probe is the honest operationalisation.

If you've read this far, the rest is just running cells and watching numbers. The methodological scaffold is the contribution — the UAR is the cherry.
