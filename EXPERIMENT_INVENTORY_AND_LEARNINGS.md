# Experiment inventory and lessons learned

This document reconstructs the experiments that are supported by code, result
files, or the final presentation. It intentionally records positive, negative,
and invalidated results. The point is not to pretend that every experiment was
confirmatory; it is to document what was attempted, what was learned, and which
claims remain defensible after the evaluation audit.

## Short answer: the two external ECAPA datasets

The ECAPA-TDNN speaker-grouping pipeline was checked on two labeled corpora
outside URTIC/ComParE:

1. **LibriSpeech English development audio** (`libri_en_dev`): 600 recordings
   from 40 known speakers. ECAPA followed by L2 normalization and KMeans at the
   true number of speakers achieved **ARI 0.996** and **purity 0.998**.
2. **Multilingual LibriSpeech German (MLS German)** (`mls_de`): 741 recordings
   from 56 known speakers. The same pipeline achieved **ARI 0.963** and
   **purity 0.972**.

These controls validate ECAPA as a representation that contains strong speaker
structure. They do **not** prove that every cluster fitted or transferred on
URTIC is a true identity. That distinction became central later.

TRILLsson1 and TRILLsson5 were also evaluated on both controls. Their recovery
was weaker, especially on German:

| Representation | LibriSpeech ARI | MLS German ARI | Interpretation |
|---|---:|---:|---|
| ECAPA | **0.996** | **0.963** | Strong known-speaker recovery |
| TRILLsson1 | 0.834 | 0.488 | Useful identity-like structure, but noisy |
| TRILLsson5 | 0.799 | 0.552 | Better than TRILLsson1 on German, still well below ECAPA |

## How to read the evidence

The project accumulated results under several protocols. They should not be
mixed into one ranking.

- **Tier A -- hidden Test evidence:** the five submitted prediction files and
  their returned Test metrics. These numbers remain valid observations.
- **Tier B -- corrected diagnostics:** repeated Train-only grouped CV and
  Train-to-Development / Development-to-Train whole-side transfer. These are
  the strongest post-audit internal checks.
- **Tier C -- historical exploratory results:** experiments on the old
  `devel_val/devel_test` boundary. The models were genuinely run, but the
  boundary was not speaker-disjoint under independently fitted
  Development-local ECAPA groups. These experiments are useful for generating
  hypotheses and understanding mechanisms, but their absolute UAR values must
  not be presented as speaker-disjoint performance estimates.
- **Tier D -- label-free or external controls:** representation stability,
  external speaker recovery, augmentation detectability, and matched control
  experiments. Many of their qualitative lessons survive the health-label
  evaluation failure because they do not depend on the invalid boundary.

## The evaluation-pipeline audit: exact findings

The evaluation problem was not a mistake in the UAR formula. The stored audit
reproduced the pipeline UAR exactly with scikit-learn balanced accuracy. The
submission file was also structurally valid: 9,551 unique Test rows, the correct
file set, and only valid labels. The failure was in how the internal estimate
was constructed and reused.

### 1. The nominal holdout stopped being a holdout

Of 79 audited experiment code cells, 45 referenced `devel_test`; 42 of those
contained comparison or decision markers. This is a static source count rather
than an execution count, but it demonstrates that the same subset was repeatedly
used to compare candidates. It could no longer provide a one-shot estimate of
generalization.

A separate forking-path simulation over 11 recorded fusion candidates estimated
mean selection optimism of about 0.0026 UAR and a 95th-percentile value of
0.0086. Thus repeated candidate choice contributed optimism, but it was too
small to explain the approximately 0.09 midterm Development-to-Test collapse by
itself.

### 2. The Development boundary was not speaker-disjoint

The shipped group map was fitted on Train and transferred to Development by
nearest Train centroid. It reported zero group overlap across the historical
Development halves because those same transferred IDs had constructed the
split. Under an independently fitted Development-local ECAPA partition, the
canonical split had 201/210 groups on both sides and 9,399/9,596 affected
recordings (97.95%). Across 11 split seeds, 191--202 groups crossed and
94.40--97.95% of recordings belonged to a crossing group.

The equivalent Train split remained disjoint because the map had actually been
fitted on Train. A positive-control Development split made from
Development-local groups also produced zero overlap. This localized the bug to
held-side ID transfer, rather than to GroupKFold itself.

### 3. Fusion and threshold selection had an effective sample of three cold groups

The Train threshold subset contained 973 chunks and 21 pseudo-speaker groups,
but only **three cold pseudo-speaker groups**. Both fusion weight beta and
decision threshold tau were selected from this subset. The apparent number of
chunks therefore overstated the amount of independent minority-class evidence
used for calibration.

### 4. The speaker probe measured the wrong task

The original probe trained on Train identities and evaluated on a different
official speaker pool labeled by nearest Train centroid. On matched held chunks
from the same identity pool, top-1 recovery was 0.0877. Across the different
official pool it was only 0.0212, a **4.14-fold understatement**, and was even
below the 0.0584 majority baseline induced by the transferred labels.

The correct separation is:

- use held chunks from the same identities or pairwise verification to measure
  how much identity information a representation retains;
- use group-disjoint or whole-side evaluation to measure cold-classifier
  generalization.

One cross-pool classification setup cannot validly answer both questions.

### 5. Split and grouping uncertainty dominated model-seed uncertainty

The same historical system ranged from 0.6280 to 0.7037 UAR under four grouping
constructions, a spread of 0.0757. This was 12.6 times the reported model-seed
standard deviation of 0.0060. Shadow-split standard deviation was 0.0169, about
2.8 times model-seed variation.

An illustrative independence calculation gave a naive chunk-i.i.d. 95% UAR
half-width of about 0.0145, versus about 0.0793 when scaled by the number of
subjects. These are illustrations rather than replacement confidence
intervals, but they explain why seed error bars were far too reassuring.

### 6. The final submitted head did not use all labeled data

The submitted WavLM head was fitted on 8,532 chunks, only 44.7% of the 19,101
available labeled Train+Development chunks. The expected challenge workflow was
to lock all choices and then refit, or construct a cross-fold ensemble, using
all Train+Development labels before Test prediction.

### 7. Consequence for the results

- The old WavLM value near 0.7111 is a real output of the historical pipeline,
  but **not** a reliable speaker-disjoint estimate and not a hidden-Test result.
- Zero overlap under the transferred group IDs was circular evidence and cannot
  validate the split.
- The old speaker-probe gate cannot establish that the historical WavLM gains
  were free of identity confounding.
- Small differences between historical fusion candidates are particularly
  vulnerable to Development reuse and calibration on three cold groups.
- The five returned hidden-Test scores, external speaker controls, side-local
  ECAPA stability tests, and corrected whole-official-side transfer diagnostics
  remain reportable with their stated limitations.

### 8. Corrected replacement protocol

1. Perform architecture, feature, fusion-weight, threshold, and epoch selection
   using nested grouped CV inside the fitting side only.
2. Fit proxy groups side-locally when they are needed for grouped fitting-side
   folds; never interpret centroid indices transferred across sides as global
   identities.
3. Freeze the complete pipeline and evaluate once on the untouched whole
   official side. Reverse Train and Development for a second transfer
   direction when used as a diagnostic.
4. Estimate thresholds from fitting-side grouped OOF predictions only.
5. Report outer-fold or proxy-group bootstrap uncertainty, explicitly calling
   the groups inferred proxies; do not use model-seed standard deviation as the
   main uncertainty statement.
6. After all choices are locked, refit or cross-fold-ensemble over all available
   labeled Train+Development data for final Test inference.

## 1. Evaluation and speaker-identity work

### Random chunk splits

An early random chunk-level split produced extremely optimistic validation
performance (one run reached about 0.97 validation UAR while transferring near
0.62). Because many chunks belong to the same person, the split allowed a model
to recognize the talker rather than the cold.

**Learning:** recordings are not independent subjects. Every selection split,
uncertainty estimate, and bootstrap must respect speaker-like groups.

### ECAPA pseudo-speaker construction

The student release contains no true speaker IDs. The project therefore
extracted 192-dimensional SpeechBrain ECAPA-VoxCeleb embeddings from the raw
16-kHz audio and examined KMeans, HDBSCAN, agglomerative, and spectral
clustering. The fixed KMeans prior was 210 clusters per official side, matching
the corpus design.

Side-local ECAPA structure was reproducible:

| Official side | KMeans seed stability, mean ARI | KMeans vs HDBSCAN ARI, all recordings |
|---|---:|---:|
| Train | 0.889 | 0.856 |
| Development | 0.897 | 0.829 |
| Test | 0.888 | 0.879 |

The HDBSCAN-non-noise sensitivity values were approximately 0.930, 0.929, and
0.944, but they exclude 2.4--3.3% of recordings and therefore must always be
reported with coverage.

**Learning:** ECAPA contains stable identity-like structure on each official
side, but a proxy cluster is still not a ground-truth person.

### External ECAPA recovery

The identical ECAPA -> L2 -> KMeans-at-true-k pipeline was applied to the two
known-speaker controls described above: LibriSpeech English and MLS German.
Shuffled-label negative controls collapsed to approximately zero ARI.

**Learning:** the strong URTIC clustering is not merely a KMeans artifact; the
same representation and clustering recipe recover known speakers on separate
English and German datasets.

### Alternative speaker embeddings

- **WavLM-base-plus for speaker verification:** tested as an architecturally
  related negative control. KMeans and HDBSCAN agreed poorly on URTIC
  (approximately ARI 0.093, with substantial HDBSCAN noise), so it was not used
  for the production proxy map.
- **TRILLsson1 and TRILLsson5:** tested on the two labeled external controls.
  TRILLsson1 was additionally extracted for all 28,652 URTIC recordings and
  clustered side-locally. Its seed stability was 0.591/0.549/0.603 for
  Train/Development/Test, and its ARI against ECAPA was 0.566/0.516/0.554.

**Learning:** an independent paralinguistic representation corroborated that
ECAPA was finding repeatable structure, but TRILLsson was the noisier identity
view. It should neither replace ECAPA nor be fused into the cold classifier.

### The transferred-ID failure

The original pipeline fitted 210 ECAPA centroids on Train and assigned
Development recordings to those Train centroids. This produced labels that
looked non-overlapping by construction. When the old Development split was
audited with an independently fitted Development-local grouping, **201 of 210
groups crossed the boundary and 9,399 of 9,596 recordings belonged to a
crossing group**.

**Learning:** validating the embedding does not validate a transferred cluster
ID. Cluster labels are local coordinate systems; a Train centroid index is not
a globally meaningful speaker identity on another corpus side.

### Grouping and uncertainty variants

The project also tried KMeans seed sweeps, k in roughly 100/210/420 regimes,
pooled Train+Development clustering, side-local offset labels, shadow split
seeds, group-level bootstrap intervals, and a prepared blinded 300-pair human
same/different-speaker audit.

**Learning:** grouping choices can move the reported number more than a model
change. Group-level uncertainty is more honest than chunk bootstrap, but it
remains proxy-group uncertainty. The human audit interface was prepared; it is
not evidence unless two annotations are actually completed and frozen.

## 2. Compact and classical acoustic features

The project extracted or tested the following feature families:

- **G1 voicing:** voiced/unvoiced behavior and related durations.
- **G2 prosody:** pitch and timing summaries.
- **G3 voice quality:** eGeMAPS-derived jitter, shimmer, HNR, and spectral-tilt
  measures.
- **G4 energy and pause:** RMS behavior, low-energy ratio, energy slope,
  voiced/unvoiced/silence structure, and pause statistics. Four absolute
  loudness variables were removed to form the final seven-dimensional
  gain-invariant subset.
- **G5 modulation:** temporal modulation-spectrum summaries.
- **G6 spectral shape:** low-order MFCC and spectral-flux eGeMAPS slices.
- **G8 OOD Mahalanobis:** distance from the healthy training distribution.
- **G9 CQT:** mean and standard deviation of 84 constant-Q bins, for 168
  dimensions.
- **Full eGeMAPSv02:** all 88 functionals.
- **Full ComParE-2016:** 6,373 openSMILE functionals.

The initial honesty audit measured both cold predictiveness and recoverable
proxy-speaker information. G4 was the strongest compact group, but absolute
gain variables were confounded; the gain-invariant slice retained much of the
cold signal with less identity/recording-condition information. G8 was
anti-predictive and was rejected. Full eGeMAPS and full ComParE generally
underperformed carefully selected low-dimensional slices and carried much more
speaker-identifiable information.

Under corrected Train-only nested grouped CV, individual families varied
strongly by fold: G4 achieved OOF UAR 0.592, CQT 0.583, G5 0.537, eGeMAPS
0.504, the small raw-acoustic signature 0.510, and HeAR 0.533. The precise
ranking is not stable enough to support a universal feature hierarchy.

**Learning:** on this sample size, more dimensions often dilute rather than add
signal. Compact, physiologically interpretable groups were easier to
regularize, audit, and combine.

## 3. Compact G4+CQT systems

Each branch used StandardScaler and class-balanced logistic regression. Branch
scores were standardized with fitting-side statistics and averaged with fixed
equal weights. The project deliberately avoided a learned fusion weight after
seeing its sign and magnitude change across folds.

Two hidden-Test variants were submitted:

| Submission | Threshold policy | Test UAR | Accuracy | F1 |
|---|---|---:|---:|---:|
| 1 | Fixed boundary | 0.6280 | 0.6498 | 0.5078 |
| 2 | Historically calibrated boundary | 0.6220 | 0.6952 | 0.5277 |

Post-audit corrected checks found:

- Six repetitions of Train-only grouped CV: G4 0.601, CQT 0.610, fixed fusion
  0.623 mean UAR.
- Train -> Development: G4 0.6422, CQT 0.6417, fusion 0.6897.
- Development -> Train: G4 0.6132, CQT 0.6679, fusion 0.6468.
- Bidirectional mean: G4 0.6277, CQT 0.6548, fusion 0.6683.
- A threshold estimated only from fitting-side grouped OOF predictions raised
  the bidirectional fusion mean to 0.6787, but its paired proxy-group interval
  crossed zero relative to the fixed boundary.

**Learning:** G4 and CQT are complementary, but fusion is not universally
better than the best branch in every transfer direction. Equal fusion is a
low-variance policy; threshold selection changes the cold/non-cold operating
point and must stay inside the fitting side.

## 4. Path signatures

Two different signature experiments must be kept separate.

### Submitted MFCC trajectory signature

Submission 3 formed a 21-dimensional path from 20 MFCC channels plus normalized
time. A depth-2 log-signature was computed on the full utterance and both
dyadic halves, producing 693 dimensions. Its class-balanced logistic-regression
score was equally fused with G4.

This was the team's best hidden-Test UAR:

| Submission | Test UAR | Accuracy | F1 |
|---|---:|---:|---:|
| G4 + 693-d MFCC path signature | **0.6582** | 0.5575 | 0.4676 |

**Learning:** order-sensitive trajectory information was genuinely useful and
transferred to the hidden Test set. Splitting the path into halves retained
coarse temporal localization that whole-utterance pooling loses.

### Small raw-acoustic signature candidate

A separate later experiment constructed a 10-dimensional depth-2 signature
from a four-channel acoustic path and tested it as another candidate in the
WavLM late-fusion ladder. It was speaker-light but weak for cold prediction and
did not improve the existing fusion.

**Learning:** “path signature” is not one universal representation. The choice
of path channels, dimensionality, and time partitioning was load-bearing. This
negative result does not invalidate the successful 693-dimensional MFCC
signature submission.

## 5. Foundation-model and health-embedding experiments

### WavLM-Large pooled-statistics track

The main WavLM track tested:

- frozen WavLM-Large features;
- mean pooling and then per-layer mean/std/skew/kurtosis pooling;
- uniform versus learned layer weights;
- an “honesty prior” that initialized layer weights from cold signal minus
  speaker recoverability;
- small MLP heads, learning-rate and dropout variants;
- three- and five-seed runs;
- late fusion with audited handcrafted groups.

Historically, the honesty-prior initialization and G4/G5 fusion produced large
improvements on the old Development split, including values near 0.70--0.71.
The audit later showed that this boundary was not speaker-disjoint, so those
numbers are exploratory and cannot be the paper headline.

The mechanism ablations remain informative with caution: the honesty prior led
optimization to a different layer-weight solution than uniform initialization;
raising the learning rate alone did not reproduce it. A pooling ablation found
that mean/std/skew/kurtosis did not clearly raise cold UAR relative to fewer
moments, but reduced proxy-speaker recoverability under the old probe.

Under corrected Train-only nested outer CV, the tested WavLM-Large head reached
about 0.552 OOF UAR and adding G4 did not produce a stable improvement.

**Learning:** frozen SSL embeddings carry cold information, but model capacity
does not overcome subject scarcity. Any layer-selection or de-confounding claim
must be re-evaluated under the corrected grouping if it is to be central.

### HuBERT variants

The project tested several HuBERT uses:

- HuBERT Base mean-pooled across layers;
- HuBERT Base with the same learned layer-weight/honesty-prior head used for
  WavLM;
- HuBERT Large pooled-statistics heads;
- HuBERT Base layer 3 as the global branch in Akshat's submitted system.

On the old exploratory WavLM-fusion protocol, mean-pooled HuBERT Base was weak,
whereas the learned layer-weight head improved its standalone result but added
little because its logits were highly correlated with WavLM. Under corrected
Train-only outer CV, HuBERT Large achieved about 0.564 OOF UAR; the folds ranged
widely, again showing subject-selection variance.

**Learning:** a second foundation model only helps fusion if it contributes an
orthogonal error pattern. A stronger standalone representation can still be a
bad fusion member when it reproduces the anchor's ranking.

### WavLM-base-plus and HeAR

- WavLM-base-plus with learned layer weighting was tested as a smaller SSL
  alternative and as a speaker-verification negative control.
- Google's HeAR health-acoustic embedding was tested with a logistic head and
  as an additional branch over G4+CQT. It was speaker-light, but did not add a
  stable marginal improvement over the compact handcrafted system.

**Learning:** a health-specific embedding is not automatically better on a
particular small health dataset. Pretraining relevance cannot substitute for
an evaluation showing complementary signal.

## 6. Akshat's global/local multi-view system

The global stream concatenated pooled HuBERT Base layer-3 features (4,608-d)
and eGeMAPSv02 (88-d), applied a variance filter, StandardScaler, PCA to 256
dimensions, and used a balanced Ridge classifier. The local stream stacked a
log-mel spectrogram, MFCCs, and delta-MFCCs into a three-channel acoustic image
and trained a lightweight Conv2D/BatchNorm/GELU/dropout/global-pooling head.
The two standardized scores were equally fused.

Submission 5 added diagonal CORAL to align feature means and marginal
variances using unlabeled target recordings.

| Submission | System | Test UAR | Accuracy | F1 |
|---|---|---:|---:|---:|
| 4 | HuBERT/eGeMAPS Ridge + acoustic CNN | 0.6563 | 0.7321 | 0.5589 |
| 5 | Submission 4 + diagonal CORAL | 0.6567 | **0.7708** | **0.5800** |

**Learning:** the global and local views transferred much better than the most
optimistic historical WavLM estimate suggested. CORAL barely changed UAR but
shifted the operating behavior toward higher accuracy and F1. Because the
comparison also includes a finite Test sample and a fixed threshold, it should
be described as an observed operating-point change, not proof that CORAL always
helps.

## 7. Phonetic and manner-aware experiments

### wav2vec2 phoneme-CTC labels

The project ran `wav2vec2-xlsr-53-espeak-cv-ft` over all 19,101 Train and
Development recordings. Approximately 84% of frames selected the blank token.
Soft-probability aggregation confirmed that the model was confidently
retreating to blank rather than merely producing diffuse uncertainty. A
blank-smearing heuristic was rejected because URTIC has no phoneme-boundary
ground truth with which to validate it.

**Learning:** the CommonVoice/IPA phoneme model did not transfer cleanly to the
URTIC German recordings. Heuristic repair would have created unmeasurable,
phoneme-dependent bias.

### Acoustic manner labels

The phoneme path was replaced by pYIN voicing and an RMS silence gate, yielding
voiced, unvoiced, and silence categories. WavLM statistics were pooled by these
categories and fed through a two-stream head. The historical three-seed result
was slightly worse than the matched WavLM baseline and slightly more
speaker-predictive.

**Learning:** a sensible intermediate representation can still fail because
per-utterance category pools are too noisy or redundant with the frozen model.
The pYIN/RMS information remained useful in the compact G4 energy/pause branch,
even though manner-aware WavLM pooling failed.

## 8. Data-level de-confounding and augmentation

### Cross-speaker waveform splicing

Waveforms from different pseudo-speakers with the same cold label were spliced
together with crossfades. Before training, a detector was asked to distinguish
original from spliced WavLM representations. It achieved about **0.998 UAR**.

A matched self-splice control used partners from the same pseudo-speaker and
still achieved about **0.990 UAR**.

**Learning:** the splice operation itself created the artifact; cross-speaker
mixing was not the main cause. A self-splice control should precede training
whenever waveform splicing is used with a sensitive SSL encoder.

### Embedding mixup

The project pivoted to mixing cached WavLM pooled statistics between
same-class, different-pseudo-speaker examples.

- Conservative mixing (roughly 15--30% partner contribution) gave a small
  historical UAR change but did not reduce the matched speaker probes.
- Aggressive mixing (roughly 30--50% partner contribution) reduced performance
  and increased seed variance before producing meaningful de-confounding.

**Learning:** there was no useful mixing-strength window in the tested range.
Weak mixing did not change the representation enough; strong mixing damaged
label-valid cold structure first.

## 9. Contrastive-learning experiments

The project tested a speaker-masked supervised contrastive pipeline on a
4096 -> 512 -> 128 projection. Positives shared the cold label but came from
different pseudo-speakers. It also ran three matched controls:

1. an untrained random projection with the same 128-dimensional bottleneck;
2. cold cross-entropy only;
3. vanilla supervised contrastive learning without speaker masking.

The initial contrastive run appeared to reduce a linear speaker probe, but the
random projection produced almost the same reduction. Cold-CE alone was better
than the speaker-masked contrastive objective on cold separation and speaker
probes. Vanilla SupCon was not worse than the masked variant.

A second experiment trained `cold CE + lambda * speaker-masked SupCon` for
lambda in 0, 0.05, 0.1, 0.25, and 0.5. Lambda zero -- pure cold CE -- was best;
larger contrastive pressure reduced class margin and did not improve the speaker
probe.

**Learning:** the apparent de-confounding came from the dimensionality
bottleneck, not the contrastive objective. Any de-confounding comparison that
changes representation dimensionality needs a matched random-projection
control. In this small-data, frozen-backbone setting, supervised contrastive
pressure was redundant or subtractive relative to cross-entropy.

## 10. Adversarial speaker removal

The executed adversarial experiments used DANN-style gradient reversal with a
210-class pseudo-speaker discriminator. They included:

- a 128-dimensional head-only/layer-weight-open setup;
- discriminator-capacity ceilings with linear and deeper MLP probes;
- an unbottlenecked 4096-dimensional substrate;
- a corrected low-lambda sweep with regularization and per-epoch discriminator
  monitoring.

The discriminator could nearly memorize the training chunks while transferring
little speaker classification to Development. In the corrected unbottlenecked
sweep, cold UAR and speaker probes were essentially flat across adversarial
weights from 0 to 0.1.

**Learning:** the adversary had no stable, generalizable pseudo-speaker signal
against which to learn a useful invariant representation. More discriminator
capacity increased memorization rather than useful pressure. This is also a
warning that pseudo-speaker probe accuracy has a measurement floor and can be
dominated by chunk fingerprints.

The original plan mentioned **MDD** and an MDD-vs-DANN comparison. The stored
results support DANN/gradient-reversal experiments, not a completed distinct
MDD implementation. The report should not claim that MDD itself was evaluated.

## 11. Fusion, calibration, ensembling, and inference variants

The project tried substantially more than a single average:

- fixed equal score averaging;
- fitted beta weights and extended beta sweeps;
- regularized logistic stacking;
- direct UAR grid-search weighting;
- mean probability versus mean logit across five seeds;
- isotonic calibration;
- fixed zero versus fitting-side OOF thresholds;
- K=1, K=2, and K=3 candidate additions;
- multi-K and five-/ten-seed ensembles;
- hyperparameter-diverse ensemble members;
- speaker-level logit smoothing;
- test-time augmentation with +/-2 dB gain and +/-2% time stretch;
- diagonal CORAL domain adaptation in the submitted multi-view system.

The historical WavLM experiments found mean-logit averaging more stable than
learned stacking. Logistic stacking learned large weights with sign flips;
direct UAR grid search selected only a few seeds and did not transfer; isotonic
calibration could not improve rank-based threshold-swept UAR; adding weaker,
hyperparameter-diverse models hurt rather than diversified beneficially.

Speaker-level smoothing was strongly harmful in the historical shadow-split
tests. Cold evidence varies across chunks from one recording session, so
averaging all chunks toward one speaker score removed local cough, breathiness,
or throat-clearing evidence. The small number of cold people also made
speaker-level UAR much noisier than chunk-level UAR.

For TTA, gain perturbations were largely absorbed by the foundation model's
per-utterance input normalization, while time stretching shifted the features
out of the trained distribution and reduced performance. The combined TTA
ensemble was worse than the unaugmented ensemble.

**Learning:** on a small calibration set, every fitted weight or threshold is
another opportunity to overfit. Fixed equal weights, fitting-side-only
thresholds, and simple mean-logit ensembles were the most reliable policies.

## 12. Final hidden-Test submissions

| Sub. | Model | UAR | Accuracy | Precision | F1 |
|---:|---|---:|---:|---:|---:|
| 1 | G4+CQT, fixed boundary | 0.6280 | 0.6498 | 0.5467 | 0.5078 |
| 2 | G4+CQT, calibrated boundary | 0.6220 | 0.6952 | 0.5484 | 0.5277 |
| 3 | G4 + 693-d MFCC path signature | **0.6582** | 0.5575 | 0.5538 | 0.4676 |
| 4 | HuBERT/eGeMAPS global + acoustic CNN local | 0.6563 | 0.7321 | 0.5659 | 0.5589 |
| 5 | Submission 4 + diagonal CORAL | 0.6567 | **0.7708** | **0.5746** | **0.5800** |

The UAR spread across all five submissions was only 0.0362 despite major
differences in representation and complexity. Submission 3 had the best UAR;
Submission 5 had the best accuracy, precision, and F1.

**Learning:** UAR alone hides materially different operating behaviors. Model
complexity was not the limiting factor; the effective number of independent
cold subjects and the chosen decision policy were.

## 13. Ideas discussed but not established as completed experiments

The repository distinguishes executed experiments from future ideas. The
following were proposed, scaffolded, or mentioned, but should not be listed as
completed positive/negative experiments without additional artifacts:

- Fisher-vector/GMM features: mentioned as another team track/future fusion
  candidate, but no completed result artifact was found in this repository.
- HuBERT discrete token histograms and a learned VQ-VAE codebook: deferred.
- Full WavLM/HuBERT transformer fine-tuning: outside the compute/risk budget.
- A distinct MDD implementation and MDD-vs-DANN comparison: planned, while the
  executed adversarial work was DANN-style gradient reversal.
- WebMAUS/Whisper forced alignment: considered before the wav2vec2 CTC and
  acoustic-manner pivot; not established as a completed evaluated branch.
- The blinded human speaker-pair audit: materials were generated, but completed
  independent annotations are not present in the stored results.

## 14. Overall lessons suitable for the report

1. **The effective sample size is people, not chunks.** Thousands of correlated
   audio segments cannot compensate for only 37 cold participants per side.
2. **An embedding can be valid while an ID-transfer procedure is invalid.**
   ECAPA recovered known speakers externally and stable structure side-locally,
   yet Train-centroid labels fragmented Development speakers.
3. **Evaluation design dominated architecture choice.** The largest apparent
   gains and failures traced back to selection boundaries, threshold reuse, or
   group construction rather than a new neural block.
4. **Complementarity mattered more than raw dimensionality.** Seven
   gain-invariant energy/pause features, CQT summaries, MFCC trajectories, and
   local spectrogram texture contributed different information. Full
   high-dimensional functionals were often weaker.
5. **Order can matter.** The MFCC path signature was the best hidden-Test UAR
   system, demonstrating that temporal evolution can survive where global
   pooling loses information.
6. **Complex models changed operating behavior more than UAR.** The
   HuBERT/eGeMAPS/CNN/CORAL system nearly matched the signature UAR while
   producing far higher accuracy and F1.
7. **Negative controls changed the conclusions.** Self-splicing exposed an
   augmentation artifact; random projection exposed a false contrastive
   de-confounding effect; discriminator ceilings exposed memorization.
8. **De-confounding metrics can themselves be confounded.** A lower speaker
   probe after a bottleneck is not evidence of invariance unless dimensionality
   and capacity are controlled.
9. **Simple decision policies were safer.** Fixed equal fusion and fitting-side
   OOF calibration were more stable than learned stacking, seed weighting, or
   repeated target-metric search.
10. **The correct scientific contribution includes what failed.** The project
    learned which conclusions survive independent evaluation, which promising
    mechanisms were artifacts, and why architectural complexity could not
    overcome a subject-limited dataset.

## Primary supporting artifacts

- `presentation/ML4Health_Group3_Final_TUM_version_2.pdf`
- `results/ecapa_recovery_libri_en_dev.json`
- `results/ecapa_recovery_mls_de.json`
- `results/speaker_proxy_trillsson_verdict.md`
- `results/shipped_group_overlap_audit.json`
- `results/speaker_proxy_method_benchmark.json`
- `results/A5a_honesty.csv`
- `results/A6_phase1_PoC_controls.json`
- `results/A6b_phase1_combined_lambda_sweep.json`
- `results/A7c_v2_unbottlenecked.json`
- `results/fixed_g4_g9_repeated_cv.json`
- `results/eval_independent_official_split_fusion.json`
- `results/eval_independent_threshold_policy.json`
- `results/corrected_outer_cv_linear.json`
- `results/corrected_outer_cv_foundations.json`
