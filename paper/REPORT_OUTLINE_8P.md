# Eight-page report outline — URTIC cold detection

## Recommended paper identity

**Working title:** *When the Number Does Not Transfer: Evaluation-Aware Cold Detection on URTIC*

**Central thesis:** On URTIC, evaluation design and the small number of cold participants mattered more than model complexity. Once model selection and evaluation were separated more carefully, simple complementary acoustic models transferred as well as substantially more complex systems, while several plausible de-confounding and adaptation strategies failed to provide reliable gains.

**Main research questions**

1. Which acoustic representations transfer to unseen URTIC partitions: compact handcrafted features, trajectory features, or foundation-model/CNN representations?
2. How strongly do speaker identity, partition construction, threshold selection, and repeated Development use affect the reported UAR?
3. What worked, what did not, and what should be done differently in a future small-sample speech-health project?

This framing is stronger than a paper centered on the historical WavLM score. It matches the final presentation, incorporates the later audits, and directly addresses the course requirement to discuss lessons learned.

## Hard page budget

The bibliography is part of the eight-page maximum. Do not include an appendix; move only indispensable detail into the main text.

| Pages | Content | Target space |
|---|---|---:|
| 1 | Title, abstract, introduction | 1.0 page |
| 2 | Related work; dataset and problem setup | 1.0 page |
| 3–4 | Evaluation protocol and methodology | 2.0 pages |
| 5–6 | Results and compact ablations | 2.0 pages |
| 7 | Discussion, lessons learned, limitations | 1.0 page |
| 8 | Conclusion, references | 1.0 page |

Aim for roughly 4,500–5,500 words only if using the IEEE two-column template; figures and tables will reduce that substantially. A safer target is about 3,500–4,200 words plus two figures and two tables.

## Detailed section outline

### Abstract — 150–180 words

Use five moves:

1. **Problem:** Detect cold from short speech recordings on the imbalanced URTIC corpus.
2. **Core difficulty:** There are thousands of chunks but only 37 cold participants in each official Train and Development partition; each participant occurs in one health state, so speaker identity is a dangerous shortcut.
3. **Approach:** Compare compact energy/pause and CQT features, path log-signatures, and a HuBERT/eGeMAPS/CNN system, while auditing grouping, thresholding, uncertainty, and Development reuse.
4. **Results:** Best hidden-Test submission was the G4 + path-signature model at 0.6582 UAR; the other complex submissions were close (0.6563–0.6567). Under an evaluation-independent whole-side diagnostic, fixed G4+CQT fusion achieved 0.6683 bidirectional mean UAR at threshold zero and 0.6787 with fitting-side-only OOF threshold transfer.
5. **Lesson:** Model complexity did not overcome limited subject count and protocol sensitivity; constrained fusion and honest evaluation were more valuable.

Do not put the historical pseudo-grouped WavLM 0.7111 in the abstract. A later audit showed that its within-Development boundary was not speaker-proxy disjoint.

### 1. Introduction — about 0.65 page

**Paragraph 1: clinical/technical motivation.** Voice changes with upper-respiratory infection, making speech a possible low-cost, non-contact health signal. State clearly that this is a benchmark study, not evidence of a clinically deployable diagnostic.

**Paragraph 2: task difficulty.** URTIC has 28,652 chunks from about 630 participants, around 9.5–10.5% cold chunks, and 210 participants per official partition. The effective sample size is closer to 37 cold participants per Train/Development partition than to 9,500 chunks. Because URTIC is cross-sectional and each participant has only one health label, identity and health are structurally entangled.

**Paragraph 3: project turning point.** The midterm’s strongest internal estimate around 0.70 produced 0.6205 on hidden Test. This motivated a shift from “find a stronger model” to “make the estimate mean something.”

**Paragraph 4: research questions and contributions.** State the three RQs above and three contributions:

- a controlled comparison of three substantially different acoustic pipelines;
- an evaluation audit showing why apparently speaker-disjoint pseudo-group splits can still fail;
- a scoped account of successful and unsuccessful modelling choices.

### 2. Related work / state of the art — about 0.55 page

Keep this focused; it is not a generic survey of speech foundation models.

**2.1 ComParE 2017 Cold task.** Introduce the official ComParE functional, bag-of-audio-words, and CNN/LSTM baselines and the official hidden-Test reference of 0.710 UAR. Be precise: the 0.710 result in the organizer paper is the majority-vote late fusion of all baseline systems, not the standalone 6,373-dimensional openSMILE+SVM system.

**2.2 Modern representations.** Briefly motivate frozen HuBERT/WavLM representations as data-efficient alternatives to full fine-tuning. Mention layer-wise acoustic/speaker information only to motivate why feature choice and speaker leakage need auditing.

**2.3 Speaker confounding and domain shift.** Explain grouped evaluation, speaker representations such as ECAPA-TDNN, and the attraction of contrastive/adversarial de-confounding or unsupervised domain adaptation. End with the gap: on a corpus with few cold subjects and missing public speaker IDs, the validity of the evaluation groups is itself an empirical question.

Use approximately 7–10 references total in the whole report: Schuller et al. (ComParE 2017), two representative 2017 Cold submissions, HuBERT, WavLM, ECAPA-TDNN, supervised contrastive learning, DANN, and eGeMAPS/openSMILE if space permits.

### 3. Dataset, metric, and evaluation design — about 0.8 page

This section is load-bearing and should precede the models.

**3.1 Dataset.** Report official partition sizes: Train 9,505, Development 9,596, Test 9,551 chunks; 970/1,011 cold chunks in Train/Development; 210 participants per partition, of whom 37 are cold and 173 non-cold. Audio is German, 16 kHz, and split into short chunks. Health state comes from a binarized self-report measure.

**3.2 Metric.** Define

\[
\mathrm{UAR}=\tfrac{1}{2}(\mathrm{Recall}_{cold}+\mathrm{Recall}_{non-cold}).
\]

Explain why UAR is appropriate for imbalance but does not fix threshold calibration or reveal the operating point. Accuracy, precision, F1, and both class recalls should be secondary metrics.

**3.3 Evaluation evolution.** Present this as a compact three-stage protocol history:

1. **Initial protocol:** multiple custom splits and repeated use of a nominal Development holdout; uncertainty was often measured over model seeds.
2. **Pseudo-speaker correction attempt:** ECAPA-TDNN embeddings, side-local clustering, and grouped folds. ECAPA contains reproducible speaker-correlated structure, but transferring Train-centroid labels to Development was later found invalid for within-Development disjointness: 201/210 Development-local groups crossed the canonical boundary and 97.95% of recordings belonged to a crossing group.
3. **Defensible final checks:** make all choices inside the fitting side, then evaluate on the untouched whole official side (Train→Development and Development→Train); use repeated grouped OOF estimates only inside the fitting side for threshold calibration and group/cluster bootstrap for uncertainty.

Important nuance: the overlap audit retracts the old “speaker-disjoint” wording but does not justify subtracting a numerical leakage penalty from the historical scores.

### 4. Methodology — about 1.2 pages

Organize by the three final-presentation model families, not by every notebook experiment.

**4.1 Compact handcrafted fusion (Submissions 1–2).**

- **G4:** seven gain-invariant energy/pause descriptors derived from RMS and voiced/unvoiced/silence segmentation.
- **G9:** 168 CQT summary features (mean and standard deviation over 84 logarithmic-frequency bins).
- Each branch: StandardScaler + class-balanced logistic regression.
- Convert each branch’s training logits to z-scores, average with fixed equal weight, and compare fixed-zero versus fitting-side OOF-calibrated threshold policies.
- Motivation: complementary temporal/energy and low-frequency spectral information with no learned fusion weight.

**4.2 Path log-signatures (Submission 3).**

- Construct a trajectory from 20 MFCCs plus normalized time.
- Use depth-2 log-signatures over the full clip and two dyadic halves (693 dimensions total).
- Fit logistic regression and average its score with G4.
- Motivation: retain direction and ordering of acoustic change that global pooling discards.

**4.3 Global/local deep system and domain adaptation (Submissions 4–5).**

- Global view: HuBERT layer-3 embeddings plus eGeMAPS, PCA, class-balanced Ridge.
- Local view: lightweight CNN over a three-channel Log-Mel/MFCC/ΔMFCC image.
- Calibrated score fusion; Submission 5 adds diagonal CORAL using unlabeled Test features.
- State clearly that this is transductive unsupervised adaptation and uses no Test labels.

**4.4 Exploratory methods and negative controls.** Use one compact paragraph or table row, not separate subsections. Mention that the project also tested frozen WavLM/HuBERT heads, learned fusion/stacking, waveform or embedding mixup, speaker-masked contrastive learning, adversarial speaker heads, calibration variants, and test-time augmentation. Their outcomes belong mainly in Results/Discussion.

### 5. Results — about 1.5 pages

#### 5.1 Hidden-Test submissions

Use one table with the final-presentation values:

| Submission | Main idea | UAR | Accuracy | F1 |
|---|---|---:|---:|---:|
| 1 | G4+CQT, fixed boundary | 0.6280 | 0.6498 | 0.5078 |
| 2 | G4+CQT, calibrated boundary | 0.6220 | 0.6952 | 0.5277 |
| 3 | G4+path signatures | **0.6582** | 0.5575 | 0.4676 |
| 4 | HuBERT/eGeMAPS + CNN | 0.6563 | 0.7321 | 0.5589 |
| 5 | Submission 4 + diagonal CORAL | 0.6567 | 0.7708 | **0.5800** |

Interpretation:

- All five submissions lie in a narrow 0.036 UAR band; none exceeds the 2017 0.710 reference.
- Very different systems achieve nearly the same UAR.
- Submissions 3 and 5 have nearly identical UAR but very different accuracy/F1, illustrating that UAR alone hides operating-point behavior.
- CORAL did not materially change UAR but improved accuracy/F1 on the hidden Test set; describe this as an observed operating-point change, not a proven causal benefit unless the exact threshold/calibration path supports it.

#### 5.2 Corrected, evaluation-independent diagnostics

Use a small second table or paired bar chart:

| Protocol/result | G4 | CQT | Fixed equal fusion |
|---|---:|---:|---:|
| Repeated Train-only grouped CV | 0.601±0.005 | 0.610±0.022 | **0.623±0.012** |
| Train→Development, threshold 0 | 0.642 | 0.642 | **0.690** |
| Development→Train, threshold 0 | 0.613 | **0.668** | 0.647 |
| Bidirectional mean, threshold 0 | 0.628 | 0.655 | **0.668** |

Then report that fitting-side-only OOF threshold transfer raised the bidirectional fusion mean from 0.6683 to 0.6787 (+0.0104), while its proxy-cluster confidence interval crossed zero. Therefore calibration is a modest expected-score choice, not a statistically established improvement.

The directional asymmetry is informative: fusion wins strongly for Train→Development but CQT alone wins for Development→Train. The correct conclusion is complementarity plus distribution sensitivity, not universal fusion dominance.

#### 5.3 What did not work

Use a compact “intervention / observation / lesson” table with at most five rows:

- Historical learned WavLM system: a 0.7111 pseudo-grouped Development point was invalid as a speaker-disjoint headline after the held-side grouping audit.
- Corrected Train-only outer CV: WavLM-Large 0.552 and HuBERT-Large 0.564 versus G4 0.592; model size did not guarantee performance.
- Learned logistic stacking: destroyed G4+CQT complementarity on a pessimistic split; fixed equal fusion was substantially better on the same split.
- Test-time augmentation: the tested recipe reduced UAR from 0.7090 to 0.6683 under the historical protocol.
- Mixup, speaker-masked contrastive learning, and DANN-style adversaries: no reliable joint improvement in cold UAR and the historical speaker-probe gate; keep this conclusion scoped to the tested frozen-backbone/proxy-label setup.

Do not reproduce the full ablation ladder. The lesson matters more than every intermediate number.

### 6. Discussion: what we learned — about 0.75 page

Make this the strongest section; it is explicitly requested by the course.

**What worked**

- Separating model selection, threshold selection, and final evaluation.
- Fixed equal fusion of low-capacity, complementary features; it avoided fitting unstable fusion weights.
- Path signatures as an orthogonal representation of temporal evolution; this produced the best hidden-Test UAR.
- Whole-official-side transfer as a check that does not depend on inferred evaluation groups.
- Negative controls, repeated partitions, and reporting both class recalls rather than only one UAR.

**What did not work, and why**

- Reusing Development for architecture, threshold, and fusion decisions made it cease to be a holdout.
- Seed standard deviation measured optimization variability, not variation across unseen subjects; partition variability was much larger.
- A speaker embedding can be valid while transferred cluster labels are invalid. Side-local ECAPA clustering was stable, but Train-centroid assignment fragmented held-side structure.
- Complex representations and learned stacking had too much flexibility relative to the effective sample size.
- Intuitive de-confounding methods failed when their controls exposed splice artifacts, bottleneck effects, memorization, or proxy-label problems.

**Health-ML implication.** Thousands of correlated clips do not replace independent participants. Report subject-aware uncertainty and class-specific behavior before claiming clinical value.

**Limitations**

- Public true speaker IDs are unavailable; ECAPA clusters remain proxies.
- Cold status is cross-sectional and based on a binarized self-report item; identity, session, and health cannot be causally separated.
- Development had substantial historical selection exposure.
- Only five hidden-Test submissions were permitted and Test labels remain unavailable, limiting statistical interpretation.
- Results concern URTIC and the tested configurations; they do not establish that foundation models or de-confounding methods fail generally.
- No clinical calibration, prospective cohort, or external dataset validation was performed.

### 7. Conclusion — about 0.2 page

Use three sentences:

1. The best hidden-Test system reached 0.6582 UAR, and three very different pipelines converged to a similar performance band.
2. The main project contribution was learning that protocol validity, subject count, and operating-point choice dominated architectural sophistication; fixed complementary fusion survived the strongest evaluation-independent checks.
3. Future work should obtain verified participant IDs or longitudinal recordings, freeze the full pipeline before whole-side evaluation, and validate on an external cohort before making health claims.

## Recommended visual package

Limit the report to four visual elements.

1. **Figure 1 — Dataset/evaluation problem:** 28,652 chunks → about 630 participants → 37 cold participants per official Train/Development partition; show why chunk count is not effective sample size.
2. **Figure 2 — Compact method diagram:** three parallel families (G4+CQT; G4+signatures; HuBERT/eGeMAPS+CNN+CORAL) leading to five submissions.
3. **Table 1 — Hidden-Test submissions:** the five-row table above.
4. **Table 2 — Corrected evidence and lessons:** either the official-side result table or a combined worked/did-not-work table. If space is tight, prefer the official-side table and discuss failures in prose.

Do not include UMAP plots, the 25-layer audit, full beta sweeps, all de-confounding architectures, per-seed tables, or a separate appendix. They distract from the eight-page story.

## Claim guardrails

- Say **“speaker-correlated proxy groups”**, not true speakers, whenever discussing ECAPA clusters.
- Do not call the historical within-Development split speaker-disjoint.
- Do not describe 0.7111 as expected Test performance or compare it inferentially with the 2017 0.710 hidden-Test baseline.
- Do not apply a numerical “leak correction”; the overlap audit invalidates the wording, not a known number of UAR points.
- Identify 0.710 correctly as the organizer’s late-fusion hidden-Test baseline.
- Do not claim a universal feature ceiling or that deep models “do not work.” Say that the tested complex models did not reliably outperform compact models under the corrected protocol.
- Separate hidden-Test submission results from later Train↔Development diagnostics; they use different models and evaluation procedures.
- Report both cold and non-cold recall when discussing health relevance or operating points.

## Source map for writing

- Final narrative and hidden-Test table: `presentation/ML4Health_Group3_Final_TUM_version_2.pdf`
- Dataset and 2017 baselines: `../schuller17_interspeech.pdf`
- Course-required paper structure: `../ML4Health_01_Introduction.pdf`, evaluation slide
- Evaluation audit: `results/audit_evaluation_protocol.json`
- Held-side grouping correction: `results/shipped_group_overlap_audit.json` and `results/manuscript_claim_audit.md`
- Repeated Train-only G4/CQT evaluation: `results/fixed_g4_g9_repeated_cv.json`
- Whole-official-side evaluation: `results/eval_independent_official_split_fusion.json`
- Threshold transfer: `results/eval_independent_threshold_policy.json`
- Final architecture reconciliation: `results/reconciled_architecture_recommendation.md`
- Corrected foundation-model comparison: `results/corrected_outer_cv_foundations.json`

## What to do with the existing 24-page draft

Do not try to shrink it line by line. Rebuild the eight-page report around this outline and reuse only selected paragraphs, citations, and figures. The current draft over-centers the historical A2→A2.5→A5b WavLM ladder, contains a very large appendix, and underrepresents the path-signature and HuBERT/CNN/CORAL systems that actually formed the final presentation and hidden-Test submissions.
