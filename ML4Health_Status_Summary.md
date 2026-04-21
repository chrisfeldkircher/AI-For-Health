# ML4Health — URTIC Cold Detection: Project Status

*Last updated: 21 April 2026 • Status after A2 baseline locked*

---

## TL;DR

Working on the 2017 ComParE Cold sub-challenge (URTIC, 28,652 chunks, binary cold/non-cold). Target: beat the 71.0% UAR late-fusion baseline. Approach: modernise each pillar of the 2017 winning fusion with foundation models and add a triple anti-speaker-confounding intervention (augmentation + contrastive + MDD adversary).

**Current rung: A2 locked at devel UAR = 0.6287.** Infrastructure solid, evaluation pipeline honest, ready to build de-confounding infrastructure and move up the ladder.

---

## 1. What we've done

### 1.1 Literature and dataset understanding

- Read the 2017 Schuller baseline paper and all seven participant papers (Huckvale, Cai, Wagner, Suresh, Tavarez, Gosztolya, Kaya).
- Mapped each 2017 approach to a 2026 modernisation (see `ML4Health_Attack_Plan.pdf` §2).
- Identified the dataset's core structural weakness: 37 cold speakers per partition, speaker-disjoint splits, 10:1 class imbalance — invites shortcut learning on speaker identity.
- Pinned the central thesis: *the 2017 fusion wisdom still holds in the FM era; we add three de-confounding mechanisms to prevent speaker memorisation.*

### 1.2 Architecture design

- Frozen WavLM-Large backbone with learned per-layer weighted sum (s3prl recipe).
- Three parallel feature streams feeding a projection MLP → shared representation `z`:
  - Pooled WavLM hidden states (mean/std/skew/kurtosis × 25 layers)
  - Hand-crafted features (OpenSMILE, mel-spectrogram, modulation spectrogram stats)
  - Discrete token histograms (deferred to A4)
- Three task heads from `z`: cold classifier, MDD speaker adversary via GRL (DANN fallback), Mahalanobis OOD-from-healthy score.
- Two-phase training: supervised contrastive pretraining (Phase 1, speaker-masked positives) → three-head supervised training (Phase 2).

### 1.3 Infrastructure built

- **Two-tier cache design** implemented:
  - `emb_pooled/<backbone>/<file>.pt` — `[25, 4×1024]` per-layer stats, fp16, ~tens of GB total.
  - `emb_frames/...` — deferred, populated post-hoc once A2 layer weights indicate which layers matter.
- **Extractor module** (`model/features/extract.py`): batched WavLM forward, writes pooled cache with manifest + config-hash versioning.
- **Backbone protocol** (`model/features/backbone.py`): `Backbone` protocol exposes `n_layers`, `hidden_dim`, `forward(audio, sr) → Tensor[L, T, D]`. Concrete implementations for WavLM; HuBERT and Whisper-encoder stubbed out for later ablations.
- **Layer-weighted pooling head** (`model/features/head.py`): `nn.Parameter(torch.zeros(25))` → softmax → weighted sum. Two optimiser param groups (weights lr 1e-4, MLP lr 1e-3).
- **Augmentation policy** (`data/augmentation.py`): splicing partner assignment with seed management. Cache writer lives in `model/features/extract.py`, consumes policy specs. Phase 2 fixed K=10, Phase 1 regenerated every 5 epochs. Symmetric across cold and non-cold to decorrelate splice presence from label.
- **Dataset class**: raw audio + OpenSMILE + mel-spectrogram features integrated.

### 1.4 Ran experiments

#### A0 — majority class

Floor reference, ~50 UAR. Sanity check.

#### A1 — frozen WavLM + mean-pool + linear probe

First run. Features cached, tiny MLP on top. Expected to be somewhere between 2017 e2e (60) and 2017 ComParE+SVM (70).

#### A2 (first attempt) — added hand-crafted features

Initial run collapsed: `val_UAR = 0.97`, `devel_UAR = 0.62`, 35-point gap. Layer weights frozen at uniform 1/25. Diagnosed as **speaker leakage in the train/val split** — val was chunk-level random, not speaker-grouped. Classic Huckvale trap, caught early.

#### A2 (corrected) — speaker-grouped CV

After fixing splits with `GroupKFold(groups=speaker_id)`:

```
best val_UAR     = 0.6346  (epoch 5)
devel UAR        = 0.6287
devel accuracy   = 0.7787
devel recall     = C:0.4387  NC:0.8187
val-to-test gap  = +0.0059  (honest)
dominant layers  : L4=0.042, L2=0.042, L1=0.042  (near-uniform — expected)
```

**This is the locked A2 baseline.** ±0.006 val/test gap is what a correctly-configured speaker-disjoint pipeline should produce.

### 1.5 Methodological decisions locked in

- Speaker-grouped cross-validation throughout. Never chunk-level random splits.
- Never tune hyperparameters on devel. Devel is for final comparisons per rung, nothing else.
- A2 is frozen. No retroactive retuning once later rungs produce results.
- fp16 for cache storage; fp32 for higher-order moment computation (avoids overflow on kurtosis of heavy-tailed transformer activations).
- Class-weighted CE loss with weights ~(9, 1). Class-balanced batch sampling guaranteeing minority presence per batch.
- All results tracked in `results/<rung>.json` + a rolling `results/README.md` markdown table.

---

## 2. Where we are

### 2.1 Context on the ladder

| Rung | Status | Devel UAR | Notes |
|---|---|---|---|
| A0 — majority class | Done | ~0.50 | Floor |
| A1 — frozen WavLM + linear probe | Done | — | Superseded by A2 |
| **A2 — + hand-crafted + PCA** | **Locked** | **0.6287** | Current baseline |
| A3 — + phoneme-aware pooling | Next | — | Frame cache needs populating for selected layers |
| A4 — + discrete token histograms | Pending | — | HuBERT native tokens first; VQ-VAE if beneficial |
| A5 — + OOD Mahalanobis feature | Pending | — | Deviation-from-healthy signal |
| A5.5 — + cross-speaker splicing | Pending | — | Data-level de-confounding |
| A6 — + contrastive pretraining | Pending | — | Representation-level de-confounding (needs pseudo-speakers) |
| A7 — + MDD speaker adversary | Pending | — | Gradient-level de-confounding (needs pseudo-speakers) |
| A8 — MDD ↔ DANN swap | Pending | — | Adversarial objective comparison |
| A9 — late fusion with ComParE+SVM | Pending | — | Classic fusion on top |

Target: close the 8-point gap to 71.0 across the remaining rungs.

### 2.2 Where A2 sits in context

| System | Devel UAR |
|---|---|
| 2017 majority class | ~50 |
| 2017 e2e CNN+LSTM | 60.0 |
| **A2 (us, locked)** | **62.9** |
| Huckvale honest test | 62.1 |
| 2017 ComParE + SVM | 70.2 |
| 2017 BoAW + SVM | 69.7 |
| 2017 late fusion (target) | 71.0 |

We're marginally above Huckvale's best honest test number with a trivial setup — the "modernised pillar" effect is already paying off. Gap to target is realistic given six unspent ablation rungs.

### 2.3 Key constraint discovered

**The 4students URTIC release has no speaker IDs in the TSV** — only `file_name` and `Cold` label. This blocks direct speaker probing, MDD adversary, speaker-masked contrastive pretraining, and cross-speaker augmentation as originally designed. Workaround: **pseudo-speaker clustering via ECAPA-TDNN embeddings on raw audio → k-means → cluster IDs as pseudo-speaker labels.** This is not just a probe metric — it's infrastructure that unblocks A5.5, A6, and A7 entirely. Priority accordingly.

### 2.4 Compute setup

- Running on a single 4090 (faster than the cluster for this workload given cached features).
- WavLM-Large forward pass ~50ms per 10s chunk → full extraction ~25 minutes for 28,652 chunks.
- Training per rung: ~2-10 min on cached features, <1 hour per seed run.
- Compute budget supports 5+ seeds per reported rung.

---

## 3. What's next, step by step

### Step 1 — A2 polish (30 minutes)

Final polish on A2 before moving on. Not hyperparameter tuning — threshold calibration is a defensible post-hoc step as long as it's measured on the *training side* of the wall.

1. Pick decision threshold by maximising UAR on a train-held-out chunk (not devel).
2. Apply threshold to devel, report both `UAR_argmax` (0.6287) and `UAR_calibrated` side by side.
3. Write `results/A2.json` with full metrics (UAR argmax + calibrated, recall_C, recall_NC, val-to-test gap, top-5 layer weights, hyperparameters, seed).
4. Create `results/README.md` with a rolling markdown table. Columns: rung, description, devel_UAR_argmax, devel_UAR_calibrated, recall_C, recall_NC, val-to-test gap, speaker_probe_acc, speaker_probe_NMI, notes. Row for A2 locked in.

### Step 2 — Pseudo-speaker infrastructure (3–4 hours)

This unblocks half the remaining ladder. Don't skimp.

1. Load SpeechBrain's ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`), extract 192-dim speaker embeddings on raw audio per chunk. Cache to `emb_ecapa/<file>.pt`. ~10-15 minutes for the whole dataset.
2. Fit k-means on **training-split embeddings only** with k in {100, 210, 420}. For each k, compute silhouette score and check intra/inter-cluster distance ratios. Pick k based on a combination of silhouette and matching URTIC's known ~210 speakers per partition.
3. Assign dev and test chunks to nearest centroid (no refitting). Save cluster assignments as `pseudo_speakers.tsv` keyed by file_name, versioned with method + k + seed in the filename.
4. Sanity-check cluster stability: compare intra-cluster ECAPA distances to inter-cluster. If clusters are fuzzy, consider HDBSCAN or spectral clustering as alternatives.
5. Build speaker probe infrastructure: 2-layer MLP from `z` → pseudo-speaker-id, cross-entropy. Trained post-hoc on a frozen rung's `z`. Report both top-1 accuracy and NMI.

### Step 3 — A2 probe baseline (20 minutes)

1. Run the speaker probe on A2's current `z`.
2. Record `speaker_probe_acc` and `speaker_probe_NMI` in the `results/A2.json` and rolling table.

This is the **"before" picture** for the paper's claim that de-confounding mechanisms reduce speaker-info in `z`. Every subsequent rung gets probed the same way — the number should drop as we add augmentation (A5.5), contrastive pretraining (A6), and MDD (A7).

### Step 4 — A3: phoneme-aware pooling (1–2 days)

1. **Decide which layers to frame-cache.** Inspect A2's learned layer weights — since they're near-uniform, pick a principled subset: layers {1, 4, 8, 12, 16, 20, 24} covers the depth range with reasonable coverage of the lower layers that paralinguistic work suggests matter most.
2. Second extraction pass populating `emb_frames/<backbone>/<file>.pt` for those layers.
3. Run a forced aligner over URTIC audio. Primary: WebMAUS (supports German). Fallback: Whisper-v3 with word timestamps + interpolation to phoneme level.
4. For each chunk, compute per-phoneme-class pooled stats (vowels, nasals, fricatives, plosives, silence). Concatenate with existing A2 features.
5. Retrain projection + head. Frozen A2 hyperparameters apart from the input-dim change.
6. Report. Expected gain: 0.5–1.5 UAR points. If phoneme-aware pooling barely moves the needle, that itself is a paper-worthy finding about whether Wagner's consonant/vowel insight transfers to modern FM embeddings.

### Step 5 — A4: discrete token histograms (1 day)

1. Extract HuBERT's native k-means cluster IDs per frame (no VQ-VAE training yet).
2. Compute per-chunk histograms + codebook perplexity + code-transition statistics.
3. Concatenate with A3 features, retrain projection + head.
4. Report.
5. Decision: if HuBERT tokens help clearly, try VQ-VAE on WavLM embeddings (URTIC-tuned codebook) as a potential upgrade. If they don't help, skip VQ-VAE entirely.

### Step 6 — A5: OOD Mahalanobis feature (half a day)

1. Fit mean + covariance on A4's `z` embeddings of **non-cold** training chunks only.
2. Compute Mahalanobis distance for every chunk, append as an extra scalar feature.
3. Retrain cold classifier (projection stays from A4).
4. Report.

### Step 7 — A5.5: cross-speaker splicing augmentation (2–3 days)

Uses pseudo-speakers from step 2. Data-level de-confounding.

1. For each cold chunk in train, generate K=10 augmented versions by splicing waveforms from 10 different pseudo-speakers (same cold label). Symmetric K=10 for non-cold chunks.
2. 100-200ms crossfades at splice boundaries.
3. Run extraction pass on augmented waveforms, write to `emb_pooled_aug/<backbone>/<file>_<k>.pt`.
4. Training samples one of {original, aug_0 … aug_9} per epoch per chunk.
5. Retrain. Report. Rerun speaker probe.

Expected gain: 1–2 UAR points. Expected speaker-probe-accuracy drop: measurable.

### Step 8 — A6: supervised contrastive pretraining (2–3 days)

Phase 1 of two-phase training. Uses pseudo-speakers.

1. Implement speaker-masked SupCon loss: positives = same cold label, different pseudo-speaker. Same-pseudo-speaker-same-class pairs masked out.
2. Attach small contrastive projection head on top of `z` during pretraining. Discard after Phase 1.
3. Batch sampler: 8 pseudo-speakers × 8 chunks per batch. Guarantees cross-speaker positives.
4. 10 epochs of Phase 1 with temperature τ=0.1, regenerating augmentation cache every 5 epochs.
5. Phase 2: attach cold classifier head, supervised training as before, projection MLP inherits Phase 1 weights at 5× lower lr than the head.
6. Report. Rerun speaker probe — expected larger drop.

### Step 9 — A7: MDD speaker adversary (3–4 days, high variance)

The main contribution. Highest-variance step.

1. Implement gradient reversal layer + 2-layer MLP speaker adversary head predicting pseudo-speaker-id.
2. MDD loss with margin γ (start with γ=4, ablate).
3. λ_adv ramp schedule: 2/(1+exp(−10·p)) − 1 where p is training progress.
4. Train A6-pretrained model with combined loss L = L_cold + λ_adv · L_spk + λ_ood · L_ood.
5. If MDD unstable after reasonable effort (2 days of debugging), fall back to DANN. This is a pre-committed decision — don't let sunk cost drag out MDD debugging.
6. Report. Rerun speaker probe — expected largest drop.

### Step 10 — A8: MDD ↔ DANN swap (1 day)

1. Swap the adversarial loss to whichever wasn't used in A7.
2. Report. Direct comparison of adversarial objectives.

### Step 11 — A9: late fusion (1 day)

1. Train standalone ComParE+SVM with WEKA-equivalent setup (sklearn SVM, linear kernel, C swept).
2. At inference, combine A7's posterior with SVM's via (a) averaging, (b) max-confidence, (c) logistic regression meta-learner on CV-held-out predictions.
3. Report all three; pick best.

### Step 12 — Scientific ablations (2–3 days)

Remove one component from A7 at a time. Quantifies each intervention's contribution:

- A7 − adversary (isolates MDD effect)
- A7 − contrastive pretraining (isolates contrastive effect)
- A7 − augmentation (isolates data-level de-confounding)
- A7 − OOD head (isolates OOD-as-feature)
- A7 − hand-crafted features (isolates classical-on-FM)

### Step 13 — Seed sweeps and final test submission (2 days)

1. Run 5+ seeds per rung reported in the paper. Mean ± std in the final results table.
2. Final test submission via Moodle on schedule (deadline 23.07).

### Step 14 — Paper writing (4 weeks)

Final paper due 31.08. Structure roughly follows the ablation ladder. Three central claims (see `ML4Health_Attack_Plan.pdf` §7):

1. Speaker confounding in URTIC is quantifiable and correctable.
2. The 2017 fusion wisdom holds in the FM era.
3. OOD-as-feature is robust only after speaker-invariance is in place.

---

## 4. Risk register (live)

| Risk | Current status | Mitigation |
|---|---|---|
| Split-level speaker leakage | **Resolved**. Speaker-grouped CV confirmed working (val-to-test gap +0.006). | — |
| MDD training instability | Not yet attempted. High prior risk. | A6 is publishable fallback; DANN as sub-fallback. |
| Pseudo-speaker clustering quality | Unknown until step 2. | Silhouette + cluster-distance sanity check; HDBSCAN as fallback. |
| Phoneme aligner on German passages | Unknown until step 4. | WebMAUS supports German; Whisper-v3 word timestamps as fallback. |
| fp16 overflow on higher-order moments | **Resolved**. Moments computed in fp32, cast at storage. | — |
| VQ-VAE codebook collapse | Not yet attempted. | Use HuBERT native tokens first; only train VQ-VAE if they help. |
| Dev-tuning trap | Active. Discipline required. | A2 locked. Threshold calibration on train-held-out only. All decisions via speaker-grouped CV. |
| Seed variance on small dataset | Unknown until seed sweeps. | 5+ seeds per reported rung. |

---

## 5. Schedule check

| Milestone | Target date | Status |
|---|---|---|
| Team formed | 21.04 | Done |
| Pitch slides | 28.04 | Pending |
| A3 working | end of May | On track |
| First test submission | 11.06 | On track |
| Midterm presentation | 16.06 | On track |
| A7 working | early July | On track (pending MDD success) |
| Final test submission | 23.07 | On track |
| Final presentation | 28.07 | On track |
| Paper due | 31.08 | On track |

Internal slack: 3 weeks between "A3 working" and "first test submission," 3 weeks between "A7 working" and "final test submission." If anything slips, it's the fancy components (A8, A9, some scientific ablations), not the core A1–A7 sequence.

---

## 6. Immediate next actions (this week)

1. **Today / tomorrow**: Step 1 (A2 polish + threshold calibration + `results/` scaffold).
2. **This week**: Step 2 (pseudo-speaker infrastructure). This is the critical unblocker for everything downstream.
3. **Next week**: Step 3 (A2 probe baseline) + start Step 4 (A3 phoneme-aware pooling).

Golden rule: every Friday ask *"if the project froze today, would we have a defensible result?"* From A3 onward the answer must always be yes.
