# Mid-term result: notes for the talk

Plain notes to read before the ML4Health mid-term talk. The deck is
`presentation/MidTerm_Postmortem_FINAL_PLAN.pptx`. This file is the longer record
behind it. Earlier drafts of this brief blamed the threshold and proposed a
recalibration fix; that was wrong, and this version replaces it with the
diagnosis the measurements actually support.

## Summary in one paragraph

We rebuilt the 2017 ComParE Cold late fusion with a frozen WavLM backbone and
matched the published baseline on our own dev split (UAR 0.711 vs 0.710). On the
hidden test it scored UAR 0.62, below every one of our 10 dev folds. We checked
the run five ways and it is correct, so this is a generalization gap, not a bug.
The gap is not a threshold or calibration problem: at the same cold-call rate we
use on dev, recall on test drops below every dev fold, and a recall drop at a
fixed call rate is a ranking loss that no threshold or rescaling can undo. The
cause is that our dev folds, although speaker-disjoint, all come from one speaker
pool; the hidden test is a different speaker population, and the cold direction
the model learned transfers worse to it. The two options for the final
submission are to report this as the limit of a frozen backbone, or to unfreeze
the top WavLM layers and train with speaker-stratified batches.

## The result and the numbers

- Hidden-test UAR: 0.6205 (accuracy 0.596, macro precision 0.542, macro F1 0.479).
- Our dev estimate: 0.6940, standard deviation 0.0157, over 10 speaker-disjoint folds.
- Our single canonical dev split: 0.7111.
- 2017 ComParE baseline: 0.710.
- The test result is below all 10 dev folds (lowest fold 0.6518).
- Test cold prior: 9.4% (895 of 9551). We predicted cold on 43.2% (4124 of 9551).
- On cold clips: recall 65.0%, precision 14.1%. On not-cold: recall 59.1%, precision 94.2%.

The confusion matrix is recovered from the four leaderboard metrics in closed
form (it reproduces all four to six decimals): TP 582, FP 3542, FN 313, TN 5114.
This uses only the published metrics, not per-clip test labels.

## What we checked (so we can say it is not a bug)

Five checks on the submitted run, all pass:
1. The manner-label cache used by the G4 features is built the same way as for train and dev.
2. The per-seed fusion weights load straight from the locked results file.
3. The threshold is re-derived on the dev-threshold split and matches the locked value to six decimals.
4. The test feature extraction uses the same config as train and dev.
5. The submitted CSV is byte-for-byte identical to the pipeline output, with zero row-level disagreements against the threshold.

## Why it is not a threshold or calibration problem

This is the core argument and it stands on the leaderboard metrics plus the dev
folds alone.

- UAR is balanced accuracy. On a 10% prior its best operating point deliberately
  calls the minority class often. At our locked threshold the cold-call rate is
  about 42% on dev and on every shadow fold (range 40.2% to 43.2%). So the 43.2%
  on test is the rate the metric wants, not over-calling.
- Hold the cold-call rate matched and look at recall. On test, recall on cold is
  0.650 and recall on not-cold is 0.591. Both fall below the entire dev range
  (floors 0.673 and 0.612).
- At a matched cold-call rate, a recall drop means the model orders the clips
  worse. Moving the threshold or rescaling the scores cannot produce or fix that.
  So calibration and threshold drift are ruled out.

## The supporting diagnostics

These three checks corroborate the argument above. They ran on the on-disk test
caches on the main machine, so they need that machine to reproduce.

- Re-standardizing the handcrafted-group scores on the test set leaves the fused
  ranking unchanged (rank correlation 1.0000). The group scores barely move from
  train to test (0.01 to 0.04 of a training standard deviation). The negative
  mean of the fused score comes from the WavLM head, not the groups.
- The WavLM head's output distribution is almost the same on dev and test (mean
  -3.52 vs -3.75, spread ratio 0.96).
- The per-dimension means and spreads of the 25 by 4096 pooled features are
  stable from dev to test (median shift 0.02 standard deviations, no dimension
  past 0.30, spread ratio about 1.0). A test-time normalization like BN-adapt
  corrects exactly these and so has nothing to correct. A weak whole-vector
  difference does exist (a simple dev-vs-test classifier reaches AUC 0.57), but it
  is a rotation, not a shift in means or spreads, so BN-adapt cannot fix it
  either.

Taken together: the score distributions are stable from dev to test, but the UAR
fell and the recalls dropped below the dev floor. Stable distributions plus a
recall drop means the score-to-label ordering got worse on the test speakers.

## What we ruled out

- Threshold or calibration drift: ruled out by the matched-rate argument.
- Handcrafted-feature shift or fusion fragility: ruled out by the
  re-standardization check (ranking unchanged).
- WavLM feature shift feeding the head: ruled out by the stable per-dimension
  moments; BN-adapt has nothing to correct.
- Single-split threshold luck and prior mismatch: train prior is 10.2%, dev 10.5%,
  so a prior correction is about 0.1 of a logit, far too small to matter.
- Ensemble averaging: the dev folds use the same averaging and stay stable, so it
  is not the driver.

## The diagnosis

One modeling issue and one protocol issue.

- Modeling: an off-pool discrimination loss. The frozen WavLM plus pooling head
  learned a cold direction that separates the held-out test population worse. The
  most likely reason is the speaker shortcut this project set out to control: our
  M8 to M19 control experiments already showed that frozen-model de-confounding
  (cross-speaker mixup, speaker-masked contrastive learning, gradient reversal)
  did not beat a plain cold classifier. We say off-pool, not speakers
  specifically, because we have no test speaker IDs and cannot decompose the axis
  directly.
- Protocol: our 10 dev folds are speaker-disjoint but all drawn from one speaker
  pool. They measure generalization to new speakers from the same pool, not to a
  new pool. A shortcut that is steady across the whole pool is invisible to every
  fold, which is why the test landed below the worst fold. The speaker-aware
  protocol still did its job: it flagged the gap rather than hiding it.

## What we change next

- Ruled out by measurement, so not worth trying: moving the threshold, rescaling
  the scores, test-time feature normalization. The scores already sit where they
  should; these only rescale them.
- Path A: report the result as the limit of a frozen backbone. The speaker-aware
  protocol already flagged the gap.
- Path B: unfreeze the top WavLM layers and train with speaker-stratified
  batches, so the cold direction leans less on who is speaking. Frozen-scope
  de-confounding already failed in M8 to M19, so unfreezing is the next lever.
- Decide on the dev folds before the next submission.

## Likely questions and short answers

- Is 0.62 just a bad result? It is below our dev estimate, but the point of the
  talk is why, and the why is specific and measurable: the model ranks cold worse
  on a new speaker population, not that the pipeline is broken.
- You have the test labels now, is the confusion matrix cheating? No. It is
  recovered from the four published leaderboard numbers in closed form. No
  per-clip test labels are used.
- Is this not just the 90/10 prior plus a different speaker set, with nothing
  wrong in the model? Partly yes, and that is the finding: the prior is matched,
  and what remains is a ranking loss on the new speaker pool.
- Why not just recalibrate or normalize at test time? Because the scores are
  already where they should be; the measurements show no shift in means or
  spreads to correct, and a ranking loss is not fixable by rescaling.
- How do you know it is speakers? We can show it is off-pool. Naming speakers
  rests on the M8 to M19 results, since we have no test speaker IDs to test the
  axis directly.
- Did your approach fail? The model hit a ceiling, but the evaluation protocol
  worked: it flagged the gap and let us locate the cause instead of guessing.

## Numbers to keep in mind

| Number | Meaning |
|---|---|
| 0.6205 | hidden-test UAR |
| 0.6940 (sd 0.0157) | our dev estimate over 10 speaker-disjoint folds |
| 0.7111 | our single canonical dev split |
| 0.710 | 2017 ComParE baseline |
| 0.6518 | lowest dev fold (test is below it) |
| 9.4% | true cold rate on test |
| 43.2% | our predicted cold rate on test |
| about 42% | cold-call rate on dev at the same threshold |
| 0.650 / 0.591 | test recall on cold / not-cold |
| 0.673 / 0.612 | lowest dev-fold recall on cold / not-cold |

## References

- Schuller et al. The INTERSPEECH 2017 Computational Paralinguistics Challenge: Addressee, Cold and Snoring. Interspeech 2017.
- Chen et al. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. IEEE JSTSP, 2022.
- Yang et al. SUPERB: Speech processing Universal PERformance Benchmark. Interspeech 2021.
- Coppock, Jones, Kiskin, Schuller. COVID-19 detection from audio: seven grains of salt. Lancet Digital Health, 2021.

## Files

- Deck: `presentation/MidTerm_Postmortem_FINAL_PLAN.pptx`
- Slide images: `presentation/slide_previews/slide_01..07.png`
- Figures: `presentation/figures/fig{1,6,8,9,10}_*.png`
- Figure code: `presentation/make_slide_figures.py`
- Deck builder: `presentation/build_pptx.py`
- Diagnostics: `presentation/_exp1_restandardize.py`, `_exp2_a2_head_shift.py`, `_exp3_covariate_shift.py`, `_verify_amplification.py`
