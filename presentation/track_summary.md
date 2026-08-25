---
title: "Cold Detection from Voice — Foundation-Model Track Summary"
subtitle: "Advanced AI in Health · ComParE 2017 Cold (URTIC)"
date: "June 2026"
---

## What I'm doing

My part of the project is the **main cold-detection classifier** for the ComParE 2017 Cold sub-challenge (URTIC corpus — German speakers reading a story, ~28k 8-second clips, binary cold / non-cold), plus the methodology that is really our second contribution: **measuring and removing speaker shortcut-learning**. The question isn't only "can we beat the 71.0% UAR baseline" — it's "how much of *anyone's* number is genuine cold detection vs. the model secretly memorising *who is speaking*."

## The problem I'm trying to solve

URTIC is small (~9k cold-labelled chunks), heavily imbalanced (~9.5% cold), and split so that no speaker appears in both train and test. That combination invites **shortcut learning on speaker identity**: a model learns "this voice = cold" instead of "this voice *quality* = cold," then collapses on unseen speakers. The historical evidence is stark — the 2017 baseline reported **71.0 UAR on devel**, but Huckvale's honest re-test on the withheld set got **62.1**, a 9-point drop. So the task is twofold: (1) build a foundation-model classifier that beats the *honest* number, and (2) prove the gains are cold signal, not leakage.

## Why this approach specifically

Three pillars, each a principled choice over the 2017 setup:

1. **Frozen foundation model instead of hand-engineered features.** WavLM-Large (316M params, pretrained on 94k hours of speech) replaces the 2017 6,373-dim ComParE+SVM. Frozen, not fine-tuned — the dataset is far too small to fine-tune without overfitting, and freezing keeps the comparison clean.
2. **An honesty-audited acceptance gate, not just "did UAR go up."** Every feature I consider adding must pass a **2-D gate**: raise cold UAR *and* not increase a speaker probe's accuracy. A feature that lifts UAR by leaking speaker identity gets rejected. This discipline is what makes our numbers survive the Huckvale check.
3. **Pseudo-speakers, because the released data hides speaker IDs.** The "4students" URTIC release strips speaker labels, so leakage isn't directly measurable. I reconstruct it: ECAPA-TDNN speaker embeddings → k-means (k=210) → "pseudo-speakers," cross-validated against HDBSCAN (ARI 0.86) plus a negative control, so the clusters are real speaker structure, not artefacts.

## The (short version of the) method

- **Backbone → fixed vector.** Frozen WavLM gives 25 layers × per-frame 1024-d states. I pool each layer to mean / std / skew / kurtosis, then a learned **softmax over the 25 layers** + small MLP produces the cold logit.
- **A2.5 — the "honesty prior."** Instead of uniform layer weights, I initialise them from a per-layer audit (each layer scored on *cold signal minus speaker leakage*). This turns out to be a **distinct optimisation attractor**: uniform init never reaches it, even at 10× learning rate. (+0.020 UAR, ~5σ.)
- **Late fusion (A5b).** `final_logit = logit_A2.5 + β · mean(z-scored handcrafted-group logits)`. Candidate groups (voicing, prosody, voice-quality, energy, modulation-spectrum, spectral-shape, OOD) are admitted by a top-K "subtractive honesty" rule — only if they clear the 2-D gate.
- **Speaker probe (the audit instrument).** Train a classifier to predict pseudo-speaker ID from the features. Chance = 1/210 = 0.0048; our pre-registered gate ceiling is ~0.078. Our fusion features sit at **0.018–0.073** → the model is *not* shortcutting.

## Why this isn't just a guess

- **Huckvale 2018** is the precedent for the whole problem — the documented 9-point honest-vs-devel collapse on this exact corpus.
- **Pasad 2021 / Chen 2022** layer-analysis work predicts speaker info concentrates in early WavLM layers — which my per-layer audit confirms on URTIC (speaker recoverability decays L0→L24; cold signal stays flat across the stack).
- Everything runs under a **pre-registered protocol**: τ tuned only on a train-side hold-out, model selection on `devel_val`, one-shot honest read on `devel_test`, 3–5 seeds per locked rung. No tuning on the test proxy.

## Problems / limitations

- The official test labels are withheld, so `devel_test` is our honest proxy — we can't make a *direct* apples-to-apples claim against the 2017 hidden-test 71.0. (Quantified: under a shadow-split robustness harness, the gap to 0.710 is smaller than the partition variance of either estimate.)
- The cold class is **sparse** (~21 cold speakers in all of devel), so speaker-level metrics are noisier than chunk-level — counterintuitively, aggregating to speakers *increases* variance.
- Handcrafted gain features leak recording-condition signal; I keep only the gain-invariant slice.

## Where it stands / next steps

Current stack on the honest proxy (leak-corrected, grouped splits):

> A2 baseline **0.6361** → A2.5 honesty-prior **0.6564** (+0.020) → K=2 late fusion **0.7037** single / **0.7090** 5-seed ensemble

That's **statistically equivalent to the 2017 71.0 baseline within partition variance**, achieved with *both* speaker-probe gates passing — i.e. the gains are honest cold signal, not memorised speakers. Next:

- **De-confounding ladder**: cross-speaker splicing augmentation (data-level), speaker-masked contrastive pretraining (representation-level), MDD/DANN gradient-reversal adversary (gradient-level) — each re-probed to show speaker info dropping.
- Fold in the **other tracks** (path-signatures features, Fisher-Vector/GMM) as additional honesty-gated fusion candidates.
- Write up the two contributions: the UAR result, and the cleaner read of how much of the 2017 number was speaker shortcut learning.
