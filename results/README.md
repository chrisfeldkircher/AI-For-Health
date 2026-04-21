# Ablation Results

Every rung on the attack-plan ladder gets one row. All UAR numbers are computed
on **devel_test** (50% of ComParE 2017 Cold devel, speaker-disjoint from train
by URTIC construction), which is the honest proxy for the withheld challenge
test set.

## Evaluation protocol (locked — do not change per-rung)

- **train** → `train_fit` (~90%, stratified on Cold label, seed=42) + `train_threshold` (~10%, stratified, seed=42)
- **devel** → `devel_val` (50%, stratified, seed=42) + `devel_test` (50%, stratified, seed=42)
- **Model selection**: best `val_UAR` on `devel_val` (early stop on patience=6).
- **Threshold calibration**: τ maximising UAR on `train_threshold` after training finishes. Never use devel for τ — that's the Huckvale dev-tuning trap.
- **Honest number**: UAR on `devel_test` reported at both argmax and τ.
- **Speaker probe**: pseudo-speaker IDs from ECAPA-TDNN embeddings clustered on train, assigned to devel by nearest-centroid. Probe = 2-layer MLP on `z`. Reports top-1 and NMI.

## Results table

All numbers are `mean ± std` across N=3 training seeds (data splits fixed at seed=42). `std` uses Bessel's correction (ddof=1).

**Seed discipline**:

- **Dev seed**: `42` (use for iteration / debugging)
- **Lock seeds**: `{42, 123, 7}` (run all three before claiming a rung, commit numbers to `results/<rung>.json`)
- **Paper seeds**: extend lock set to 5 if a borderline rung needs tighter bounds. Keep the original three.
- **Never compare a 1-seed number to the 3-seed distribution.**

| Rung | UAR (argmax)    | UAR (τ-cal)     | τ*    | calib_delta | recall_C        | recall_NC       | probe top-1       | probe NMI        | val→test        | notes     |
|------|-----------------|-----------------|-------|-------------|-----------------|-----------------|-------------------|------------------|-----------------|-----------|
| A2   | 0.6428 ± 0.0034 | 0.6464 ± 0.0082 | ~0.48 | +0.0036     | 0.4321 ± 0.0284 | 0.8607 ± 0.0192 | 0.0501 ± 0.0009   | 0.3772 ± 0.0030  | −0.001 ± 0.005  | see below |

Recalls shown at τ*. `calib_delta = UAR_calib − UAR_argmax` (sign and size both informative). Speaker probe: 2-layer MLP on `z` (proj_dim=128), trained on train_fit z with pseudo-speaker targets from `k210_seed42.tsv`, evaluated on all of devel. Chance top-1 = 0.0048 (1/210). Probe train top-1 ≈ 0.92 for all seeds — the 18× train/devel gap is the main diagnostic and should shrink as de-confounding rungs land.

**Statistical floor**: A2 argmax σ = 0.0034 → any later rung needs ~+0.007 UAR (2σ) to be distinguishable from training-seed noise at N=3. For tighter bounds on borderline rungs, bump to N=5.

### A2 architecture

Frozen WavLM-Large (25 hidden states, pooled mean+std+skew+kurt per layer) → FeatureStandardiser (per-position z-score, fit on train) → softmax layer weights (lr × 0.1) → 2-layer MLP 128-d + BatchNorm + GELU + dropout 0.5 → 2-class linear. Balanced-batch sampler, no class weights in loss, AdamW `base_lr=1e-3`, cosine schedule, early stop patience 6. Best epoch 3/25.

### A2 notes

- **Calibration bought ~0 UAR but added variance.** Mean gain +0.0036 (< calibration σ of 0.0082). τ* varies 0.47–0.50 across seeds. Reason: balanced sampler + no class weights already centres the decision boundary; the `C=0.43` recall asymmetry is *not* a threshold-placement problem, it's a representation problem — the model can't separate ~57% of cold samples from non-cold in z-space regardless of where the boundary sits. This rules out "A2 under-reports because of miscalibration" and points at representation geometry as the real ceiling. The argmax number is the cleaner reference for rung-over-rung comparisons because its σ is 2.4× tighter.
- **Train UAR = 0.9383 on `train_threshold`** vs **0.6510 on devel_test**. The probe fits train hard and generalises weakly. Expected for a linear-ish probe on FM features with ~9k samples per class across unseen speakers. This is the gap the remaining rungs have to close.
- **Layer weights stayed ≈ uniform** (all ≈ 0.042 = 1/24) because early stop fired at epoch 9 (best=3). At `layer_lr = base_lr × 0.1`, three epochs isn't enough for zero-init softmax weights to specialise. Not pursued further — reviewer confirmed this is expected for an underconstrained linear probe.
- **val→test gap = −0.003** — devel_test slightly outperformed devel_val, which is noise around the true value. Splits are consistent; speaker-disjointness is holding.
