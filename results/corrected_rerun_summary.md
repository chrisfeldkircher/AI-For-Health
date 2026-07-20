# Single-partition corrected rerun (superseded)

**Do not use this ranking alone for architecture selection.** The later repeated-partition reconciliation found that the chosen outer split seed was pessimistic for CQT and that a fixed equal G4+CQT fusion averaged 0.623 UAR. See `cqt_protocol_reconciliation.json` and `fixed_g4_g9_repeated_cv.json`.

All selection was restricted to official Train. Official Development was not used.

| Model | OOF UAR | 95% group-bootstrap CI | Delta vs G4 | P(better than G4) |
|---|---:|---:|---:|---:|
| G4_fixed_tau0 | 0.592 | [0.532, 0.656] | baseline | — |
| G4_inner_tuned_tau | 0.592 | [0.530, 0.654] | -0.000 | 0.461 |
| G9_CQT | 0.583 | [0.513, 0.665] | -0.009 | 0.408 |
| unconstrained_stack | 0.573 | [0.498, 0.659] | -0.019 | 0.299 |
| G4_anchor_G5 | 0.593 | [0.528, 0.657] | +0.001 | 0.492 |
| HuBERT_large | 0.564 | [0.502, 0.649] | -0.028 | 0.245 |
| G4_anchor_HuBERT | 0.570 | [0.503, 0.638] | -0.023 | 0.066 |
| WavLM_large | 0.552 | [0.491, 0.634] | -0.040 | 0.147 |
| G4_anchor_WavLM | 0.564 | [0.483, 0.645] | -0.028 | 0.122 |

## Decision

This single partition supports excluding the current neural branches, but it does not justify discarding CQT or freezing G4 alone. The repeated-partition analysis supersedes that decision. Do not submit the current unconstrained learned stack or foundation-head architecture.

## Final-fit rule

After architecture and hyperparameters are frozen, refit once on official Train+Development and predict Test with the fixed zero threshold. Do not compare architectures, fusion weights, thresholds, or seeds on Development again after this point.
