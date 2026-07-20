# Required paper correction after the speaker-group overlap audit

## Decision

Do not make the proposed one-number edit from `0.856` to `0.93`. The new
side-local benchmark validates the ECAPA representation, but it simultaneously
shows that the Train-centroid labels historically used to split Development are
not a valid held-side speaker partition.

## Numbers that can be reported

Report the raw-space, side-local KMeans-versus-HDBSCAN agreement on all
recordings as the primary result:

| Official side | All-recording ARI | HDBSCAN-non-noise ARI | Non-noise coverage |
|---|---:|---:|---:|
| Train | 0.856 | 0.930 | 97.28% |
| Development | 0.829 | 0.929 | 96.69% |
| Test | 0.879 | 0.944 | 97.59% |

The approximately `0.93` values are sensitivity analyses after excluding
HDBSCAN noise and must be accompanied by coverage. The six-seed KMeans
stability means are 0.889, 0.897 and 0.888 for Train, Development and Test. The
external known-speaker ECAPA recovery result on MLS German is 0.963 ARI.

The pooled historical all-recording ARI `0.309` should not be presented as
method instability. It is evidence that assigning held-side recordings to
Train centroids fragments the held-side structure. It remains useful only for
that diagnostic purpose.

## Claims that must be removed or qualified

Remove every statement that the historical `devel_val/devel_test` split is
strictly speaker-disjoint. Under Development-local ECAPA groups, the canonical
split has 201/210 groups on both sides and affects 97.95% of recordings. Across
11 split seeds, the affected fraction is 94.40--97.95%.

Consequently:

- the old canonical and shadow WavLM results cannot be described as
  speaker-disjoint;
- zero overlap under the shipped IDs is not a validation result, because those
  are the same IDs used to construct the split;
- the old probe interpretation and its claimed de-confounding foundation must
  be revised;
- the figure comparing the shipped map to a pooled clustering is a
  fragmentation diagnostic, not validation of held-side group assignments.

## Results that remain valid for the frozen submission

The fixed G4+CQT architecture remains supported by tests that do not depend on
the invalid historical Development sub-split:

- whole-side Train to Development: G4 0.6422, CQT 0.6417, fusion 0.6897 UAR;
- whole-side Development to Train: G4 0.6132, CQT 0.6679, fusion 0.6468 UAR;
- bidirectional mean: G4 0.6277, CQT 0.6548, fusion 0.6683 UAR;
- evaluation-independent threshold transfer used groups fitted locally within
  each fitting side;
- the final Train+Development OOF threshold candidate also fits groups
  separately within Train and Development before offsetting IDs.

The combined-data pooled-k420 CV remains only a data-scaling diagnostic. It is
not needed as the load-bearing justification for G4+CQT.

## Rerun scope

Rerun the old WavLM ladder only if those results will remain central to the
paper. Use Development-local groups for any Development sub-split, freeze all
choices on Train, and use whole official Development as the honest evaluation.
If there is insufficient time, remove the speaker-disjoint and hidden-test
comparison claims and present that ladder as exploratory history.

No new TRILLsson, openSMILE, spectrogram-CNN or fusion search is justified for
the current submission. The already prepared blinded pair audit can be held for
reviewer response rather than made a submission prerequisite.
