# Pseudo-speaker groupings: which file to use for what

URTIC ships no speaker IDs, so speaker groups are k-means clusters of ECAPA
embeddings (see `../ecapa-voxceleb/README.md`). Two labelings exist; they serve
different purposes. Do not mix groupings within one experiment.

## k210_seed42.tsv (shipped, canonical for locked results)

- KMeans k=210 fit on TRAIN only; devel/test assigned by nearest train centroid.
- Every locked result (A2 through A5b, the mid-term submission, the shadow
  splits) was produced under this grouping. Keep using it for anything that
  compares against those numbers.
- Known limitation (results/speaker_pipeline_verification.json, V7/V8): devel
  and test speakers have no centroid of their own, so their clips fragment
  across clusters (top-1 NN same-cluster: train 0.97, devel 0.49, test 0.47).
  On the devel val/test split, 27% of clips have a same-speaker-proxy neighbor
  on the other side. Measured consequence: this does NOT bias the reported UAR
  (results/audit_leakage_optimism.json; leaked clips score worse, not better,
  and val_test_gap is negative), because everything is fit on the clean train
  split. It DOES mean "speaker-disjoint" is only literally true on train, and
  devel speaker-probe targets are fragmented.
- k100_seed42.tsv / k420_seed42.tsv: same recipe at coarser/finer k. Built as a
  robustness bracket around the ~210-speakers-per-split assumption (NOT a seed
  sweep; all use seed 42). No experiment has consumed them yet.

## pooled_k420_seed42.tsv (faithful within-split groupings)

- KMeans k=420 fit on train+devel POOLED (~210 speakers per split); test
  clustered separately (k=210, IDs offset +1000).
- Fixes the fragmentation: top-1 NN same-cluster 0.97-0.98 on ALL splits;
  devel split same-speaker leakage drops 0.27 -> 0.015
  (results/pooled_pseudo_speakers_diagnostics.json).
- Use for: honest "speaker-disjoint" claims about devel/test splits, speaker
  PROBE targets on devel, A6/A7 pseudo-speaker targets, and any NEW evaluation
  that does not need comparability with the locked k210 results.
- Rebuild with `python build_pooled_pseudo_speakers.py` at the repo root.

## Rule of thumb

Comparing against a locked number: k210_seed42.tsv.
Making a claim about speaker honesty, or training against speaker targets:
pooled_k420_seed42.tsv.
