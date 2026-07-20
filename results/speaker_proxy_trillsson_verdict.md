# Speaker-proxy verification and TRILLsson verdict

## Decision

Keep the side-local ECAPA proxy groups for grouped evaluation and uncertainty
analysis. Do **not** replace them with TRILLsson and do **not** fuse TRILLsson
embeddings or proxy labels into the health classifier.

The earlier statement that the speaker identities were "non-reproducible" was
too broad. The low pooled audit agreement (ARI about 0.31) compared partitions
with different scope and granularity: side-local fixed-210 labels versus a
pooled Train+Development density solution with roughly twice as many clusters.
When the comparison is made within each official side, the original high
agreement is reproduced.

These are still proxy identities, not ground-truth URTIC speaker IDs.

## Tests run

No cold/health labels were loaded in any URTIC speaker test.

### ECAPA reproducibility within each official side

All ECAPA embeddings were L2-normalized. KMeans used `k=210`, `n_init=10` and
six seeds. Agglomerative and spectral clustering were independent clustering
algorithms on the same ECAPA representation. HDBSCAN selected its own cluster
count.

| Side | KMeans seed ARI, mean | Agglomerative vs KMeans ARI | Spectral vs KMeans ARI | HDBSCAN vs KMeans ARI | HDBSCAN non-noise clusters | Noise |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0.889 | 0.844 | 0.902 | 0.856 | 204 | 2.72% |
| Development | 0.897 | 0.818 | 0.915 | 0.829 | 205 | 3.31% |
| Test | 0.888 | 0.863 | 0.917 | 0.879 | 205 | 2.41% |

Across sides, mean KMeans seed ARI was 0.891 and mean fixed-k cross-method ARI
was 0.877. KMeans, agglomerative and spectral clustering unanimously retained
92.9%, 94.2% and 94.5% of directed ECAPA 10-nearest-neighbour edges on Train,
Development and Test respectively; a majority retained 99.0% or more.

For HDBSCAN specifically, the conservative all-recording ARI values above are
0.856, 0.829 and 0.879. Excluding the 2.4–3.3% of recordings HDBSCAN marks as
noise raises ARI to 0.930, 0.929 and 0.944. The latter are sensitivity results,
not directly interchangeable headlines; they must be reported with coverage.

### Labeled external speaker controls

The identical KMeans-at-true-k recovery test was applied to cached labeled
LibriSpeech English and MLS German audio. TRILLsson used up to three
deterministic two-second windows per recording, mean pooling and L2
normalization.

| Representation | Control | ARI | Purity | Mean speaker fragmentation | Seed ARI, mean |
|---|---|---:|---:|---:|---:|
| ECAPA | LibriSpeech English | **0.996** | **0.998** | 1.025 | not re-run here |
| TRILLsson1 | LibriSpeech English | 0.834 | 0.890 | 1.775 | 0.761 |
| TRILLsson5 | LibriSpeech English | 0.799 | 0.865 | 1.525 | 0.772 |
| ECAPA | MLS German | **0.963** | **0.972** | see ECAPA audit | not re-run here |
| TRILLsson1 | MLS German | 0.488 | 0.762 | 2.397 | 0.475 |
| TRILLsson5 | MLS German | 0.552 | 0.796 | 2.229 | 0.530 |

Both TRILLsson variants pass the shuffled-label negative control, but neither
is sufficiently accurate or stable to influence the production speaker map.
The strongest released variant, TRILLsson5, does not close the gap to ECAPA.

### Independent TRILLsson1-to-ECAPA cross-view check on URTIC

TRILLsson1 was extracted for all 28,652 URTIC recordings and clustered
side-locally at the known design prior of 210 speakers.

| Side | TRILLsson seed ARI, mean | TRILLsson vs ECAPA-KMeans ARI | NMI | ECAPA-KMeans cohesion in TRILLsson 10-NN graph |
|---|---:|---:|---:|---:|
| Train | 0.591 | 0.566 | 0.821 | 0.769 |
| Development | 0.549 | 0.516 | 0.799 | 0.755 |
| Test | 0.603 | 0.554 | 0.818 | 0.785 |

ECAPA agrees with TRILLsson nearly as much as TRILLsson agrees with another
seed of itself. This is useful independent evidence that ECAPA is capturing a
real, repeatable identity-like structure. It is not evidence that TRILLsson is
the better partition.

## Consequence for performance evaluation

Use ECAPA proxy IDs only as an evaluation/control variable:

1. Split Train folds by side-local ECAPA proxy group, never randomly by chunk.
2. Select thresholds from grouped out-of-fold predictions on the fitting side.
3. Keep official cross-side transfer (Train to Development and the reverse) as
   the strongest evaluation-independent check.
4. Bootstrap paired performance differences over proxy groups rather than
   individual chunks, while explicitly calling the groups inferred proxies.
5. Do not choose `k`, clustering method, speaker smoothing, model architecture,
   fusion weight or threshold using Test cold labels.

Analyses that used random chunk folds, or that assigned Development/Test to
Train-only centroids and treated those assignments as coherent identities,
should be rerun. Corrected grouped and official-side transfer analyses do not
need to be rerun merely because TRILLsson was tested.

Side-local clustering of an unlabeled evaluation side is transductive. It is
acceptable as a disclosed grouping/uncertainty audit, but the strict inductive
health score should not require side-wide Test clustering. This is one reason
not to put proxy IDs or speaker smoothing inside the deployed predictor.

## Architectural recommendation

The speaker result does not support adding a TRILLsson branch or a spectrogram
CNN to the frozen submission. Keep the existing monolithic fixed equal-fusion
candidate:

```text
G4: balanced logistic regression on 7 gain-invariant features
G9: balanced logistic regression on 168 CQT features
score = 0.5 * (zscore(G4 logit) + zscore(G9 logit))
decision = score >= grouped-OOF threshold
```

ECAPA remains outside this predictor. Use it for grouped folds, grouped OOF
threshold calibration, group-balanced diagnostics and group-level uncertainty.
Do not feed ECAPA/TRILLsson embeddings or cluster IDs to the classifier.

For a future, non-submission experiment, a spectrogram CNN is worth testing
only as a pre-registered residual branch against the frozen G4+CQT baseline in
nested grouped CV and official cross-side transfer. It should be accepted only
if the paired group-bootstrap interval and both transfer directions improve;
the current evidence gives no reason to expect TRILLsson fusion itself to help.

## Reproducible artifacts

- `benchmark_speaker_proxies.py`: label-free ECAPA stability/method benchmark.
- `validate_trillsson_speaker_proxy.py`: official TRILLsson extraction,
  labeled controls and URTIC cross-view benchmark.
- `speaker_proxy_method_benchmark.json`: ECAPA benchmark results.
- `trillsson1_recovery_*.json`, `trillsson5_recovery_*.json`: labeled controls.
- `trillsson1_ecapa_cross_view_*.json`: all-side cross-view results.
