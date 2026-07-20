# Reconciliation of the three external-agent responses

## Bottom line

The submission decision stays unchanged: retain the frozen monolithic G4+CQT
equal fusion, do not add TRILLsson, openSMILE or a CNN branch, and do not use
new Development measurements for selection.

The ECAPA representation conclusion is verified, but the historical held-side
group assignments are not. The openSMILE measurements are verified. Some of
the agents' paper language, the evaluation claims and one CNN citation need
correction.

## 1. The ECAPA ARI discrepancy is conclusively resolved

The original `0.8557/0.9620` result came from cell 14 of `model/test.ipynb`:
raw L2-normalized 192-D ECAPA, 9,505 Train recordings, KMeans-210 versus
HDBSCAN-204. UMAP was explicitly disabled.

The later `0.3094/0.7678` result compared a different object: the single
Train-fitted 210-label map over all 19,101 Train+Development recordings against
a pooled HDBSCAN partition with 406 non-noise clusters. Development alone had
ARI 0.193 under those fragmented Train-centroid assignments.

Direct controls from the same stored embeddings:

| Comparison | ARI | NMI |
|---|---:|---:|
| Side-local Train KMeans vs side-local HDBSCAN | 0.8557 | 0.9620 |
| Side-local Development KMeans vs side-local HDBSCAN | 0.8292 | 0.9608 |
| Side-local Test KMeans vs side-local HDBSCAN | 0.8792 | 0.9667 |
| Train-fit k210 map vs pooled HDBSCAN, Train+Development | 0.3094 | 0.7678 |
| Offset side-local k210+k210 map vs pooled HDBSCAN | 0.7267 | 0.9584 |

Therefore the gap is caused by scope, granularity and held-side assignment
fragmentation—not reduced-space inflation. The paper should report the
side-local raw-space values and retain `0.309` only as evidence that assigning
new official sides to Train centroids is invalid.

### Metric-policy clarification from the follow-up adjudication

The later `0.930/0.929/0.944` values are also reproducible, but they are ARI
after excluding every recording HDBSCAN marked as noise. Coverage is 97.28%,
96.69% and 97.59% for Train, Development and Test. Likewise its pooled `0.373`
excludes noise, whereas `0.309` includes all points. These numbers cannot be
compared without stating the noise policy.

Use the conservative all-point ARI (`0.856/0.829/0.879`) as the primary result
and the non-noise-only ARI (`0.930/0.929/0.944`) as a disclosed sensitivity
analysis. The original `0.856` provenance is not murky: it is raw-space Train,
cell 14 of `model/test.ipynb`. Do not retire it in favor of an undisclosed
filtered metric.

The final paper table should retain both disputed all-recording values:
`0.856` as the Train-local KMeans--HDBSCAN comparison and `0.309` as the
pooled shipped-map mismatch. The contrast is useful only when the compared
objects and noise policy are explicit.

Two further labels matter. First, the comprehensive KMeans stability statistic
is the mean over all 15 pairs from six seeds: `0.889/0.897/0.888` for
Train/Development/Test. Values such as `0.888/0.885/0.908` describe only one
seed pair and should not headline stability. Second, HDBSCAN recovering
`204/205/205` clusters is concordant with the chosen `k=210` prior, but 210 is
not a ground-truth side-level speaker count in the released labels.

### The side-local result does not validate the historical Development split

The follow-up agent made a consequential logical jump: agreement between two
methods fitted side-locally shows that ECAPA contains stable speaker structure;
it does not make the old Train-centroid IDs valid on Development.

`audit_shipped_group_overlap.py` reproduced the exact historical
`stratified_grouped_split` and audited its boundaries using the independently
fitted Development-local KMeans labels. On the canonical Development split:

| Audit view | Overlapping proxy groups | Affected recordings |
|---|---:|---:|
| Historical Train-centroid IDs | 0 | 0/9,596 |
| Independent Development-local IDs | **201/210** | **9,399/9,596 (97.95%)** |

Across canonical plus ten shadow seeds, 191--202 Development-local groups cross
the boundary and 94.40--97.95% of recordings belong to a crossing group. The
side-local positive-control split gives 0 overlap. On Train, where the shipped
partition was actually fitted, the historical seed-42 split also gives 0
overlap under the independently recovered Train-local partition.

Therefore the paper must not call the old `devel_val/devel_test` partitions
speaker-disjoint. The old IDs report zero overlap by construction while
fragmenting the held-side speaker structure across both halves. This does not
invalidate whole-side Train-to-Development evaluation or corrected side-local
grouped tests.

## 2. TRILLsson is closed as a speaker-proxy replacement

The agents' interpretation matches the completed controls: ECAPA substantially
outperforms TRILLsson1 and TRILLsson5 on known-speaker recovery, especially on
MLS German. TRILLsson-to-ECAPA agreement on URTIC is near TRILLsson's own seed
stability, which corroborates ECAPA's structure while demonstrating that
TRILLsson is the noisier identity view.

Do not replace ECAPA, fuse the embeddings, or feed either proxy ID into the
health predictor.

## 3. The openSMILE negative result is real, but the universal claim is too strong

The frozen rerun reproduced:

| Candidate | Grouped Train UAR | Fold SD | ECAPA-proxy-ID top-1 |
|---|---:|---:|---:|
| ComParE-2016, 6,373-d | 0.5507 | 0.0530 | 0.875 |
| eGeMAPS, 88-d | 0.5383 | 0.0734 | 0.859 |

On fixed official Train-to-Development evaluation, adding ComParE as an equal
third z-normalized branch reduced AUC from 0.7710 to 0.7475.

This supports: "the tested balanced-logistic full-ComParE branch and fixed
equal fusion did not improve the frozen pipeline." It does not establish an
"airtight feature-invariant ceiling." The reported SD is across five folds of
one partition, proxy-ID recovery is not true-speaker accuracy, and the test did
not cover every classifier or fusion rule. The negative submission decision is
stronger than the universal scientific claim.

## 4. Correct the CNN baseline statement

The official ComParE 2017 paper does not report a spectrogram CNN at
Dev/Test 62.6/64.8. It reports raw-waveform CNN+LSTM standalone Cold UAR of
59.1/60.0 for two LSTM layers and 58.6/59.6 for three layers. The 62.6/64.8 row
is late fusion of the end-to-end system with ComParE functionals.

Reference: [Schuller et al., Interspeech 2017, Table 3](https://www.isca-archive.org/interspeech_2017/schuller17_interspeech.pdf).

A new CNN still should not be introduced for this submission: it would be a new
architecture after extensive Development reuse and could not be selected
honestly. It remains a future pre-registered experiment, not an already
identically tested method.

The corrected organizer result also does not prove that deep learning in
general cannot exceed a universal task ceiling. It shows only that the reported
raw-wave CNN+LSTM configurations scored about 0.59--0.60 under their protocol.
That is relevant negative prior evidence, not an architecture-class theorem or
an apples-to-apples comparison with the current whole-side G4+CQT diagnostics.

## 5. Human-pair audit prepared

A blinded 300-pair case-control audit has been generated:

- 100 pairs from each official side;
- 60 pairs in each of five method-agreement/disagreement strata;
- 600 unique recordings, with no recording reused;
- public CSV and browser interface contain no method predictions;
- two independent labels, `same`, `different` or `unsure`, plus confidence;
- hidden key opened only after both annotations are frozen;
- scorer reports Cohen's kappa and method balanced accuracy on the agreed,
  decisive cases.

Because disagreement cases are deliberately enriched, the resulting method
accuracy is not a population prevalence estimate. The audit is characterization
only and cannot alter the frozen submission.

## Next action

Do not reopen the frozen G4+CQT architecture: its decisive point estimates come
from whole-side Train-to-Development and Development-to-Train evaluation, and
its final OOF threshold uses explicit side-local groups. Do rerun or retire any
WavLM-paper result whose selection or evaluation depends on the historical
within-Development split, and remove its speaker-disjoint wording immediately.

The blinded human-pair kit can be parked as a response-to-reviewers asset. It is
not required to choose the submission, but it remains the only prepared check
that is independent of agreement among clustering algorithms.
