# Reconciled architecture recommendation

## Outcome

Do not freeze G4 alone based on the original single-partition rerun. Keep CQT and use the following as the primary expected-score candidate:

```text
g4_logit = balanced logistic regression on 7 gain-invariant G4 features
g9_logit = balanced logistic regression on 168 CQT features

z4 = z-score(g4_logit; parameters fitted on the final training pool)
z9 = z-score(g9_logit; parameters fitted on the final training pool)

final_score = 0.5 * (z4 + z9)
prediction = Cold if final_score >= tau else Non-Cold
```

This architecture has no learned fusion weight, no neural branch and no seed ensemble. The final calibration check below gives a narrow expected-score preference to a repeated grouped-OOF threshold (`tau = 0.319184`) over the parameter-free `tau = 0` fallback.

## Repeated Train-only grouped CV

| Model | Mean UAR | Split-seed SD | Six split-seed values |
|---|---:|---:|---|
| G4 | 0.601 | 0.005 | 0.603, 0.603, 0.601, 0.605, 0.605, 0.592 |
| CQT/G9 | 0.610 | 0.022 | 0.602, 0.636, 0.631, 0.598, 0.616, 0.579 |
| Fixed equal G4+CQT | **0.623** | 0.012 | 0.612, 0.627, 0.637, 0.617, 0.637, 0.608 |

The fusion beat G4 in all six partitions. Its mean gain was +0.0216 UAR, but the repaired paired inferred-cluster bootstrap interval was [-0.0157, +0.0549]. Therefore it is the better expected-score candidate, not a statistically proven subject-level winner.

## Why the earlier numbers moved

- A5j tested G4+G5, not G4+CQT. It cannot establish CQT stability.
- The actual CQT audit A5g ranged from 0.578 to 0.672 across split seeds (SD 0.039). Its 0.674 headline was the favorable seed-42 half of Development after that candidate family had already been inspected on Development.
- The new single outer split seed was pessimistic for CQT: 0.579 with a fixed threshold, versus 0.610 across six repeated partitions.
- Training on 80% versus 90% of Train produced 0.6230 versus 0.6236 for fixed G4+CQT. Reduced outer-fold training data does not explain the drop.
- `liblinear` versus historical `lbfgs` changed CQT UAR by less than 0.001. Nested C tuning did not help. Nested threshold tuning changed CQT by only +0.004 and did not explain the drop.
- The exact corrected outer folds had 5–8 inferred clusters containing cold chunks, not three. The old 10% threshold subsets had only 3–5 such clusters and were substantially noisier.
- The earlier learned stack and inner-selected threshold destroyed complementarity. Fixed equal fusion recovers it without learning another parameter.

## Why 0.59 was below the 0.6205 hidden score

The 0.59 figure was G4 alone under one pessimistic partition, not a ceiling for the complete pipeline. Repeated fixed G4+CQT estimates 0.623, almost exactly the prior hidden score. The 80%/90% diagnostic rules out material pessimistic bias from smaller training folds.

The hidden score nevertheless belongs to a different submitted model and one finite Test draw, so it should be treated as corroboration rather than a validation target.

## Remaining limitations

- True subject IDs are unavailable. The k=210 clusters are proxies, and 15/210 contain both labels, so neither grouped CV nor its bootstrap is a true subject-level analysis.
- Marginal confidence-interval overlap is not a pairwise test. The repaired paired tests show no candidate has a conclusive subject-cluster-level advantage over G4.
- CQT is more partition-sensitive than G4. If minimizing variance is more important than expected UAR, retain G4 alone as the fallback.

## Combined Train+Development monolithic-versus-ensemble test

After the architecture above was frozen, it was evaluated with six repetitions of five outer grouped folds on combined Train+Development. Official Train and Development group IDs were kept disjoint. Each untouched outer fold compared a monolithic fit against a ten-model inner grouped-fold ensemble.

| Model | Repeated outer-CV UAR | Split-seed SD |
|---|---:|---:|
| Monolithic G4 | 0.6224 | 0.0032 |
| 10-fold ensemble G4 | 0.6222 | 0.0035 |
| Monolithic fixed G4+CQT | **0.6665** | 0.0087 |
| 10-fold ensemble fixed G4+CQT | 0.6668 | 0.0082 |

The monolithic fusion gain over monolithic G4 was +0.0440 with a paired inferred-cluster bootstrap interval of [+0.0159, +0.0724]. The ensemble gain over monolithic fusion was only +0.0003 with interval [-0.0005, +0.0011]. Use the monolithic fusion; the ensemble adds no measurable benefit.

This combined-data CV is a data-scaling diagnostic, not the unknowable Test UAR of the full refit. Development was used repeatedly earlier in the project, inferred groups are imperfect, and the final deployment uses 100% rather than 80% of the combined pool. In particular, 0.592 is neither a guaranteed lower bound nor the expected deployment score.

## Evaluation-independent grouping challenge

The combined-CV grouping was fitted on the full combined feature pool, so its result was challenged as potentially transductive. The decisive follow-up removed pseudo-grouping from model fitting and splitting entirely: models were fitted on one official speaker-disjoint side and evaluated on the other with fixed architecture and threshold.

| Direction | G4 | CQT | Fixed G4+CQT | Fusion gain vs G4 |
|---|---:|---:|---:|---:|
| Train → Development | 0.6422 | 0.6417 | **0.6897** | +0.0475 |
| Development → Train | 0.6132 | **0.6679** | 0.6468 | +0.0336 |
| Bidirectional mean | 0.6277 | 0.6548 | **0.6683** | +0.0405 |

Pseudo groups were used only after prediction for uncertainty resampling. Under the k210 proxy map, the bidirectional paired interval for fusion minus G4 was [+0.0090, +0.0698]; under the pooled-k420 proxy map it was [+0.0055, +0.0752]. The CQT contribution therefore survives the evaluation-independent test. It is not explained by clustering the outer-evaluation pool.

The apparently contradictory 0.529 `fusion_G4_G9` result used a learned logistic stack and an inner-selected threshold. On the exact same pessimistic split, fixed equal fusion with threshold zero scored 0.6076. The structural difference is the fusion rule, not merely the grouping.

## Evaluation-independent threshold transfer

The remaining question was whether threshold zero wastes the fusion's threshold-free separation. This was tested without using the opposite official side for fitting, grouping or threshold selection. For each direction, ECAPA proxy speakers were clustered only within the fitting side; six repeated five-fold grouped-OOF runs estimated UAR-optimal thresholds; their median was then carried unchanged to the untouched official side.

| Direction | Fusion at zero | OOF-median threshold | Change |
|---|---:|---:|---:|
| Train to Development | 0.6897 | 0.6948 | +0.0051 |
| Development to Train | 0.6468 | 0.6626 | +0.0158 |
| Bidirectional mean | 0.6683 | **0.6787** | +0.0104 |

The fusion AUC also exceeded CQT AUC in both directions: +0.0378 and +0.0329. At deployment thresholds, the OOF-calibrated fusion exceeded raw-threshold CQT by +0.024 on the bidirectional mean. Proxy-cluster paired intervals nevertheless crossed zero: calibrated fusion minus threshold-zero fusion was [-0.0053, +0.0270], and calibrated fusion minus CQT was [-0.0086, +0.0556]. The calibration is therefore a modest expected-score choice, not a statistically established winner.

Matching a 43% predicted-Cold rate was rejected as a selection principle. UAR weights class recalls equally but does not imply any target prediction prevalence. The rate-matched calculation remains diagnostic only.

## Final fitted candidates

The frozen monolithic and ensemble systems were fit on all 19,101 Train+Development chunks. Their Test predictions agree on 99.47% of files. The monolithic fusion predicts 51.37% Cold, consistent with its 51.12% mean Cold rate in combined-data OOF predictions. The previous submission predicted 43.18% Cold.

Applying the frozen repeated grouped-OOF threshold rule to Train+Development produced six threshold estimates from 0.211 to 0.399 (median 0.319184). The resulting non-destructive candidate predicts 36.92% Cold and differs from the threshold-zero fusion on 1,380/9,551 Test files. All changes are Cold to Non-Cold. This large decision change is why the threshold-zero CSV remains an important fallback despite the calibrated candidate's higher cross-direction point estimate.

## Final-fit rule

Use the already-fitted monolithic two-branch system with equal fusion. For one expected-score submission, the repeated grouped-OOF threshold candidate (`tau = 0.319184`) has the narrow evidence-based preference. If two slots are available, submit the threshold-zero fusion as the calibration hedge. Keep G4-only as the architecture/identity-confound fallback if a third slot is available. Do not select further weights, thresholds, architectures or seeds on Development.

## Impact of the historical Development-group audit

The shipped Train-centroid k210 IDs do not make the old within-Development
split speaker-proxy disjoint: an independent Development-local partition finds
201/210 groups crossing the canonical boundary and 97.95% of recordings in a
crossing group. This invalidates the old WavLM paper's speaker-disjoint wording,
not the final architecture decision above.

The load-bearing G4+CQT point estimates were obtained by fitting on one whole
official side and evaluating on the other without pseudo groups. Its threshold
transfer and final OOF candidate explicitly refit KMeans separately within each
official side. Repeated Train-only CV is also unaffected because the shipped
partition was fitted on Train and independently reproduces zero canonical
Train overlap. Treat the pooled combined-data CV as a secondary scaling
diagnostic only.
