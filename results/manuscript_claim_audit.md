# Manuscript claim audit after the Development-split verification

## Verdict

The Development split retraction is valid, but the proposed `+0.004 UAR leak`
guard is not. The independently reproduced simplified-fusion diagnostic scores
`0.6317` under the historical grouping and `0.6353` under corrected pooled
grouping: corrected minus historical is `+0.0036`, opposite to leakage optimism,
and the corrected-grouping across-seed SD is `0.0317`. A separate leaked-versus-
clean subset comparison also finds the nominally leaked subset harder, but that
observational subset contrast is not a causal estimate either.

The defensible wording is therefore: the historical Development boundary was
not disjoint under the side-local ECAPA proxy; retract that claim, but do not
apply or imply a numerical UAR correction from the crossing diagnostics.

## Five requested claim checks

1. **ECAPA ARI: pass after tightening.** The paper reports all-recording ARI
   `0.856/0.829/0.879`, non-noise ARI `0.930/0.929/0.944` with coverage,
   HDBSCAN `204/205/205`, six-seed stability `0.889/0.897/0.888`, the pooled
   mismatch `0.309`, and external MLS-German known-speaker ARI `0.963`.
2. **Feature ceiling: pass.** The active manuscript does not claim an airtight
   or feature-invariant universal ceiling. Any result statement must remain
   scoped to the tested feature/classifier/fusion combinations.
3. **CNN citation: pass by absence.** The active manuscript does not make the
   incorrect standalone `64.8` claim. If added later, the organizer paper's
   standalone raw-wave CNN+LSTM values are about `0.59--0.60`; `0.648` is late
   fusion. This is negative prior evidence, not proof about all CNNs.
4. **Development split: corrected.** The paper reports `201/210`, `97.95%` and
   the `27.46%` cross-boundary nearest-neighbour diagnostic, explicitly says
   these do not estimate or bound UAR optimism, and applies no score correction.
5. **Firewall: pass.** The paper states that cold labels must not select the
   speaker embedding, clustering method or `k`, and that proxy groups are fitted
   independently within the official side where they are used.

## Submission consequence

The frozen G4+CQT submission remains unchanged. Its load-bearing evidence is
whole-official-side Train-to-Development and Development-to-Train evaluation,
and its final threshold grouping is side-local. Historical WavLM results remain
same-partition exploratory evidence unless rerun under the corrected protocol.
