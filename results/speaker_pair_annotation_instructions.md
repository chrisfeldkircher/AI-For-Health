# Blinded URTIC speaker-pair annotation

## Firewall

- Do not open `speaker_pair_annotation_key.json` until both annotators have
  independently exported their completed CSVs.
- Judge only whether A and B sound like the same person. Ignore cold symptoms,
  lexical content, recording quality and the expected class.
- No cold labels were used to construct these pairs.
- The audit is characterization only and must not change the already-frozen
  submission predictions, clustering choice or health-model architecture.

## Annotation

1. From the repository root run `python -m http.server 8000`.
2. Open `http://localhost:8000/results/speaker_pair_annotation.html`.
3. Enter an annotator code. Listen to both recordings as often as needed.
4. Choose Same speaker, Different speakers or Unsure; add confidence 1–3.
5. Download the CSV when finished. Each annotator must work independently.

Use Unsure rather than guessing when a clip is too short or distorted. Do not
discuss individual pairs until both files have been frozen. The 300 pairs are a
disagreement-enriched case-control set, so later balanced accuracy describes
this hard audit set—not the prevalence-weighted accuracy over all possible
URTIC pairs.
