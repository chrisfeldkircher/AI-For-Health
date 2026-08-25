# URTIC course report

This directory contains the evaluation-aware rewrite of the Group 3 final
report. The compiled paper is **7 pages including references**, below the
course maximum of 8 pages. There is intentionally no appendix.

## Paper structure

- `main.tex`: ACM `acmart` conference template, title/authors, abstract and AI-use disclosure
- `sections/01_introduction.tex`: problem, turning point and research questions
- `sections/02_related_work.tex`: ComParE 2017, modern representations and confounding
- `sections/03_background.tex`: URTIC, UAR and the corrected evaluation design
- `sections/04_method.tex`: the three submitted model families
- `sections/05_deconfounding.tex`: hidden-Test and corrected diagnostic protocols
- `sections/06_results.tex`: five submissions, whole-side evidence and negative controls
- `sections/07_discussion.tex`: lessons learned, limitations, ethics and what to change
- `sections/08_conclusion.tex`: conclusion
- `references.bib`: bibliography
- `main.pdf`: compiled report

## Compile

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The current build has no unresolved citations, references or overfull boxes.
The remaining underfull-box messages are non-fatal line-spacing notices in the
two-column layout.

## Evidence sources

Load-bearing numbers are drawn from:

- `../presentation/ML4Health_Group3_Final_TUM_version_2.pdf`
- `../results/audit_evaluation_protocol.json`
- `../results/shipped_group_overlap_audit.json`
- `../results/fixed_g4_g9_repeated_cv.json`
- `../results/eval_independent_official_split_fusion.json`
- `../results/eval_independent_threshold_policy.json`
- `../results/corrected_outer_cv_foundations.json`
- `../results/reconciled_architecture_recommendation.md`

The full project-wide record of attempted models, controls, negative results,
evaluation caveats and lessons is in
`../EXPERIMENT_INVENTORY_AND_LEARNINGS.md`.

The pre-rewrite source tree is archived at
`../../paper_before_8page_rewrite_20260825.tar.gz`.

## Final author check

Before submission, the group should confirm the author order/affiliation,
course-specific filename requirements, and whether Moodle expects anonymous
or named submission. The generative-AI disclosure is included because the
course material requires disclosure when such tools are used.
