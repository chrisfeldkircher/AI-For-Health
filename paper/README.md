# Paper: Honesty-Audited Late Fusion for Cold Detection

LaTeX source for the URTIC cold-detection write-up, structured for the TUM Hauptseminar IEEE-style template (`latex8.sty`/`latex8.bst`/`latex8.bib`, IEEE Computer Society 2-column conference style).

## Layout

```text
paper/
  main.tex              top-level document; uses \usepackage{latex8}; \input's section files
  references.bib        BibTeX bibliography (cite-key-keyed)
  sections/
    01_introduction.tex
    02_related_work.tex
    03_background.tex
    04_method.tex
    05_deconfounding.tex
    06_results.tex
    07_discussion.tex
    08_conclusion.tex
  README.md             this file
```

## Compiling

The TUM template ships as `Hauptseminar_IEEE_style_LaTeX.zip` from <https://www.ce.cit.tum.de/fileadmin/w00cgn/lmt/Templates/Hauptseminar_IEEE_style_LaTeX.zip>. Unzip into this `paper/` directory so `latex8.sty`, `latex8.bst` and friends sit alongside `main.tex`. Then:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or via `latexmk`:

```bash
latexmk -pdf main.tex
```

## Notes for editors

- All numerical results in the body cite from project files: `results/A2_grouped_honestprior.json`, `results/A5b_grouped_honestprior_betasweep_extended.json`, `results/A5b_k2_5seed_lock.json`, `results/A5b_k2_5seed_speaker_probes.json`, `results/A5b_k2_5seed_ensemble.json`, `results/A5b_k3_egemaps_5seed.json`, the A5.5/A6/A7/A7c JSON files. Don't change a number in the paper without updating the source JSON path in the relevant `\cite{}` or footnote.
- Methodological-contribution paragraphs (M8 through M14) trace to `EXPLAINER.md §14.1`. Editing one should bring its EXPLAINER counterpart along.
- The cumulative-stack table in `06_results.tex` is the load-bearing figure; it's reproduced verbatim from `plan.md §4.11.1.4` cumulative-final-state table.
- The de-confounding ladder closure (data / representation-head / representation-LW+adversary) is the negative-result spine. See `plan.md §4.10.1.3` for the v1→v2 worked example and `EXPLAINER.md §13` diary entries for the per-rung diagnostic chain.
