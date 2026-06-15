"""Build the mid-term post-mortem .pptx on the HLU TUM template.

Run from project root with the datascience env:
  & "$env:USERPROFILE\AppData\Local\miniconda3\envs\datascience\python.exe" presentation/build_pptx.py

Inputs:
  HLU_Presentation_Template_Oct_2022_Modified_by_Tian.pptx (project root)
  presentation/figures/fig{1..7}_*.png

Output:
  presentation/MidTerm_Postmortem_FINAL_PLAN.pptx

Strategy:
  - Start from a copy of the template (so master/theme + Corporate Design colors persist).
  - Delete the 27 sample slides (preserves the masters).
  - Add 7 new slides via the appropriate layouts:
      slide 1  Title              -- master[0] / '1_Start'
      slide 2  Hidden-test result -- master[4] / '1_große Bilder' (title + body + image)
      slide 3  Diagnosis          -- master[4] / '1_große Bilder'
      slide 4  Current arch       -- master[4] / '1_große Bilder'
      slide 5  Failure-mode rank  -- master[4] / '1_große Bilder'
      slide 6  Proposed arch      -- master[4] / '1_große Bilder'
      slide 7  Plan + closer      -- master[4] / '4_Inhalt + Text' (bullets only)
  - Override the master's "Dr. rer. nat. Erika Mustermann" footer with the user's info.

Each content slide gets:
  - title       (placeholder idx=0)
  - subtitle    (placeholder idx=14 or 18, depending on layout -- BODY type)
  - figure      (placeholder idx=17 PICTURE for '1_große Bilder')
  - speaker notes (used during delivery; live in slide.notes_slide.notes_text_frame)
"""
from __future__ import annotations
import copy
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT       = Path(__file__).resolve().parent.parent
TEMPLATE   = ROOT / "HLU_Presentation_Template_Oct_2022_Modified_by_Tian.pptx"
FIG_DIR    = ROOT / "presentation" / "figures"
OUT_PPTX   = ROOT / "presentation" / "MidTerm_Postmortem_FINAL_PLAN.pptx"

# --- Header / footer / metadata personalisation ------------------------------
AUTHOR_FOOTER = "Christoph Feldkircher, Tony Lee, Sai Sashank Chouksey  |  TUM HLU 2026  |  Cold Detection on URTIC"
PRESENTATION_TITLE   = "Cold Detection on URTIC"
PRESENTATION_SUBTITLE = "Mid-term result & plan for the final submission"
PRESENTATION_DATE     = "Munich, 16 June 2026"

NSMAP = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


# -----------------------------------------------------------------------------
def _strip_existing_slides(prs: Presentation) -> None:
    """Delete every existing slide (keeps masters/layouts/theme intact)."""
    sldIdLst = prs.slides._sldIdLst
    rId_to_drop = [sldId.get(f"{{{NSMAP['r']}}}id") for sldId in list(sldIdLst)]
    for sldId in list(sldIdLst):
        sldIdLst.remove(sldId)
    part = prs.part
    for rId in rId_to_drop:
        try:
            part.drop_rel(rId)
        except KeyError:
            pass


def _get_layout(prs: Presentation, master_idx: int, layout_name: str):
    master = prs.slide_masters[master_idx]
    for layout in master.slide_layouts:
        if layout.name == layout_name:
            return layout
    raise KeyError(f"layout {layout_name!r} not in master[{master_idx}]")


def _placeholder_by_idx(slide, idx: int):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            return ph
    return None


def _placeholder_by_name(slide, name_substr: str):
    for ph in slide.placeholders:
        if name_substr.lower() in ph.name.lower():
            return ph
    return None


def _set_text(tf, text: str, font_size: int | None = None, bold: bool = False,
              color: tuple[int, int, int] | None = None) -> None:
    """Replace text in a text frame with a single styled paragraph."""
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    if font_size is not None:
        run.font.size = Pt(font_size)
    if bold:
        run.font.bold = True
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def _add_bullets(tf, items: list[str], font_size: int = 16, bullet_color=(0, 0, 0)) -> None:
    """Replace text frame content with a styled bullet list.

    The HLU master already has bullet glyphs via the layout's text style, so we
    rely on the layout for the bullet character and just write each line as a
    paragraph at level 0.
    """
    tf.clear()
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.level = 0
        # If the bullet text contains an em-dash + sub-text, split into runs so the
        # sub-text reads as a continuation, not an emphatic point. Keep simple here:
        # one run per paragraph, master style controls the bullet.
        run = p.add_run()
        run.text = item
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(*bullet_color)


def _set_footer(slide, text: str) -> None:
    """Add an explicit footer text-box so the master's placeholder text doesn't leak through.

    The HLU template's layouts inherit a German footer string from the master
    ("Dr. rer. nat. Erika Mustermann | ..."). python-pptx does not auto-clone
    that placeholder onto new slides, so the master text shows through by
    inheritance. To suppress it cleanly, we drop a thin opaque white text-box
    over the master's footer rect (L=0.34 T=5.31 W=7.07 H=0.30 inches) and put
    OUR footer text inside.
    """
    box = slide.shapes.add_textbox(Inches(0.34), Inches(5.31),
                                    Inches(8.50), Inches(0.30))
    # Opaque white fill to mask the inherited master footer beneath
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(255, 255, 255)
    box.line.fill.background()
    tf = box.text_frame
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.word_wrap = True
    _set_text(tf, text, font_size=9, color=(110, 110, 110))


def _replace_picture_placeholder(slide, ph, image_path: Path) -> None:
    """Delete the picture placeholder and add the image as a free, height-capped shape.

    The HLU template's picture placeholder is 10 x 3.68 inches (aspect 2.72).
    Our figures range from ~1.77 (fig1) to ~2.29 (fig4) -- all TALLER than the
    placeholder. python-pptx's `.insert_picture` aspect-fits, causing tall figures
    to overflow the slide vertically. We bypass it entirely: remove the
    placeholder, then add the picture as a free shape centered horizontally, with
    explicit height capped at 3.5 inches (so it ends by y=5.0 and leaves room for
    the footer at y=5.31).
    """
    from PIL import Image
    sp = ph._element
    sp.getparent().remove(sp)
    with Image.open(str(image_path)) as im:
        img_w, img_h = im.size
    aspect = img_w / img_h
    top_in = 0.95
    max_h_in = 4.20
    max_w_in = 9.60
    # Fit by whichever bound binds first
    h_by_w = max_w_in / aspect
    if h_by_w <= max_h_in:
        h_in = h_by_w
        w_in = max_w_in
    else:
        h_in = max_h_in
        w_in = max_h_in * aspect
    left_in = (10.0 - w_in) / 2.0
    slide.shapes.add_picture(str(image_path),
                              Inches(left_in), Inches(top_in),
                              width=Inches(w_in), height=Inches(h_in))


def _set_speaker_notes(slide, text: str) -> None:
    notes = slide.notes_slide.notes_text_frame
    notes.text = text


# -----------------------------------------------------------------------------
def _add_title_slide(prs: Presentation, layout, title: str, subtitle_lines: list[str],
                     speaker_notes: str = "") -> None:
    slide = prs.slides.add_slide(layout)
    title_ph = _placeholder_by_idx(slide, 0)
    if title_ph is not None:
        _set_text(title_ph.text_frame, title, font_size=30, bold=True, color=(0, 51, 89))
    content_ph = _placeholder_by_idx(slide, 10)
    if content_ph is not None:
        # title-layout content placeholder = subtitle / author block
        content_ph.text_frame.clear()
        for i, line in enumerate(subtitle_lines):
            p = content_ph.text_frame.paragraphs[0] if i == 0 else content_ph.text_frame.add_paragraph()
            r = p.add_run()
            r.text = line
            r.font.size = Pt(16 if i == 0 else 13)
            r.font.color.rgb = RGBColor(60, 60, 60)
    _set_footer(slide, AUTHOR_FOOTER)
    if speaker_notes:
        _set_speaker_notes(slide, speaker_notes)


def _add_image_slide(prs: Presentation, layout, title: str, subtitle: str,
                     image_path: Path, speaker_notes: str = "") -> None:
    slide = prs.slides.add_slide(layout)
    title_ph = _placeholder_by_idx(slide, 0)
    if title_ph is not None:
        _set_text(title_ph.text_frame, title, font_size=22, bold=True, color=(0, 0, 0))
    # IMPORTANT: do NOT write to the subtitle (body) placeholder on image slides --
    # its default position (T=1.34, H=0.55) collides with the picture area, so it
    # ends up overlapping the figure's own title. The matplotlib figure carries
    # the relevant numbers internally.
    _ = subtitle  # kept in signature for caller-side documentation
    # picture placeholder
    pic_ph = None
    for ph in slide.placeholders:
        if "PICTURE" in str(ph.placeholder_format.type):
            pic_ph = ph
            break
    if pic_ph is None:
        # fallback: pick the Bildplatzhalter name
        pic_ph = _placeholder_by_name(slide, "bildplatzhalter")
    if pic_ph is not None:
        _replace_picture_placeholder(slide, pic_ph, image_path)
    else:
        # Last-ditch: drop the picture in at the master's image rect
        slide.shapes.add_picture(str(image_path), Inches(0.0), Inches(1.94),
                                  width=Inches(10.0))
    _set_footer(slide, AUTHOR_FOOTER)
    if speaker_notes:
        _set_speaker_notes(slide, speaker_notes)


def _add_bullets_slide(prs: Presentation, layout, title: str, subtitle: str,
                       bullets: list[str], speaker_notes: str = "") -> None:
    slide = prs.slides.add_slide(layout)
    title_ph = _placeholder_by_idx(slide, 0)
    if title_ph is not None:
        _set_text(title_ph.text_frame, title, font_size=22, bold=True, color=(0, 0, 0))
    body_ph = _placeholder_by_name(slide, "textplatzhalter") \
              or _placeholder_by_name(slide, "text placeholder")
    if body_ph is not None and subtitle:
        _set_text(body_ph.text_frame, subtitle, font_size=14, color=(70, 70, 70))
    content_ph = _placeholder_by_name(slide, "inhaltsplatzhalter") \
                 or _placeholder_by_idx(slide, 1)
    if content_ph is not None:
        _add_bullets(content_ph.text_frame, bullets, font_size=16)
    _set_footer(slide, AUTHOR_FOOTER)
    if speaker_notes:
        _set_speaker_notes(slide, speaker_notes)


# -----------------------------------------------------------------------------
def main() -> None:
    assert TEMPLATE.exists(), f"template not found: {TEMPLATE}"
    for k in (1, 2, 4, 6, 7):
        f = FIG_DIR / f"fig{k}_*.png"
        # use glob to be tolerant of the suffix
    figs = {p.stem.split("_")[0]: p for p in sorted(FIG_DIR.glob("fig*.png"))}
    for k in ("fig1", "fig2", "fig4", "fig6", "fig7"):
        assert k in figs, f"missing figure {k} in {FIG_DIR}"

    prs = Presentation(str(TEMPLATE))
    _strip_existing_slides(prs)

    title_layout  = _get_layout(prs, 0, "1_Start")
    image_layout  = _get_layout(prs, 4, "1_große Bilder")
    bullet_layout = _get_layout(prs, 4, "4_Inhalt + Text")

    # ---- Slide 1 -- Title ---------------------------------------------------
    _add_title_slide(
        prs, title_layout,
        title=PRESENTATION_TITLE,
        subtitle_lines=[
            PRESENTATION_SUBTITLE,
            "",
            "Christoph Feldkircher  |  Tony Lee  |  Sai Sashank Chouksey",
            "Technical University of Munich  --  Human-Like Understanding (HLU) 2026",
            PRESENTATION_DATE,
        ],
        speaker_notes=(
            "Open with the headline number and frame the talk. "
            "We submitted the locked multi-K system to the ComParE 2017 URTIC Cold "
            "leaderboard; it scored UAR 0.6205 -- well below our shadow-mean. "
            "The next three minutes are about WHY, and what we change for the "
            "end-of-semester second submission."
        ),
    )

    # ---- Slide 2 -- Hidden-test result --------------------------------------
    _add_image_slide(
        prs, image_layout,
        title="Hidden-test result: we predict cold 4x too often",
        subtitle=("UAR 0.6205  |  Acc 0.596  |  MacroP 0.542  |  MacroF1 0.479  "
                  "--  shadow-mean was 0.6940 +/- 0.0157, devel was 0.7111"),
        image_path=figs["fig1"],
        speaker_notes=(
            "Hidden-test confusion matrix. We called COLD on 4124 out of 9551 "
            "utterances -- 43.2 percent. The true rate is 9.4 percent. "
            "UAR landed at 0.6205. Our internal validation -- the shadow-mean over "
            "ten alternative speaker-disjoint partitions -- was 0.6940 plus minus "
            "0.0157. So we are about five times our own uncertainty estimate below "
            "where we expected to be. "
            "We audited the pipeline five ways: checkpoints, betas, features, "
            "threshold derivation, submission CSV -- all bit-faithful. So this "
            "isn't a bug. The question for the next two minutes: what kind of "
            "failure is this?"
        ),
    )

    # ---- Slide 3 -- Diagnosis -----------------------------------------------
    _add_image_slide(
        prs, image_layout,
        title="The ranker survived; the threshold did not",
        subtitle=("UAR > chance means the model still RANKS cold above non-cold; "
                  "MacroF1 at trivial-NC baseline means the operating point is wrong"),
        image_path=figs["fig2"],
        speaker_notes=(
            "Here is the test multi-K ensemble logit distribution. The red dashed "
            "line is where we locked tau -- minus 1.625, tuned on one 10 percent "
            "stratified-grouped train holdout. The green line is where tau would "
            "need to be to make our predicted cold-rate match the published 9.4 "
            "percent prior: PLUS 7.09. That's a gap of 8.7 logit units. "
            "The model still ranks cold above non-cold -- UAR is above chance -- "
            "but the operating point is in the wrong place by close to an order "
            "of magnitude. The ranker survived; the threshold did not."
        ),
    )

    # ---- Slide 4 -- Current architecture ------------------------------------
    _add_image_slide(
        prs, image_layout,
        title="What we used  --  the locked multi-K pipeline",
        subtitle=("Frozen WavLM-Large + audit-derived A2.5 head x 5 seeds + K1/K2 late fusion "
                  "with G4_gi (7-d) + G5_mod (64-d) + 5-seed ensemble + single-split tau"),
        image_path=figs["fig6"],
        speaker_notes=(
            "This is the system we shipped. Frozen WavLM-Large, audit-derived "
            "layer-weighted A2.5 head trained with 5 seeds, K1 and K2 late fusion "
            "with the gain-invariant G4 and the modulation-spectrogram G5 "
            "handcrafted feature groups, 5-seed mean-logit ensemble, and finally "
            "tau equal to minus 1.625. "
            "Everything to the LEFT of the threshold box is shadow-validated and "
            "byte-identical to the paper headline. Everything FAILS at that final "
            "red box. The threshold was tuned on a single 10-percent train holdout "
            "and on hidden test the logits sit shifted, producing a 34-percentage-"
            "point over-prediction of cold."
        ),
    )

    # ---- Slide 5 -- Failure-mode ranking ------------------------------------
    _add_image_slide(
        prs, image_layout,
        title="Failure-mode ranking after adversarial critic pass",
        subtitle=("Calibration drift dominates; speaker confound and feature-shift are "
                  "credible but second-order; the bottom two are noise"),
        image_path=figs["fig4"],
        speaker_notes=(
            "Five candidate failure modes, ranked by posterior probability after "
            "an adversarial critic forced us to de-conflate them. Calibration / "
            "scale drift sits at 0.42 -- the operating-point story you just saw. "
            "Speaker confound at 0.24, because our negative-control set already "
            "showed that representation-level speaker debiasing fails under the "
            "frozen-backbone constraint. Feature-distribution shift at 0.22, "
            "credible but we have never actually measured it on disk. The bottom "
            "two -- single-split tau variance and ensemble degradation -- are "
            "essentially noise: the shadow-sigma is too small and the shadow "
            "protocol itself already ensembles robustly."
        ),
    )

    # ---- Slide 6 -- Proposed architecture (SCOP) ----------------------------
    _add_image_slide(
        prs, image_layout,
        title="What we change  --  Shadow-Calibrated Operating Point (SCOP)",
        subtitle=("Replace the single-split tau with a shadow-aggregated, prior-corrected "
                  "calibrator -- fit on speaker-disjoint shadow folds, NEVER on test labels"),
        image_path=figs["fig7"],
        speaker_notes=(
            "Our second submission. The locked pipeline stays byte-identical -- "
            "same WavLM, same A2.5 head, same betas, same checkpoints. What we "
            "replace is the final block: a single tau becomes SCOP. "
            "Move one: a Bayes prior-correction toward the published 9.4 percent "
            "cold prior. That's a CONSTANT, not a tuned hyperparameter. "
            "Move two: a calibrator -- Platt, isotonic, or quantile-match -- fit "
            "on ten speaker-disjoint shadow folds, selected by leave-one-partition-"
            "out with a stability gate. The acceptance rule is pre-registered in "
            "shadow units only: mean shadow UAR lift at least plus 0.015, and no "
            "fold loses more than 0.005. The hidden-test labels we now have are "
            "used ONLY for the post-mortem confusion matrix you saw on slide 2 -- "
            "never to choose tau or the calibrator family."
        ),
    )

    # ---- Slide 7 -- Plan & lesson -------------------------------------------
    _add_bullets_slide(
        prs, bullet_layout,
        title="Audit-and-recalibrate, not retrain",
        subtitle="The lesson: once a five-check audit rules out bugs, the failure signature IS the diagnosis",
        bullets=[
            "SCOP attacks the dominant operating-point failure with one degree of freedom",
            "Frozen backbone, locked betas -- paper headline byte-identical",
            "Pre-registered shadow gate: mean lift >= +0.015, no fold loses > 0.005",
            "Label-free TTA-Z runs in parallel as a feature-shift hedge",
            "Cost: ~15 days; risk: medium-low; fallback to submission_1 is built in",
        ],
        speaker_notes=(
            "To close. We've split what looked like one big calibration-drift mode "
            "into four shadow-testable sub-modes. SCOP attacks the dominant one "
            "with a single new degree of freedom. The frozen backbone and locked "
            "betas mean the paper's M1 through M19 ablations stand untouched. "
            "A label-free feature-shift audit -- TTA-Z -- runs in parallel as a "
            "hedge. "
            "The lesson is the framing: once a five-check audit rules out bugs, "
            "the failure signature itself becomes the diagnosis. Questions."
        ),
    )

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    sz = OUT_PPTX.stat().st_size
    print(f"[done] wrote {OUT_PPTX}  ({sz/1024:.1f} KB)")
    print(f"[done] {len(prs.slides)} slides in deck")


if __name__ == "__main__":
    main()
