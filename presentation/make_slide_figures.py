"""Generate slide-ready figures for the 3-min mid-term post-mortem presentation.

Run from project root with the datascience env's Python:
  & "$env:USERPROFILE\AppData\Local\miniconda3\envs\datascience\python.exe" presentation/make_slide_figures.py

Outputs into presentation/figures/ as 300-DPI PNGs sized for slides.

What each figure says:
  fig1_confusion_matrix.png     -- 2x2 with 43% predicted-C vs 9.4% true prior callout
  fig2_logit_shift.png          -- histogram of the 9551 test multi-K logits with tau_locked
                                   and tau_prior_match (= empirical 90.6th percentile of test logits,
                                   i.e. the threshold whose predicted-C-rate matches the
                                   PUBLISHED challenge cold prior of 9.4%) overlaid
  fig3_scop_pipeline.png        -- locked-pipeline box -> SCOP box -> decision, with the
                                   "fit on shadow folds; never sees test labels" annotation
  fig4_failure_modes.png        -- ranked horizontal bar chart of the 5 modes after the
                                   adversarial critic pass
  fig5_pred_c_rate_vs_tau.png   -- predicted-cold rate as a function of tau on test
                                   (no labels needed); annotates where tau is and where it
                                   would need to move to match the published 9.4% prior

No labels used anywhere: tau_prior_match depends ONLY on the test logit empirical CDF and
the published challenge cold prior (0.094). This is faithful to the SCOP framing -- the
hidden-test labels we now have are NEVER used to fit or pick anything.
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV = ROOT / "results" / "test_predictions_multiK.csv"

# Locked constants
TAU_MULTIK   = -1.625
N_TEST       = 9551
N_C_TRUE     = 895      # backed out from leaderboard
N_NC_TRUE    = 8656
TRUE_C_PRIOR = N_C_TRUE / N_TEST    # 0.0937 -- matches the published challenge prior
PUB_PRIOR    = 0.094    # used by SCOP first move (challenge constant, not tuned)

# Slide-friendly defaults
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.95,
    "legend.fontsize": 11,
    "savefig.dpi": 300,
    "figure.dpi": 110,
})

C_RED   = "#c0392b"
C_GREEN = "#196f3d"
C_BLUE  = "#2980b9"
C_GREY  = "#7f8c8d"
C_AMBER = "#e67e22"


def _load_test_logits() -> np.ndarray:
    with PRED_CSV.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        out = np.array([float(r["ensemble_logit_multiK"]) for r in rdr], dtype=np.float64)
    assert out.size == N_TEST, f"expected {N_TEST} test logits, got {out.size}"
    return out


# -----------------------------------------------------------------------------
def fig1_confusion_matrix() -> None:
    cm = np.array([[582, 313], [3542, 5114]], dtype=int)  # rows: True C/NC, cols: Pred C/NC
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    im = ax.imshow(cm, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=cm.max() * 1.08)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "#1a1a1a"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=26, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted COLD", "Predicted NC"], fontsize=12)
    ax.set_yticklabels([f"True COLD\n({N_C_TRUE}; 9.4%)", f"True NC\n({N_NC_TRUE}; 90.6%)"],
                       fontsize=12)

    # per-class metric strip on the right
    ax.text(2.45, 0,  "Recall  65.0%\nPrec.   14.1%\nF1      0.232",
            fontsize=11, va="center", family="monospace", color="#222")
    ax.text(2.45, 1,  "Recall  59.1%\nPrec.   94.2%\nF1      0.726",
            fontsize=11, va="center", family="monospace", color="#222")

    ax.text(0.5, 1.10,
            "Predicted-C rate: 43.2%      vs      True prior: 9.4%",
            transform=ax.transAxes, ha="center", fontsize=15, fontweight="bold",
            color=C_RED,
            bbox=dict(boxstyle="round,pad=0.55", fc="#fff5e6", ec=C_RED, lw=1.5))

    # No matplotlib title -- callout above carries the visceral payload; metric
    # strip on the right carries the per-class numbers; slide title says
    # "we predict cold 4x too often". Keeping all three at the top would overlap.
    ax.set_xlim(-0.5, 1.9)
    plt.tight_layout()
    out = OUT_DIR / "fig1_confusion_matrix.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def fig2_logit_shift(logits: np.ndarray) -> float:
    """Histogram of test multi-K logits with tau_locked and tau_prior_match.

    tau_prior_match here is the quantile of the *test logit distribution* at
    1 - PUB_PRIOR. Computing it requires only test logits + the published prior
    -- no test labels touch this number.
    """
    tau_pm = float(np.quantile(logits, 1.0 - PUB_PRIOR))

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    bins = np.linspace(-30, 35, 90)
    ax.hist(logits, bins=bins, color=C_BLUE, alpha=0.78, edgecolor="white", linewidth=0.4)

    # leave headroom above the histogram for the annotation
    y_top_data = ax.get_ylim()[1]
    ax.set_ylim(0, y_top_data * 1.30)
    y_top = ax.get_ylim()[1]

    mean_logit = float(logits.mean())
    ax.axvline(mean_logit, color=C_GREY, linewidth=1.4, linestyle=":",
               label=f"mean test logit = {mean_logit:.2f}")
    ax.axvline(TAU_MULTIK, color=C_RED, linewidth=2.6, linestyle="--",
               label=f"tau_locked = {TAU_MULTIK:.3f}   -->  pred-C 43.2%")
    ax.axvline(tau_pm, color=C_GREEN, linewidth=2.6, linestyle="-",
               label=f"tau_prior_match = {tau_pm:.3f}   -->  pred-C 9.4%")

    # shade the over-prediction region (between the two thresholds)
    ax.axvspan(TAU_MULTIK, tau_pm, color=C_RED, alpha=0.10)
    mid = 0.5 * (TAU_MULTIK + tau_pm)
    # Plain text label inside the shaded zone -- no arrow (the shaded band is
    # already a visual link, and arrows fight the legend / bar tops).
    ax.text(mid, y_top_data * 0.55,
            "over-prediction\nzone\n(34 pp)",
            ha="center", va="center", fontsize=12, color=C_RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_RED, lw=1.2, alpha=0.95))

    ax.set_xlabel("multi-K ensemble logit on hidden test  (9551 utterances)")
    ax.set_ylabel("count")
    ax.set_title(f"Operating point sits {tau_pm - TAU_MULTIK:+.2f} logit units away from the prior-matched threshold",
                 pad=10)
    ax.legend(loc="upper right")
    ax.set_xlim(-30, 35)
    plt.tight_layout()
    out = OUT_DIR / "fig2_logit_shift.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return tau_pm


# -----------------------------------------------------------------------------
def fig3_scop_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(5.75, 5.15, "SCOP  --  Shadow-Calibrated Operating Point",
            fontsize=15, fontweight="bold", ha="center", color="#1a1a1a")

    # Locked pipeline (left)
    locked = mpatches.FancyBboxPatch((0.3, 1.6), 5.0, 2.3,
                                     boxstyle="round,pad=0.12",
                                     fc="#d6dbdf", ec="#2c3e50", linewidth=2)
    ax.add_patch(locked)
    ax.text(2.8, 3.62, "LOCKED  (untouched)", fontsize=11.5, fontweight="bold",
            ha="center", color="#2c3e50")
    ax.text(2.8, 3.18, "frozen WavLM-Large", fontsize=10.5, ha="center")
    ax.text(2.8, 2.82, "+ A2.5 audit-derived head x 5 seeds", fontsize=10.5, ha="center")
    ax.text(2.8, 2.46, "+ K1/K2 late fusion (betas locked)", fontsize=10.5, ha="center")
    ax.text(2.8, 2.10, "+ 5-seed mean-logit ensemble", fontsize=10.5, ha="center")
    ax.text(2.8, 1.74, "= byte-identical to submission_1", fontsize=10.2, ha="center",
            style="italic", color="#34495e")

    # SCOP box (right)
    scop = mpatches.FancyBboxPatch((6.5, 1.6), 4.4, 2.3,
                                   boxstyle="round,pad=0.12",
                                   fc="#a9dfbf", ec=C_GREEN, linewidth=2)
    ax.add_patch(scop)
    ax.text(8.7, 3.62, "NEW  --  SCOP", fontsize=11.5, fontweight="bold",
            ha="center", color=C_GREEN)
    ax.text(8.7, 3.18, "1. Bayes prior-correction to 9.4%", fontsize=10.5, ha="center")
    ax.text(8.7, 2.85, "(published challenge constant)", fontsize=9.5, ha="center",
            style="italic", color="#222")
    ax.text(8.7, 2.46, "2. Calibrator fit on 10 SHADOW folds", fontsize=10.5, ha="center")
    ax.text(8.7, 2.13, "(speaker-disjoint; LOPO selection)", fontsize=9.5, ha="center",
            style="italic", color="#222")
    ax.text(8.7, 1.76, "never sees test labels", fontsize=10.2, ha="center",
            style="italic", color=C_GREEN, fontweight="bold")

    # Arrows
    ax.annotate("", xy=(6.45, 2.75), xytext=(5.35, 2.75),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#2c3e50"))
    ax.text(5.9, 2.95, "multi-K\nlogits", fontsize=10, ha="center")

    ax.annotate("", xy=(11.4, 2.75), xytext=(10.95, 2.75),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=C_GREEN))
    ax.text(11.42, 2.75, "C / NC", fontsize=12, fontweight="bold",
            va="center", color=C_GREEN)

    # Footer / acceptance rule
    ax.text(5.75, 0.85,
            "Pre-registered acceptance (shadow units only):  "
            "mean shadow UAR lift >= +0.015,   no fold loses > 0.005",
            fontsize=10.5, ha="center", color=C_GREEN, style="italic")
    ax.text(5.75, 0.35,
            "Frozen backbone, locked betas, paper headline byte-identical",
            fontsize=10.5, ha="center", color="#2c3e50")

    plt.tight_layout()
    out = OUT_DIR / "fig3_scop_pipeline.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def fig4_failure_modes() -> None:
    modes = [
        ("Calibration / scale drift",     0.42, C_RED,   "shadow-testable"),
        ("Speaker confound",              0.24, C_AMBER, "needs new control"),
        ("Feature-distribution shift",    0.22, "#f39c12", "shadow-testable"),
        ("Single-split tau variance",     0.07, C_GREY,  "shadow-testable"),
        ("Ensemble degradation",          0.05, "#bdc3c7", "passive amplifier"),
    ]
    labels = [m[0] for m in modes]
    values = [m[1] for m in modes]
    colors = [m[2] for m in modes]
    tags   = [m[3] for m in modes]

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=1.6)

    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"  {v:.2f}", va="center", fontsize=12, fontweight="bold")
        tag_color = C_GREEN if "shadow-testable" in tags[i] else "#922b21" if "control" in tags[i] else "#555"
        ax.text(0.47, i, tags[i], va="center", fontsize=10, color=tag_color, style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("posterior  P(dominant cause | evidence)", fontsize=11.5)
    ax.set_xlim(0, 0.56)
    # No matplotlib title -- the slide title duplicates it verbatim
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    out = OUT_DIR / "fig4_failure_modes.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def fig5_pred_c_rate(logits: np.ndarray, tau_pm: float) -> None:
    tau_grid = np.linspace(-15, 15, 601)
    rate = np.array([(logits >= t).mean() for t in tau_grid])

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.plot(tau_grid, rate, color=C_BLUE, linewidth=2.6, label="pred-C rate on test")
    ax.axhline(PUB_PRIOR, color=C_GREEN, linestyle="--", linewidth=1.5, alpha=0.9,
               label=f"published cold prior = {PUB_PRIOR:.3f}")
    ax.axvline(TAU_MULTIK, color=C_RED, linestyle="--", linewidth=2)
    ax.axvline(tau_pm, color=C_GREEN, linestyle="-", linewidth=2)

    rate_at_locked = float((logits >= TAU_MULTIK).mean())
    ax.scatter([TAU_MULTIK], [rate_at_locked], s=110, color=C_RED, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.scatter([tau_pm], [PUB_PRIOR], s=110, color=C_GREEN, zorder=5,
               edgecolor="white", linewidth=1.5)

    ax.annotate(f"tau_locked = {TAU_MULTIK:.3f}\npred-C = {rate_at_locked*100:.1f}%",
                xy=(TAU_MULTIK, rate_at_locked), xytext=(-13.5, 0.70),
                fontsize=11.5, color=C_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.6))
    ax.annotate(f"tau_prior_match = {tau_pm:.3f}\npred-C = {PUB_PRIOR*100:.1f}%",
                xy=(tau_pm, PUB_PRIOR), xytext=(9.0, 0.42),
                fontsize=11.5, color=C_GREEN, fontweight="bold", ha="right",
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.6))

    delta = tau_pm - TAU_MULTIK
    ax.text(0.5, 0.94,
            f"To match the published 9.4% prior, tau would need to shift by {delta:+.2f} logit units",
            transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold",
            color=C_GREEN,
            bbox=dict(boxstyle="round,pad=0.45", fc="#eafaf1", ec=C_GREEN, lw=1.2))

    ax.set_xlabel("threshold tau")
    ax.set_ylabel("predicted-cold rate on test")
    ax.set_title("Prior-match diagnostic   (no test labels used)", pad=10)
    ax.set_xlim(-15, 15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right")
    plt.tight_layout()
    out = OUT_DIR / "fig5_pred_c_rate_vs_tau.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def _pipeline_box(ax, x, y, w, h, title, sub, fc, ec, title_color="#1a1a1a",
                  sub_color="#555", lw=2.0):
    """Helper -- consistent rounded box with two-line content."""
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=lw)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h * 0.66, title, fontsize=12.5, fontweight="bold",
            ha="center", va="center", color=title_color)
    ax.text(x + w / 2, y + h * 0.27, sub, fontsize=9.5, ha="center", va="center",
            color=sub_color, style="italic")


def _arrow(ax, x0, x1, y, color="#1a1a1a", lw=2.4):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color))


# -----------------------------------------------------------------------------
def fig6_current_architecture() -> None:
    """Locked multi-K pipeline (submission_1) with the FAILURE MODE annotated."""
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(6.75, 6.05, "Locked multi-K pipeline  (submission_1)",
            fontsize=17, fontweight="bold", ha="center", color="#1a1a1a")
    ax.text(6.75, 5.6, "frozen WavLM-Large + audit-derived A2.5 head + handcrafted-feature late fusion + 5-seed ensemble",
            fontsize=11, ha="center", color="#555", style="italic")

    boxes = [
        ("WavLM-Large\n(FROZEN)",      "25 transformer\nlayers",              "#d6dbdf", "#34495e"),
        ("Layer-wise pool\n+ audit prior",     "mean+std,\nstat_dim 4096",            "#d6eaf8", "#2874a6"),
        ("A2.5 head\n(x 5 seeds)",     "proj 128,\ndropout 0.5",              "#d4efdf", "#1e8449"),
        ("K1 / K2 fusion\nG4 + G5",     "betas locked\nper seed",              "#fcf3cf", "#9a7d0a"),
        ("5-seed\nensemble",            "mean of\nmulti-K logits",             "#e8daef", "#6c3483"),
        ("tau = -1.625\nthreshold",     "tuned on 1\ntrain holdout",           "#fadbd8", C_RED),
    ]
    n = len(boxes)
    box_w, box_h = 1.85, 1.45
    box_y = 3.05
    gap = 0.30
    total_w = n * box_w + (n - 1) * gap
    start_x = (13.5 - total_w) / 2

    for i, (title, sub, fc, ec) in enumerate(boxes):
        x = start_x + i * (box_w + gap)
        is_last = i == n - 1
        title_c = C_RED if is_last else "#1a1a1a"
        sub_c   = "#922b21" if is_last else "#555"
        _pipeline_box(ax, x, box_y, box_w, box_h, title, sub, fc, ec,
                      title_color=title_c, sub_color=sub_c)
        if i < n - 1:
            arrow_x0 = x + box_w + 0.02
            arrow_x1 = x + box_w + gap - 0.02
            _arrow(ax, arrow_x0, arrow_x1, box_y + box_h / 2)

    # arrow down + C/NC box below the threshold
    last_x = start_x + (n - 1) * (box_w + gap)
    cx = last_x + box_w / 2
    ax.annotate("", xy=(cx, 2.0), xytext=(cx, box_y - 0.05),
                arrowprops=dict(arrowstyle="->", lw=2.4, color=C_RED))
    cnc = mpatches.FancyBboxPatch((last_x + 0.30, 1.30), 1.25, 0.70,
                                   boxstyle="round,pad=0.04", fc="#fadbd8", ec=C_RED, lw=2)
    ax.add_patch(cnc)
    ax.text(last_x + 0.30 + 0.625, 1.65, "C / NC", fontsize=13,
            fontweight="bold", ha="center", va="center", color=C_RED)

    # FAILURE MODE banner at the bottom -- this is the rhetorical payload of the slide
    ax.axhline(0.94, xmin=0.06, xmax=0.94, color=C_RED, lw=0.8, alpha=0.4)
    ax.text(6.75, 0.62,
            "FAILURE MODE  --  tau tuned on ONE 10% train holdout (SPLIT_SEED=42)",
            fontsize=13.5, fontweight="bold", ha="center", color=C_RED)
    ax.text(6.75, 0.20,
            "On hidden test, logits shifted: 43.2% predicted COLD vs 9.4% true prior  "
            "-- operating-point gap = +8.72 logit units",
            fontsize=11, ha="center", color="#922b21", style="italic")

    plt.tight_layout()
    out = OUT_DIR / "fig6_current_architecture.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def fig7_scop_architecture() -> None:
    """Proposed SCOP-adapted pipeline for the end-of-semester second submission."""
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(6.75, 6.05,
            "Proposed SCOP-adapted pipeline  (submission_2)",
            fontsize=17, fontweight="bold", ha="center", color="#1a1a1a")
    ax.text(6.75, 5.6,
            "same locked backbone -- only the operating-point block is replaced",
            fontsize=11, ha="center", color="#555", style="italic")

    # Same first 5 boxes (locked), then the SCOP block replaces the threshold
    locked_boxes = [
        ("WavLM-Large\n(FROZEN)",      "25 transformer\nlayers",         "#d6dbdf", "#34495e"),
        ("Layer-wise pool\n+ audit prior",     "mean+std,\nstat_dim 4096",       "#d6eaf8", "#2874a6"),
        ("A2.5 head\n(x 5 seeds)",     "proj 128,\ndropout 0.5",         "#d4efdf", "#1e8449"),
        ("K1 / K2 fusion\nG4 + G5",     "betas locked\nper seed",         "#fcf3cf", "#9a7d0a"),
        ("5-seed\nensemble",            "mean of\nmulti-K logits",        "#e8daef", "#6c3483"),
    ]
    box_w, box_h = 1.85, 1.45
    box_y = 3.05
    gap = 0.30
    n_locked = len(locked_boxes)
    scop_w = 2.5  # the SCOP box is wider to show its two sub-steps
    total_w = n_locked * box_w + (n_locked - 1) * gap + gap + scop_w
    start_x = (13.5 - total_w) / 2

    for i, (title, sub, fc, ec) in enumerate(locked_boxes):
        x = start_x + i * (box_w + gap)
        _pipeline_box(ax, x, box_y, box_w, box_h, title, sub, fc, ec)
        _arrow(ax, x + box_w + 0.02, x + box_w + gap - 0.02, box_y + box_h / 2)

    # SCOP box (the NEW part)
    scop_x = start_x + n_locked * (box_w + gap)
    scop_rect = mpatches.FancyBboxPatch((scop_x, box_y - 0.20), scop_w, box_h + 0.40,
                                         boxstyle="round,pad=0.08",
                                         fc="#a9dfbf", ec=C_GREEN, lw=2.6)
    ax.add_patch(scop_rect)
    ax.text(scop_x + scop_w / 2, box_y + box_h + 0.05, "NEW  --  SCOP",
            fontsize=12.5, fontweight="bold", ha="center", color=C_GREEN)
    ax.text(scop_x + scop_w / 2, box_y + box_h * 0.62,
            "1. Bayes prior-correction", fontsize=10.5, ha="center", color="#196f3d")
    ax.text(scop_x + scop_w / 2, box_y + box_h * 0.40,
            "    to published 9.4% prior", fontsize=9, ha="center",
            style="italic", color="#196f3d")
    ax.text(scop_x + scop_w / 2, box_y + box_h * 0.20,
            "2. Shadow calibrator (LOPO)", fontsize=10.5, ha="center", color="#196f3d")
    ax.text(scop_x + scop_w / 2, box_y + box_h * 0.00,
            "    10 speaker-disjoint folds", fontsize=9, ha="center",
            style="italic", color="#196f3d")

    # Section labels above the boxes (no overlapping braces)
    locked_x0 = start_x - 0.04
    locked_x1 = start_x + n_locked * box_w + (n_locked - 1) * gap + 0.04
    locked_mid = (locked_x0 + locked_x1) / 2
    label_y = box_y + box_h + 0.35      # comfortably above the box tops (4.50 -> 4.85)
    # Tinted background strips so the sections read at a glance from a distance
    ax.add_patch(mpatches.Rectangle((locked_x0, label_y - 0.05),
                                    locked_x1 - locked_x0, 0.40,
                                    fc="#ecf0f1", ec="#34495e", lw=1.1, alpha=0.85))
    ax.text(locked_mid, label_y + 0.15,
            "LOCKED  --  byte-identical to submission_1",
            fontsize=11.5, fontweight="bold", ha="center", va="center",
            color="#34495e")

    ax.add_patch(mpatches.Rectangle((scop_x - 0.02, label_y - 0.05),
                                    scop_w + 0.04, 0.40,
                                    fc="#eafaf1", ec=C_GREEN, lw=1.1, alpha=0.95))
    ax.text(scop_x + scop_w / 2, label_y + 0.15,
            "REPLACED",
            fontsize=11.5, fontweight="bold", ha="center", va="center",
            color=C_GREEN)

    # arrow down + C/NC box below the SCOP block
    cx = scop_x + scop_w / 2
    ax.annotate("", xy=(cx, 2.0), xytext=(cx, box_y - 0.25),
                arrowprops=dict(arrowstyle="->", lw=2.4, color=C_GREEN))
    cnc = mpatches.FancyBboxPatch((scop_x + scop_w / 2 - 0.625, 1.30),
                                   1.25, 0.70, boxstyle="round,pad=0.04",
                                   fc="#a9dfbf", ec=C_GREEN, lw=2)
    ax.add_patch(cnc)
    ax.text(scop_x + scop_w / 2, 1.65, "C / NC", fontsize=13,
            fontweight="bold", ha="center", va="center", color=C_GREEN)

    # Footer
    ax.text(6.75, 0.62,
            "PRE-REGISTERED ACCEPTANCE (shadow units only):  "
            "mean shadow UAR lift >= +0.015,   no fold loses > 0.005",
            fontsize=12, fontweight="bold", ha="center", color=C_GREEN)
    ax.text(6.75, 0.20,
            "Frozen backbone  |  Locked betas  |  Paper headline byte-identical  |  Hidden-test labels never touched",
            fontsize=11, ha="center", color="#196f3d", style="italic")

    plt.tight_layout()
    out = OUT_DIR / "fig7_scop_architecture.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def main() -> None:
    print(f"[setup] output dir: {OUT_DIR}")
    logits = _load_test_logits()
    print(f"[setup] loaded {logits.size} test logits  "
          f"(min {logits.min():.2f}, mean {logits.mean():.2f}, max {logits.max():.2f})")

    fig1_confusion_matrix()
    tau_pm = fig2_logit_shift(logits)
    fig3_scop_schematic()
    fig4_failure_modes()
    fig5_pred_c_rate(logits, tau_pm)
    fig6_current_architecture()
    fig7_scop_architecture()

    print(f"\n[done] tau_prior_match = {tau_pm:+.4f}  "
          f"(vs tau_locked = {TAU_MULTIK:+.4f};  delta = {tau_pm - TAU_MULTIK:+.4f})")
    print(f"[done] 7 PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
