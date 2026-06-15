"""Generate the figures for the ML4Health mid-term talk (corrected diagnosis).

Run from the project root with the datascience env:
  & "$env:USERPROFILE\AppData\Local\miniconda3\envs\datascience\python.exe" presentation/make_slide_figures.py

Writes 300-DPI PNGs into presentation/figures/. Plain styling: no slogans, no
dashes used as punctuation.

Figures:
  fig1_confusion_matrix.png    test confusion matrix, with predicted-cold vs true-prior callout
  fig6_architecture.png        what we used (the fusion pipeline)
  fig8_recall_below_shadow.png the result that matters: same cold-call rate, lower recall on test
  fig9_what_we_change.png       what will not work (measured) and the two paths forward
  fig10_pool_shift.png          why dev folds did not see it (same pool) vs the new test pool

All numbers come from results/ on disk: the back-solved test confusion matrix
(TP=582, FP=3542, FN=313, TN=5114) and the 11 dev/shadow folds in
results/A5b_k2_multi_k_ensemble.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MULTIK_JSON = ROOT / "results" / "A5b_k2_multi_k_ensemble.json"

# back-solved test confusion matrix (reproduces all four leaderboard metrics)
TP, FP, FN, TN = 582, 3542, 313, 5114
N_TEST = TP + FP + FN + TN          # 9551
N_C_TRUE = TP + FN                  # 895
N_NC_TRUE = FP + TN                 # 8656
TEST_RECALL_C = TP / N_C_TRUE       # 0.6503
TEST_RECALL_NC = TN / N_NC_TRUE     # 0.5908

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

C_RED = "#c0392b"
C_GREEN = "#196f3d"
C_BLUE = "#2980b9"
C_PURPLE = "#7d3c98"
C_GREY = "#7f8c8d"
TUM_BLUE = "#0065bd"


# -----------------------------------------------------------------------------
def fig1_confusion_matrix() -> None:
    cm = np.array([[TP, FN], [FP, TN]], dtype=int)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.imshow(cm, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=cm.max() * 1.08)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "#1a1a1a"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=26, fontweight="bold", color=color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted cold", "Predicted not-cold"], fontsize=12)
    ax.set_yticklabels([f"True cold\n({N_C_TRUE}, 9.4%)",
                        f"True not-cold\n({N_NC_TRUE}, 90.6%)"], fontsize=12)
    ax.text(2.45, 0, "recall  65.0%\nprec.   14.1%", fontsize=11, va="center",
            family="monospace", color="#222")
    ax.text(2.45, 1, "recall  59.1%\nprec.   94.2%", fontsize=11, va="center",
            family="monospace", color="#222")
    ax.text(0.5, 1.16, "We called cold on 43% of clips. The real rate is 9%.",
            transform=ax.transAxes, ha="center", fontsize=14.5, fontweight="bold",
            color=C_RED,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff5e6", ec=C_RED, lw=1.5))
    ax.set_xlim(-0.5, 1.9)
    plt.tight_layout()
    out = OUT_DIR / "fig1_confusion_matrix.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def _box(ax, x, y, w, h, title, sub, fc, ec):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=2.0)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h * 0.64, title, fontsize=12, fontweight="bold",
            ha="center", va="center", color="#1a1a1a")
    ax.text(x + w / 2, y + h * 0.26, sub, fontsize=9.5, ha="center", va="center",
            color="#555", style="italic")


def fig6_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 3.9))
    ax.set_xlim(0, 13.0); ax.set_ylim(0, 3.9); ax.axis("off")

    boxes = [
        ("WavLM-Large", "frozen, 25 layers", "#d6dbdf", "#34495e"),
        ("Pooling head", "layer weights from\ncold vs speaker audit", "#d6eaf8", "#2874a6"),
        ("Classifier", "5 seeds", "#d4efdf", "#1e8449"),
        ("Late fusion", "+ G4 gain-invariant\n+ G5 modulation", "#fcf3cf", "#9a7d0a"),
        ("Average", "mean over\n5 seeds", "#e8daef", "#6c3483"),
        ("Threshold", "set for best\ndev UAR", "#fdebd0", "#ca6f1e"),
    ]
    n = len(boxes); bw, bh, gap = 1.85, 1.55, 0.28
    total = n * bw + (n - 1) * gap
    x0 = (13.0 - total) / 2
    yb = 2.05
    for i, (t, s, fc, ec) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        _box(ax, x, yb, bw, bh, t, s, fc, ec)
        if i < n - 1:
            ax.annotate("", xy=(x + bw + gap - 0.02, yb + bh / 2),
                        xytext=(x + bw + 0.02, yb + bh / 2),
                        arrowprops=dict(arrowstyle="->", lw=2.2, color="#34495e"))
    lastx = x0 + (n - 1) * (bw + gap) + bw / 2
    ax.annotate("", xy=(lastx, 1.05), xytext=(lastx, yb - 0.05),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#34495e"))
    ax.text(lastx, 0.8, "cold / not-cold", fontsize=12, fontweight="bold",
            ha="center", color="#1a1a1a")
    ax.text(6.5, 0.25,
            "On our dev split this matched the 2017 ComParE baseline: UAR 0.711 vs 0.710.",
            fontsize=11.5, ha="center", color="#196f3d")
    plt.tight_layout()
    out = OUT_DIR / "fig6_architecture.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def fig8_recall_below_shadow() -> None:
    mk = json.loads(MULTIK_JSON.read_text())
    folds = mk["multi_k_per_split"]
    recC = np.array([f["recall_C"] for f in folds])
    recNC = np.array([f["recall_NC"] for f in folds])

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    cols = {"C": C_BLUE, "NC": C_PURPLE}
    xpos = {"C": 0.0, "NC": 1.0}
    rng = np.random.RandomState(0)
    for key, vals, test_v in (("C", recC, TEST_RECALL_C), ("NC", recNC, TEST_RECALL_NC)):
        x = xpos[key]
        lo, hi = vals.min(), vals.max()
        ax.add_patch(mpatches.Rectangle((x - 0.22, lo), 0.44, hi - lo,
                                        fc=cols[key], ec="none", alpha=0.16))
        jx = x + (rng.rand(len(vals)) - 0.5) * 0.26
        ax.scatter(jx, vals, s=55, color=cols[key], alpha=0.85, edgecolor="white",
                   linewidth=0.6, zorder=4)
        ax.scatter([x], [test_v], s=240, marker="v", color=C_RED, edgecolor="white",
                   linewidth=1.5, zorder=6)
        # test label centered below its own marker (no horizontal collisions)
        ax.text(x, test_v - 0.022, f"test {test_v:.3f}", ha="center", va="top",
                fontsize=12, fontweight="bold", color=C_RED)
        # dev-fold range centered above its own band
        ax.text(x, hi + 0.010, f"dev folds {lo:.3f} to {hi:.3f}", ha="center",
                va="bottom", fontsize=10, color=cols[key])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["recall on cold", "recall on not-cold"], fontsize=13, fontweight="bold")
    ax.set_ylabel("per-class recall at the same 42% cold-call rate")
    ax.set_title("Same cold-call rate as on dev, lower recall on test", pad=12)
    ax.set_xlim(-0.7, 1.7); ax.set_ylim(0.55, 0.84)
    ax.text(0.5, 0.572,
            "At the same cold-call rate, lower recall means the model orders test clips worse.\n"
            "Changing the threshold or rescaling the scores cannot fix that.",
            transform=ax.transData, ha="center", fontsize=10.5, style="italic", color="#444")
    plt.tight_layout()
    out = OUT_DIR / "fig8_recall_below_shadow.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def fig9_what_we_change() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.9); ax.axis("off")

    # left: what we measured does not work
    left = mpatches.FancyBboxPatch((0.3, 1.0), 5.6, 3.7, boxstyle="round,pad=0.1",
                                   fc="#fdedec", ec=C_RED, lw=2)
    ax.add_patch(left)
    ax.text(3.1, 4.35, "Ruled out by measurement", fontsize=12.5, fontweight="bold",
            ha="center", color=C_RED)
    dead = [
        "Move the threshold",
        "Rescale or re-standardize the scores",
        "Test-time feature normalization (BN-adapt)",
    ]
    for i, d in enumerate(dead):
        y = 3.75 - i * 0.62
        ax.text(0.85, y, "x", fontsize=15, fontweight="bold", color=C_RED, va="center")
        ax.text(1.25, y, d, fontsize=11.5, va="center", color="#1a1a1a")
    ax.text(3.1, 1.45,
            "The scores already sit where they should.\nThese only rescale them, so they cannot help.",
            fontsize=10, ha="center", va="center", style="italic", color="#555")

    # right: two paths
    right = mpatches.FancyBboxPatch((6.6, 1.0), 5.6, 3.7, boxstyle="round,pad=0.1",
                                    fc="#eafaf1", ec=C_GREEN, lw=2)
    ax.add_patch(right)
    ax.text(9.4, 4.35, "Two paths that can", fontsize=12.5, fontweight="bold",
            ha="center", color=C_GREEN)
    ax.text(6.95, 3.75, "A.  Report it as the frozen-model limit", fontsize=11.5,
            fontweight="bold", va="center", color="#1a1a1a")
    ax.text(7.2, 3.32, "the speaker-aware protocol already flagged the gap",
            fontsize=9.8, va="center", style="italic", color="#555")
    ax.text(6.95, 2.6, "B.  Unfreeze the top WavLM layers", fontsize=11.5,
            fontweight="bold", va="center", color="#1a1a1a")
    ax.text(7.2, 2.17, "train with speaker-stratified batches so the cold",
            fontsize=9.8, va="center", style="italic", color="#555")
    ax.text(7.2, 1.87, "direction depends less on who is speaking",
            fontsize=9.8, va="center", style="italic", color="#555")
    ax.text(9.4, 1.35, "Pick on the dev folds before submitting again.",
            fontsize=10, ha="center", va="center", color="#196f3d")

    ax.text(6.25, 0.45,
            "Frozen-scope de-confounding (mixup, contrastive, gradient reversal) "
            "already failed in our M8 to M19 controls.",
            fontsize=10, ha="center", color="#444", style="italic")
    plt.tight_layout()
    out = OUT_DIR / "fig9_what_we_change.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def fig10_pool_shift() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.5))
    ax.set_xlim(0, 12.0); ax.set_ylim(0, 4.5); ax.axis("off")

    # Pool A
    poolA = mpatches.FancyBboxPatch((0.4, 1.4), 5.2, 2.9, boxstyle="round,pad=0.1",
                                    fc="#eaf2fb", ec=TUM_BLUE, lw=2)
    ax.add_patch(poolA)
    ax.text(3.0, 3.95, "Speaker pool A", fontsize=12.5, fontweight="bold",
            ha="center", color=TUM_BLUE)
    ax.text(3.0, 3.5, "training + dev + all 10 shadow folds", fontsize=10.5,
            ha="center", color="#1a1a1a")
    rng = np.random.RandomState(1)
    for _ in range(22):
        cx = 0.9 + rng.rand() * 4.2
        cy = 1.75 + rng.rand() * 1.4
        ax.add_patch(mpatches.Circle((cx, cy), 0.09, fc=TUM_BLUE, ec="none", alpha=0.55))
    ax.text(3.0, 1.5, "folds change which speakers are held out,\nbut all come from pool A",
            fontsize=9.3, ha="center", style="italic", color="#555")

    # Pool B
    poolB = mpatches.FancyBboxPatch((6.4, 1.4), 5.2, 2.9, boxstyle="round,pad=0.1",
                                    fc="#fdedec", ec=C_RED, lw=2)
    ax.add_patch(poolB)
    ax.text(9.0, 3.95, "Speaker pool B", fontsize=12.5, fontweight="bold",
            ha="center", color=C_RED)
    ax.text(9.0, 3.5, "the hidden test", fontsize=10.5, ha="center", color="#1a1a1a")
    for _ in range(22):
        cx = 6.9 + rng.rand() * 4.2
        cy = 1.75 + rng.rand() * 1.4
        ax.add_patch(mpatches.Circle((cx, cy), 0.09, fc=C_RED, ec="none", alpha=0.55))

    ax.annotate("", xy=(6.35, 2.85), xytext=(5.65, 2.85),
                arrowprops=dict(arrowstyle="->", lw=2.4, color="#34495e"))
    ax.text(6.0, 0.7,
            "The cold direction learned on pool A separates pool B worse. "
            "Folds drawn from pool A cannot measure that gap.",
            fontsize=11, ha="center", color="#1a1a1a")
    plt.tight_layout()
    out = OUT_DIR / "fig10_pool_shift.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
def main() -> None:
    print(f"[setup] output dir: {OUT_DIR}")
    fig1_confusion_matrix()
    fig6_architecture()
    fig8_recall_below_shadow()
    fig9_what_we_change()
    fig10_pool_shift()
    print("[done] 5 figures written")


if __name__ == "__main__":
    main()
