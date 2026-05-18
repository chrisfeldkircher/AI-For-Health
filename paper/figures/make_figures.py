"""Deterministic paper figures from paper_data/ + results/ only.

No model training, no cache reads. Reads:
  paper_data/eda/umap_coords_ecapa_2d.csv
  paper_data/eda/pseudo_speaker_cluster_stats.csv
  paper_data/cumulative_stack.csv
  paper_data/shadow_summary.csv
  paper_data/shadow_distributions_long.csv
  paper_data/layer_audit_wavlm_grouped.csv
  results/pseudo_speaker_ecapa_diagnostics.json

Writes paper/figures/fig{1,2,3,4} as PDF (for the LaTeX includegraphics)
plus PNG (for visual inspection at print size). Datascience env.

Every plotted scalar is echoed to stdout for cross-check against source.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
PD = ROOT / "paper_data"
EDA = PD / "eda"
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4,
})
COL = 3.45   # IEEE single-column width (in)
DBL = 7.16   # IEEE double-column width (in)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf + .png")


# ---------------------------------------------------------------------------
# Fig 1 — ECAPA pseudo-speaker validation (double-column, 2 panels)
# ---------------------------------------------------------------------------
def fig1():
    print("[fig1] ECAPA pseudo-speaker validation")
    u = pd.read_csv(EDA / "umap_coords_ecapa_2d.csv")
    cs = pd.read_csv(EDA / "pseudo_speaker_cluster_stats.csv")
    diag = json.loads((RES / "pseudo_speaker_ecapa_diagnostics.json").read_text())
    knn = diag["knn_label_cohesion"]; hdb = diag["hdbscan"]
    agg = diag["agglomerative_k_matched"]
    sil = diag["silhouette_shipped_labels_RELATIVE"]

    fig = plt.figure(figsize=(DBL, 2.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.95])
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2])

    # (A) ECAPA manifold density (19,101 chunks) -- structure exists, robust to
    #     the ~9.5% cold skew that made a cold-rate scatter one purple blob.
    hb = axA.hexbin(u.umap_x, u.umap_y, gridsize=46, bins="log",
                    cmap="viridis", linewidths=0)
    axA.set_title("(A) ECAPA UMAP manifold\n(19,101 chunks, log density)")
    axA.set_xlabel("UMAP-1"); axA.set_ylabel("UMAP-2")
    axA.set_xticks([]); axA.set_yticks([]); axA.grid(False)
    cb = fig.colorbar(hb, ax=axA, fraction=0.046, pad=0.02)
    cb.set_label("log$_{10}$ count", fontsize=7)

    # (B) per-cluster tightness vs separation (the honest cluster geometry)
    sz = (cs.n_chunks - cs.n_chunks.min()) / (cs.n_chunks.max() - cs.n_chunks.min())
    sb = axB.scatter(cs.intra_cluster_mean_cosine, cs.nearest_other_cluster_cosine,
                     c=cs.cold_rate, s=8 + 34 * sz, alpha=0.7, cmap="magma",
                     edgecolors="k", linewidths=0.2)
    lo = min(cs.intra_cluster_mean_cosine.min(), cs.nearest_other_cluster_cosine.min())
    hi = max(cs.intra_cluster_mean_cosine.max(), cs.nearest_other_cluster_cosine.max())
    axB.plot([lo, hi], [lo, hi], "k--", lw=0.6, label="intra = nearest-other")
    axB.set_xlabel("intra-cluster mean cosine")
    axB.set_ylabel("nearest-other-cluster cosine")
    axB.set_title("(B) Per-cluster tightness vs\nseparation (k=210, marker$\\propto$size)")
    axB.legend(loc="lower right", framealpha=0.9, fontsize=6)
    cbB = fig.colorbar(sb, ax=axB, fraction=0.046, pad=0.02)
    cbB.set_label("cluster cold rate", fontsize=7)

    # (C) raw-L2 validation numbers on their own clean axes (no data overlap)
    axC.axis("off")
    axC.set_title("(C) raw-L2 192-D validation")
    lines = [
        ("Validated: SHIPPED k=210 labels", "0.0", True),
        ("(production cluster.py space;", "0.0", False),
        ("NOT the optimistic UMAP-32)", "0.0", False),
        ("", "", False),
        (f"kNN label cohesion (k=10)", "0.0", True),
        (f"  {knn['mean_same_pseudo_speaker']:.3f}  vs chance {knn['chance_baseline_sum_p2']:.4f}", "0.0", False),
        (f"  = {knn['lift_over_chance']:.0f}x  (load-bearing)", "0.0", False),
        ("", "", False),
        (f"HDBSCAN (independent)", "0.0", True),
        (f"  {hdb['n_clusters']} clusters, noise {hdb['noise_frac']*100:.1f}%", "0.0", False),
        (f"  non-degenerate; NMI {hdb['nmi_vs_shipped']:.2f}", "0.0", False),
        (f"  (ARI {hdb['ari_vs_shipped']:.2f}: 406>210 granularity)", "0.0", False),
        ("", "", False),
        (f"Agglomerative@210  NMI {agg['nmi_vs_shipped']:.2f}", "0.0", False),
        (f"Silhouette {sil['value']:.3f}  (relative only)", "0.0", False),
    ]
    y = 0.97
    for s, _, bold in lines:
        axC.text(0.0, y, s, transform=axC.transAxes, fontsize=6.0,
                 va="top", ha="left", family="monospace",
                 fontweight="bold" if bold else "normal")
        y -= 0.063
    _save(fig, "fig1")
    print(f"  CHECK kNN={knn['mean_same_pseudo_speaker']:.4f} chance={knn['chance_baseline_sum_p2']:.4f}"
          f" lift={knn['lift_over_chance']:.1f} | HDBSCAN n={hdb['n_clusters']} noise={hdb['noise_frac']:.4f}"
          f" NMI={hdb['nmi_vs_shipped']:.4f} ARI={hdb['ari_vs_shipped']:.4f} | sil={sil['value']:.4f}"
          f" | branch={diag['narrative_branch']}")


# ---------------------------------------------------------------------------
# Fig 2 — cumulative audited stack (single-column)
# ---------------------------------------------------------------------------
def fig2():
    print("[fig2] cumulative stack")
    cum = pd.read_csv(PD / "cumulative_stack.csv").set_index("stage")
    sh = pd.read_csv(PD / "shadow_summary.csv").set_index("method")

    ladder = ["A2_grouped", "A2.5_honestprior", "A5b_K1_n3", "A5b_K2_n5"]
    labels = ["A2", "A2.5", "K=1", "K=2"]
    xs = np.arange(len(ladder))
    ys = [cum.loc[r, "devel_test_uar"] for r in ladder]
    es = [cum.loc[r, "devel_test_uar_std"] for r in ladder]

    fig, ax = plt.subplots(figsize=(COL, 2.7))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#1f4e79", capsize=2.5,
                ms=5, lw=1.3, label="controlled ladder (5-seed mean $\\pm\\sigma$)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.2)

    # ensemble / multi-K fork: canonical hollow + shadow filled
    xe = len(ladder)
    canon_ml = cum.loc["A5b_K2_mean_logit_ens", "devel_test_uar"]   # 0.709
    canon_mk = cum.loc["A5b_multiK", "devel_test_uar"]              # 0.7111
    sh_ml_m, sh_ml_s = sh.loc["K2_only", "shadow_mean"], sh.loc["K2_only", "shadow_std"]
    sh_mk_m, sh_mk_s = sh.loc["multi_K", "shadow_mean"], sh.loc["multi_K", "shadow_std"]

    ax.scatter([xe, xe + 1], [canon_ml, canon_mk], facecolors="none",
               edgecolors="#c00000", s=55, lw=1.4, zorder=5,
               label="canonical split (single partition)")
    ax.errorbar([xe, xe + 1], [sh_ml_m, sh_mk_m], yerr=[sh_ml_s, sh_mk_s],
                fmt="s", color="#c00000", capsize=2.5, ms=5, lw=1.2,
                label="shadow mean $\\pm\\sigma$ (10 partitions)")
    for x, yc, ys_, lab in [(xe, canon_ml, sh_ml_m, "K=2 ens"),
                            (xe + 1, canon_mk, sh_mk_m, "multi-K")]:
        ax.annotate(f"{yc:.3f}", (x, yc), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.2, color="#c00000")
        ax.annotate(f"{ys_:.3f}", (x, ys_), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=6.2, color="#c00000")

    ax.axhline(0.710, ls="--", color="0.4", lw=1.0)
    ax.text(0.15, 0.7105, "2017 hidden-test baseline 0.710", fontsize=6.0,
            color="0.3", va="bottom")
    ax.set_xticks(list(xs) + [xe, xe + 1])
    ax.set_xticklabels(labels + ["K=2 ens", "multi-K"], fontsize=7)
    ax.set_ylabel("devel_test UAR")
    ax.set_title("Cumulative audited stack (shadow-first)")
    ax.set_ylim(0.62, 0.725)
    ax.legend(loc="lower right", fontsize=5.6, framealpha=0.92)
    _save(fig, "fig2")
    print(f"  CHECK A2={ys[0]:.4f} A2.5={ys[1]:.4f} K1={ys[2]:.4f} K2n5={ys[3]:.4f}"
          f" | K2ens canon={canon_ml:.4f} shadow={sh_ml_m:.4f}±{sh_ml_s:.4f}"
          f" | multiK canon={canon_mk:.4f} shadow={sh_mk_m:.4f}±{sh_mk_s:.4f}")


# ---------------------------------------------------------------------------
# Fig 3 — honesty-prior layer weighting (single-column, stacked small multiples)
# ---------------------------------------------------------------------------
def fig3():
    print("[fig3] honesty-prior layer small-multiples")
    la = pd.read_csv(PD / "layer_audit_wavlm_grouped.csv").sort_values("layer")
    L = la.layer.to_numpy()
    sub1 = la.subtractive_honesty_lam1.to_numpy()
    T_INV = 50.0
    z = T_INV * sub1
    prior = np.exp(z - z.max()) / np.exp(z - z.max()).sum()

    fig, ax = plt.subplots(4, 1, figsize=(COL, 5.0), sharex=True)
    ax[0].plot(L, la.cold_uar, "o-", color="#1f4e79", ms=3, lw=1.1)
    ax[0].set_ylabel("cold UAR"); ax[0].set_title("Per-layer audit $\\rightarrow$ honesty prior")
    ax[1].plot(L, la.speaker_top1, "o-", color="#c00000", ms=3, lw=1.1)
    ax[1].set_ylabel("speaker top-1")
    ax[2].axhline(0, color="0.6", lw=0.6)
    ax[2].plot(L, sub1, "o-", color="#2e7d32", ms=3, lw=1.1)
    ax[2].set_ylabel("sub@1\n(cold$-$spk gain)")
    ax[3].bar(L, prior, color="#7b3f9e", width=0.7)
    ax[3].set_ylabel("prior weight\nsoftmax(50$\\cdot$sub@1)")
    ax[3].set_xlabel("WavLM-Large layer index")
    ax[3].set_xticks(L[::2])
    for a in ax:
        a.grid(alpha=0.25, lw=0.4)
    _save(fig, "fig3")
    print(f"  CHECK argmax sub@1 = L{int(L[sub1.argmax()])} (sub@1={sub1.max():.4f});"
          f" argmax prior = L{int(L[prior.argmax()])} (w={prior.max():.4f});"
          f" prior max/min = {prior.max()/prior.min():.2f}x")


# ---------------------------------------------------------------------------
# Fig 4 — shadow robustness (double-column)
# ---------------------------------------------------------------------------
def fig4():
    print("[fig4] shadow robustness")
    ld = pd.read_csv(PD / "shadow_distributions_long.csv")
    sh = pd.read_csv(PD / "shadow_summary.csv").set_index("method")

    methods = ["K2_only", "multi_K"]
    nice = {"K2_only": "K=2 ensemble", "multi_K": "multi-K"}
    fig, ax = plt.subplots(figsize=(DBL, 2.9))
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        sub = ld[ld.method == m]
        shadow = sub[sub.partition == "shadow"]
        canon = sub[sub.partition == "canonical"]
        xj = i + (rng.random(len(shadow)) - 0.5) * 0.22
        ax.scatter(xj, shadow.devel_test_uar, s=22, alpha=0.7, color="#1f4e79",
                   label="shadow split" if i == 0 else None, zorder=3)
        sm, ss = sh.loc[m, "shadow_mean"], sh.loc[m, "shadow_std"]
        ax.errorbar(i, sm, yerr=ss, fmt="s", color="#c00000", ms=7, capsize=4,
                    lw=1.6, zorder=4,
                    label="shadow mean $\\pm\\sigma$" if i == 0 else None)
        ax.scatter([i], canon.devel_test_uar, marker="*", s=170,
                   facecolors="none", edgecolors="#c00000", lw=1.6, zorder=5,
                   label="canonical split" if i == 0 else None)
        # seed-23 annotation (low outlier)
        s23 = shadow[shadow.split_seed == 23]
        if len(s23):
            y23 = float(s23.devel_test_uar.iloc[0])
            ax.annotate(f"seed 23\n({y23:.3f})", (i, y23),
                        textcoords="offset points", xytext=(20, -2),
                        fontsize=5.8, color="0.3",
                        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))

    ax.axhline(0.710, ls="--", color="0.4", lw=1.0)
    ax.text(-0.45, 0.7108, "2017 hidden-test baseline 0.710", fontsize=6.2,
            color="0.3", va="bottom")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([nice[m] for m in methods])
    ax.set_ylabel("devel_test UAR")
    ax.set_title("Shadow-split robustness: canonical is one partition in a distribution")
    ax.set_xlim(-0.5, len(methods) - 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              fontsize=6.4, framealpha=0.92)
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "fig4")
    for m in methods:
        print(f"  CHECK {m}: canon={sh.loc[m,'canonical_uar']:.4f}"
              f" shadow={sh.loc[m,'shadow_mean']:.4f}±{sh.loc[m,'shadow_std']:.4f}")


if __name__ == "__main__":
    print(f"ROOT={ROOT}")
    fig1(); fig2(); fig3(); fig4()
    print("\n[done] 4 figures -> paper/figures/*.{pdf,png}")
