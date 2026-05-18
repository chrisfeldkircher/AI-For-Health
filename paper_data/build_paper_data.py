"""Consolidate paper-figure source data into a compact, plot-ready directory.

Reads from `results/*.json` and `results/*.csv` (the experiment outputs) and emits
small CSVs into `paper_data/` so the paper-figure plotting can be done on any
machine without copying the multi-GB feature caches.

Usage (from project root):
    python paper_data/build_paper_data.py

Outputs (under paper_data/):
    cumulative_stack.csv                — per-stage UAR + speaker probes
    shadow_distributions_long.csv       — long-form: method × shadow_seed × UAR + recC + recNC
    shadow_summary.csv                  — per-method shadow_mean, shadow_std, canonical, paired lift
    standalone_uar_predictor.csv        — per-candidate standalone UAR + K-fusion verdict
    methodology_table.csv               — M8–M19 row text
    speaker_smoothing_alpha_sweep.csv   — per-α canonical UAR + recall pattern
    beta_sweep_k2.csv                   — β-grid lift on K=2 (extended sweep)
    ablations_calibration_tta.csv       — calibration variants + TTA variants
    per_seed_locked.csv                 — per-seed locked β / τ / UAR (K=1, K=2, multi-K-K1, multi-K-K2)
    layer_audit_wavlm.csv               — A5d_layer_honesty (copy)
    layer_audit_wavlm_grouped.csv       — A5d_grouped_layer_honesty (copy)
    layer_audit_hubert.csv              — A5d_hubert_layer_honesty (copy)
    hubert_a25_layer_audit.csv          — per-layer audit reused by HuBERT-A2.5 head
    hubert_a25_per_seed_standalone.csv  — per-seed standalone UAR + final layer weights
    compare_lr_per_shadow.csv           — ComParE-2016+LR per-shadow standalone (M14 pre-flight floor)
    multi_k_per_split.csv               — per-split (canonical + 10 shadow) UAR for all multi-K variants
"""
from __future__ import annotations
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent  # project root
RESULTS = ROOT / "results"
OUT = ROOT / "paper_data"
OUT.mkdir(exist_ok=True)


def load(name: str) -> dict[str, Any]:
    p = RESULTS / name
    with open(p, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_csv(name: str, header: list[str], rows: list[list[Any]]) -> None:
    p = OUT / name
    with open(p, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {p.relative_to(ROOT)}  ({len(rows)} rows)")


def fnum(x: Any, n: int = 6) -> Any:
    if isinstance(x, float):
        return round(x, n)
    return x


# ---------------------------------------------------------------------------
# 1. CUMULATIVE STACK
# ---------------------------------------------------------------------------
def build_cumulative_stack() -> None:
    """Per-stage UAR + speaker probe values, mirroring paper Table I (cumulative stack)."""
    a2 = load("A2_grouped.json")
    a25 = load("A2_grouped_honestprior.json")
    a5b = load("A5b_grouped_honestprior_betasweep_extended.json")
    k2 = load("A5b_k2_5seed_lock.json")
    k2_ens = load("A5b_k2_5seed_ensemble.json")
    k2_probes = load("A5b_k2_5seed_speaker_probes.json")
    multi_k = load("A5b_k2_multi_k_ensemble.json")

    rows: list[list[Any]] = []

    # A2 grouped (leak-corrected baseline)
    a2_dist = a2["distribution"]["uar_argmax"]
    rows.append([
        "A2_grouped", "frozen WavLM-Large + 2-layer MLP, uniform layer-weight init",
        fnum(a2_dist["mean"]),
        fnum(a2_dist["std"]),
        a2_dist.get("n", 3),
        fnum(a2["speaker_probe_mlp"]["top1"]["mean"]),
        fnum(a2["speaker_probe_lr"]["top1"]["mean"]),
    ])
    # A2.5 honesty-prior
    # A2.5 at N=5 (aggregate_n5.a25_arg_uar) so the cumulative stack matches
    # the paper table tab:stack (5-seed), not the 3-seed A2_grouped_honestprior
    # distribution. Speaker probes stay the 3-seed audit values.
    n3 = k2["aggregate_n3"]
    n5 = k2["aggregate_n5"]
    a25_n5 = n5["a25_arg_uar"]
    rows.append([
        "A2.5_honestprior", "+ honesty-prior layer-weight init (T_INV * sub@1), N=5",
        fnum(a25_n5["mean"]),
        fnum(a25_n5["std"]),
        a25_n5.get("n", 5),
        fnum(a25["speaker_probe_mlp"]["top1"]["mean"]),
        fnum(a25["speaker_probe_lr"]["top1"]["mean"]),
    ])
    # K=1 at N=5 (locked β for A2.5 + G4_gain_invariant) -- matches tab:stack
    rows.append([
        "A5b_K1_n5", "+ K=1 late fusion w/ G4_gain_invariant (n=5 seeds)",
        fnum(n5["k1_locked_uar"]["mean"]),
        fnum(n5["k1_locked_uar"]["std"]),
        5,
        None, None,
    ])
    # K=2 (locked β for A2.5 + G4_gain_invariant + G5_modulation)
    rows.append([
        "A5b_K2_n3", "+ K=2 late fusion w/ G4_gain_invariant + G5_modulation (n=3 seeds)",
        fnum(n3["k2_locked_uar"]["mean"]),
        fnum(n3["k2_locked_uar"]["std"]),
        3,
        None, None,
    ])
    rows.append([
        "A5b_K2_n5", "K=2 5-seed expansion (canonical headline)",
        fnum(n5["k2_locked_uar"]["mean"]),
        fnum(n5["k2_locked_uar"]["std"]),
        5,
        fnum(k2_probes["aggregate"]["probe_i_literal"]["top1"]["mean"]),
        fnum(k2_probes["aggregate"]["probe_ii_bb_concat"]["top1"]["mean"]),
    ])
    # 5-seed mean-logit ensemble (K=2)
    rows.append([
        "A5b_K2_mean_logit_ens", "K=2 5-seed mean-logit ensemble (paper-supplementary)",
        fnum(k2_ens["ensemble_mean_logit"]["tau_devel_test"]["uar"]),
        None,
        5,
        None, None,
    ])
    # Multi-K ensemble (per-seed avg of K=1 + K=2)
    rows.append([
        "A5b_multiK", "Multi-K ensemble: per-seed avg(K=1, K=2) then 5-seed mean-logit (canonical 0.7111)",
        fnum(multi_k["multi_k_canonical_uar"]),
        None,
        5,
        None, None,
    ])

    write_csv(
        "cumulative_stack.csv",
        [
            "stage", "description",
            "devel_test_uar", "devel_test_uar_std", "n_seeds",
            "speaker_probe_mlp_top1", "speaker_probe_lr_top1",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# 2. SHADOW DISTRIBUTIONS (long + summary)
# ---------------------------------------------------------------------------
def build_shadow_distributions() -> None:
    """Long-form: method × split_seed × UAR + recall pattern, plus aggregate summary."""
    multi_k = load("A5b_k2_multi_k_ensemble.json")
    diverse = load("A5b_multi_k_10seed_diverse.json")
    compare = load("A5b_compare_svm_k3.json")
    smoothing = load("A5b_k2_speaker_smoothing.json")
    shadow_only = load("A5b_k2_shadow_splits.json")

    long_rows: list[list[Any]] = []

    def add_method(method: str, splits: list[dict[str, Any]]) -> None:
        for r in splits:
            long_rows.append([
                method,
                r.get("split_seed"),
                "canonical" if r.get("split_seed") == 42 else "shadow",
                r.get("n_test"),
                fnum(r.get("uar")),
                fnum(r.get("recall_C")),
                fnum(r.get("recall_NC")),
            ])

    # K=2-only baseline (from multi-K JSON which includes canonical split_seed=42)
    add_method("K2_only", multi_k["k2_only_baseline_per_split"])
    # Multi-K (K=1 + K=2 per-seed averaging)
    add_method("multi_K", multi_k["multi_k_per_split"])
    # Multi-K with K=3 ComParE-LR
    add_method("multi_K_with_K3_compare_lr", compare["multi_k_with_k3_per_split"])
    # K=3-only (ComParE-LR alone fused with A2.5; floor test)
    add_method("K3_only_compare_lr", compare["k3only_per_split"])
    # ComParE-LR standalone (M14 pre-flight)
    add_method("compare_lr_standalone", compare["standalone_per_split"])
    # Diverse-anchor 10-seed (A1.0 / α-sweep variants)
    add_method("multi_K_diverse_10seed", diverse["10seed_diverse_per_split"])
    # Speaker smoothing at best α (shadow eval)
    smoothing_splits = smoothing["shadow_eval_at_best_alpha"]["smoothed_per_split"]
    add_method("speaker_smoothing_best_alpha", smoothing_splits)

    write_csv(
        "shadow_distributions_long.csv",
        [
            "method", "split_seed", "partition",
            "n_test_chunks", "devel_test_uar", "recall_C", "recall_NC",
        ],
        long_rows,
    )

    # Aggregate summary (one row per method)
    summary_rows: list[list[Any]] = []

    def shadow_stats(per_split: list[dict[str, Any]]) -> tuple[float | None, float, float, int]:
        """Return (canonical, shadow_mean, shadow_std, n_shadow)."""
        canon = None
        shadows: list[float] = []
        for r in per_split:
            u = r.get("uar")
            if u is None:
                continue
            if r.get("split_seed") == 42:
                canon = float(u)
            else:
                shadows.append(float(u))
        if not shadows:
            return canon, float("nan"), float("nan"), 0
        m = sum(shadows) / len(shadows)
        v = sum((x - m) ** 2 for x in shadows) / max(1, len(shadows) - 1)
        return canon, m, v**0.5, len(shadows)

    methods = {
        "K2_only": multi_k["k2_only_baseline_per_split"],
        "multi_K": multi_k["multi_k_per_split"],
        "multi_K_with_K3_compare_lr": compare["multi_k_with_k3_per_split"],
        "K3_only_compare_lr": compare["k3only_per_split"],
        "compare_lr_standalone": compare["standalone_per_split"],
        "multi_K_diverse_10seed": diverse["10seed_diverse_per_split"],
        "speaker_smoothing_best_alpha": smoothing_splits,
    }
    for name, splits in methods.items():
        canon, sm, sstd, n = shadow_stats(splits)
        summary_rows.append([name, fnum(canon), fnum(sm), fnum(sstd, 5), n])

    # Add paper-cited paired-lift summaries (from JSON aggregates)
    paired_rows = [
        [
            "multi_K_vs_K2_only", "multi-K paired lift over K=2-only across 10 shadow splits + canonical",
            fnum(multi_k["canonical_lift"]),
            fnum(multi_k["shadow_lift"]),
            fnum(multi_k["shadow_lift_std"], 5),
            multi_k["n_positive_shadows"],
            10,
        ],
        [
            "multi_K_with_K3_vs_multi_K", "multi-K-with-K=3 (ComParE-LR) paired lift over multi-K",
            fnum(compare["canonical_lift"]),
            fnum(compare["shadow_lift"]),
            fnum(compare["shadow_lift_std"], 5),
            compare["n_positive_shadows"],
            10,
        ],
        [
            "diverse_anchor_vs_5seed", "10-seed diverse-anchor paired lift over 5-seed multi-K",
            fnum(diverse["canonical_lift"]),
            fnum(diverse["shadow_lift"]),
            fnum(diverse["shadow_lift_std"], 5),
            diverse["n_positive_shadows"],
            10,
        ],
    ]

    # Save shadow_summary.csv and paired_lift_summary.csv as separate files
    write_csv(
        "shadow_summary.csv",
        ["method", "canonical_uar", "shadow_mean", "shadow_std", "n_shadow_splits"],
        summary_rows,
    )
    write_csv(
        "paired_lift_summary.csv",
        ["comparison", "description", "canonical_lift", "shadow_lift_mean", "shadow_lift_std", "n_positive", "n_total"],
        paired_rows,
    )

    # Save the canonical_z metric from the shadow-only run (anchored to K=2-only baseline)
    z_rows = [
        [
            "K2_only", "shadow-distribution z-score for canonical K=2-only ensemble",
            fnum(shadow_only["ref_canonical"]["ensemble_uar"]),
            fnum(shadow_only["aggregate"]["ensemble_uar_across_shadow"]["mean"]),
            fnum(shadow_only["aggregate"]["ensemble_uar_across_shadow"]["std"], 5),
            fnum(shadow_only["delta_canonical_vs_shadow_mean"]),
            fnum(shadow_only["canonical_z_in_shadow_distribution"], 3),
        ],
    ]
    write_csv(
        "canonical_z_score.csv",
        [
            "method", "description",
            "canonical_uar", "shadow_mean", "shadow_std",
            "delta_canonical_minus_shadow_mean", "canonical_z_sigma",
        ],
        z_rows,
    )


# ---------------------------------------------------------------------------
# 3. STANDALONE-UAR-PREDICTOR (M14)
# ---------------------------------------------------------------------------
def build_standalone_uar_predictor() -> None:
    """Per-candidate standalone UAR + K-fusion verdict, mirroring paper Table M14."""
    rows: list[list[Any]] = []

    # Pull standalone numbers from per-candidate JSONs
    # G4_gain_invariant + G5_modulation are inside A5b extended betasweep
    a5b = load("A5b_grouped_honestprior_betasweep_extended.json")

    # G4 standalone via g4_alone_aggregate
    g4_alone = a5b["g4_alone_aggregate"]
    rows.append([
        "G4_gain_invariant", "handcrafted gain-invariant family (3 dims; cold-LR probe)",
        fnum(g4_alone.get("mean")), fnum(g4_alone.get("std"), 5), None,
        "K=1_ADMIT (G_other winner under exhaustive K=1 sweep, K=2 locked at β=8)", None,
    ])
    # G5_modulation winner under exhaustive K=2 sweep
    # We don't have a "standalone G5" row in this JSON, but we have it implicit
    # via the K=2 exhaustive sweep — record what we have from compare and k3 hubert
    k3_egemaps = load("A5b_k3_egemaps_5seed.json")
    # eGeMAPS standalone is identical across seeds (deterministic LR on fixed features) — take first seed
    eg_per_seed = k3_egemaps["g_egemaps_standalone_uar_per_seed"]
    eg_uars = [v["uar_devel_test"] for v in eg_per_seed.values()]
    rows.append([
        "G_egemaps_v02_full_88d",
        "eGeMAPSv02 superset replacement / addition",
        fnum(sum(eg_uars) / len(eg_uars)),
        None, None,
        "K=3_NO_ADMIT (audit-driven slicing validates)",
        None,
    ])
    # HuBERT-base mean-pooled
    k3_hub = load("A5b_k3_hubert_5seed.json")
    rows.append([
        "HuBERT_base_meanpool",
        "HuBERT-base mean-pooled chunk repr (M14 pre-flight: fail)",
        fnum(k3_hub["hubert_standalone"]["uar_devel_test"]),
        None, None,
        f"K=3_NO_ADMIT (m14_pre_flight: {k3_hub['m14_pre_flight']['verdict']})",
        None,
    ])
    # HuBERT-base + LW softmax (A2.5-style)
    k3_hub_lw = load("A5b_k3_hubert_lw_5seed.json")
    rows.append([
        "HuBERT_base_LW_softmax",
        "HuBERT-base + honesty-prior layer-weighted softmax (A2.5 architecture transferred)",
        fnum(k3_hub_lw["standalone_5seed"]["mean"]),
        fnum(k3_hub_lw["standalone_5seed"]["std"]),
        None,
        f"K=3_NO_ADMIT (logit-correlation w/ WavLM-A2.5 anchor)",
        None,
    ])
    # ComParE-LR
    compare = load("A5b_compare_svm_k3.json")
    rows.append([
        "ComParE_2016_LR",
        "classical 6373-d ComParE-2016 functionals + regularised cold-LR",
        fnum(compare["standalone_canonical_uar"]),
        fnum(compare.get("standalone_shadow_std")),
        fnum(compare["standalone_shadow_mean"]),
        f"K=3_NEUTRAL (canonical {compare['canonical_lift']:+.4f}, shadow paired {compare['shadow_lift']:+.5f}, {compare['n_positive_shadows']}/10 positive)",
        None,
    ])

    write_csv(
        "standalone_uar_predictor.csv",
        [
            "candidate", "description",
            "standalone_canonical_uar", "standalone_std_or_seed_std", "standalone_shadow_mean",
            "K_fusion_verdict", "locked_beta",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# 4. METHODOLOGY TABLE M8–M19 (text rows for paper Table 1)
# ---------------------------------------------------------------------------
def build_methodology_table() -> None:
    rows = [
        ["M8", "de-confounding ladder", "audio-splice self-control: A5.5 phase 3.5"],
        ["M9", "de-confounding ladder", "α-endpoint controls: α∈{0,1} canonical points anchor mixup α-sweep"],
        ["M10", "de-confounding ladder", "bottleneck-confound test: 128-d projection vs 4096-d un-bottlenecked substrate"],
        ["M11", "de-confounding ladder", "subtractive auxiliary-objective interaction: λ-grid with main loss only as reference"],
        ["M12", "de-confounding ladder", "in-flight memorisation-gap probe at every epoch (loss-vs-UAR-vs-probe trajectory)"],
        ["M13", "de-confounding ladder", "disc-ceiling-before-λ-sweep ordering: A7 baseline before adversarial sweep"],
        ["M14", "de-confounding ladder", "substrate noise floor (speaker-probe top-1 ceiling at this corpus scale) + standalone-UAR-predictor for K-fusion admission"],
        ["M15", "ensemble aggregation", "monotonic-calibration invariance: isotonic must not move ranking-only UAR"],
        ["M16", "ensemble aggregation", "small-calibration-split LR-stacking overfit: LR-stacked weights beat grid only on tiny held-out, lose on full devel_test"],
        ["M17", "ensemble aggregation", "FM-substrate TTA must use perturbations that survive input normalisation"],
        ["M18", "shadow-split robustness", "cheap canonical-pipeline-logit-replay across N alternative devel partitions to quantify devel-side overfit risk on small-data corpora"],
        ["M19", "sparse-minority-class corollary", "speaker-level aggregation increases variance rather than reducing it when minority-class speaker count is small (URTIC chunk-UAR σ=0.017 vs speaker-UAR σ=0.082)"],
    ]
    write_csv(
        "methodology_table.csv",
        ["m_id", "category", "discipline_one_line"],
        rows,
    )


# ---------------------------------------------------------------------------
# 5. SPEAKER-SMOOTHING α-SWEEP CURVE
# ---------------------------------------------------------------------------
def build_speaker_smoothing_sweep() -> None:
    sm = load("A5b_k2_speaker_smoothing.json")
    rows: list[list[Any]] = []
    for r in sm["alpha_sweep_on_canonical"]:
        rows.append([
            fnum(r["alpha"], 3),
            fnum(r["tau"], 3),
            fnum(r["uar_train_threshold"]),
            fnum(r["canonical_devel_test_uar"]),
            fnum(r["canonical_recall_C"]),
            fnum(r["canonical_recall_NC"]),
        ])
    write_csv(
        "speaker_smoothing_alpha_sweep.csv",
        [
            "alpha", "tau_at_train_threshold",
            "train_threshold_uar", "canonical_devel_test_uar",
            "canonical_recall_C", "canonical_recall_NC",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# 6. β-SWEEP K=2 (extended grid)
# ---------------------------------------------------------------------------
def build_beta_sweep_k2() -> None:
    a5b = load("A5b_grouped_honestprior_betasweep_extended.json")
    rows: list[list[Any]] = []
    per_beta = a5b.get("per_beta_aggregate") or {}
    for beta, agg in per_beta.items():
        if isinstance(agg, dict):
            tt = agg.get("uar_train_threshold") or {}
            dt = agg.get("uar_devel_test") or {}
            rows.append([
                fnum(float(beta), 2),
                fnum(tt.get("mean")),
                fnum(tt.get("std"), 5),
                fnum(dt.get("mean")),
                fnum(dt.get("std"), 5),
                tt.get("n"),
            ])
    # Sort by beta numerically
    rows.sort(key=lambda r: r[0])
    write_csv(
        "beta_sweep_k2.csv",
        [
            "beta",
            "uar_train_threshold_mean", "uar_train_threshold_std",
            "uar_devel_test_argmax_mean", "uar_devel_test_argmax_std",
            "n_seeds",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# 7. CALIBRATION + TTA ABLATIONS
# ---------------------------------------------------------------------------
def build_ablations_calibration_tta() -> None:
    cal = load("A5b_k2_ensemble_calibrated.json")
    tta = load("A5b_k2_tta_ensemble.json")
    rows: list[list[Any]] = []
    # Calibration variants
    ml = cal["mean_logit"]
    rows.append(["mean_logit", "5-seed K=2 mean-logit (canonical headline)",
                 fnum(ml["devel_test_uar"]), fnum(ml.get("recall_C")), fnum(ml.get("recall_NC")), 0.0])
    lr = cal["lr_stacked"]
    rows.append(["lr_stacked", "5-seed K=2 LR-stacked weights on small held-out",
                 fnum(lr["devel_test_uar"]), fnum(lr.get("recall_C")), fnum(lr.get("recall_NC")),
                 fnum(lr["devel_test_uar"] - ml["devel_test_uar"])])
    gs = cal["uar_grid_search"]
    rows.append(["uar_grid_search", "5-seed K=2 best per-seed weights via UAR grid",
                 fnum(gs["devel_test_uar"]), fnum(gs.get("recall_C")), fnum(gs.get("recall_NC")),
                 fnum(gs["devel_test_uar"] - ml["devel_test_uar"])])
    iso = cal["isotonic_ablation"]
    rows.append(["isotonic_calibrated", "5-seed K=2 mean-logit + isotonic on devel_val",
                 fnum(iso["devel_test_uar"]), None, None,
                 fnum(iso["delta_vs_pre_iso"])])
    # TTA variants
    orig = tta["original_only_reproduction"]
    rows.append(["tta_original_only", "TTA pipeline reproduction (original only)",
                 fnum(orig["devel_test_uar"]), None, None,
                 fnum(orig.get("delta_vs_ref_0p7090"))])
    te = tta["tta_ensemble"]
    rows.append(["tta_5x", "5-version TTA mean-logit (original + 4 augmentations)",
                 fnum(te["tau_devel_test_uar"]), fnum(te.get("recall_C")), fnum(te.get("recall_NC")),
                 fnum(te["delta_vs_no_tta"])])

    write_csv(
        "ablations_calibration_tta.csv",
        ["variant", "description", "devel_test_uar", "recall_C", "recall_NC", "delta_vs_mean_logit"],
        rows,
    )


# ---------------------------------------------------------------------------
# 8. PER-SEED LOCKED β / τ / UAR
# ---------------------------------------------------------------------------
def build_per_seed_locked() -> None:
    multi_k = load("A5b_k2_multi_k_ensemble.json")
    k2lock = load("A5b_k2_5seed_lock.json")
    rows: list[list[Any]] = []
    seeds_order = multi_k["all_seeds"]
    for seed in seeds_order:
        s = str(seed)
        beta_k1 = multi_k["locked_betas_k1"].get(s)
        beta_k2 = multi_k["locked_betas_k2"].get(s)
        per_seed = k2lock["per_seed"].get(s, {})
        # Per-seed locked numbers are nested under "k1_locked"/"k2_locked"
        k1 = per_seed.get("k1_locked", {}) if isinstance(per_seed, dict) else {}
        k2 = per_seed.get("k2_locked", {}) if isinstance(per_seed, dict) else {}
        a25_uar = per_seed.get("a2_arg_uar") if isinstance(per_seed, dict) else None
        k1_uar = (k1.get("devel_test") or {}).get("uar")
        k2_uar = (k2.get("devel_test") or {}).get("uar")
        k1_tau = k1.get("tau")
        k2_tau = k2.get("tau")
        rows.append([
            seed, fnum(beta_k1, 2), fnum(beta_k2, 2),
            fnum(k1_tau, 3), fnum(k2_tau, 3),
            fnum(a25_uar), fnum(k1_uar), fnum(k2_uar),
        ])
    write_csv(
        "per_seed_locked.csv",
        ["seed", "beta_K1_locked", "beta_K2_locked", "tau_K1_locked", "tau_K2_locked",
         "a25_uar", "k1_locked_uar", "k2_locked_uar"],
        rows,
    )


# ---------------------------------------------------------------------------
# 9. HUBERT-A2.5 LAYER AUDIT + PER-SEED + FINAL WEIGHTS
# ---------------------------------------------------------------------------
def build_hubert_a25() -> None:
    d = load("A5b_k3_hubert_lw_5seed.json")

    # Per-layer audit
    rows: list[list[Any]] = []
    for r in d["per_layer_audit"]:
        rows.append([
            r["layer"],
            fnum(r["cold_uar"]),
            fnum(r["speaker_top1"]),
            r["n_pseudo"],
            fnum(r["label_gain"]),
            fnum(r["speaker_gain"]),
            fnum(r["sub_at_1"]),
        ])
    write_csv(
        "hubert_a25_layer_audit.csv",
        ["layer", "cold_uar", "speaker_top1", "n_pseudo", "label_gain", "speaker_gain", "sub_at_1"],
        rows,
    )

    # Per-seed standalone
    rows = []
    for r in d["standalone_per_seed"]:
        rows.append([
            r["seed"], fnum(r["tau"], 3),
            fnum(r.get("uar_train_thr")), fnum(r["uar_devel_test"]),
            fnum(r.get("recall_C")), fnum(r.get("recall_NC")),
        ])
    write_csv(
        "hubert_a25_per_seed_standalone.csv",
        ["seed", "tau", "uar_train_threshold", "uar_devel_test", "recall_C", "recall_NC"],
        rows,
    )

    # Final learned layer weights (per seed) — wide
    rows = []
    n_layers = d.get("hubert_n_layers", 13)
    for r in d["head_training_per_seed"]:
        flw = r.get("final_layer_weights") or []
        rows.append([r["seed"], fnum(r.get("best_val_uar")), r.get("best_epoch"),
                     fnum(r.get("cos_sub_at_1_vs_final"), 5),
                     *[fnum(w, 5) for w in flw]])
    write_csv(
        "hubert_a25_final_layer_weights.csv",
        ["seed", "best_val_uar", "best_epoch", "cos_sub_at_1_vs_final",
         *[f"L{i}" for i in range(n_layers)]],
        rows,
    )


# ---------------------------------------------------------------------------
# 10. PER-SPLIT TABLE for ALL multi-K VARIANTS (helper for ROC/PR ablations)
# ---------------------------------------------------------------------------
def build_multi_k_per_split() -> None:
    """Single CSV: per-split UAR + recall pattern for every multi-K variant we tested.

    Useful for the per-split spread figure (canonical vs shadow box plot).
    """
    multi_k = load("A5b_k2_multi_k_ensemble.json")
    compare = load("A5b_compare_svm_k3.json")
    diverse = load("A5b_multi_k_10seed_diverse.json")

    rows: list[list[Any]] = []
    sources = {
        "K2_only_5seed_mean_logit": multi_k["k2_only_baseline_per_split"],
        "multi_K_5seed_mean_logit": multi_k["multi_k_per_split"],
        "multi_K_with_K3_compare_lr": compare["multi_k_with_k3_per_split"],
        "K3_only_compare_lr": compare["k3only_per_split"],
        "multi_K_10seed_diverse_anchor": diverse["10seed_diverse_per_split"],
    }
    for method, splits in sources.items():
        for r in splits:
            rows.append([
                method, r.get("split_seed"),
                "canonical" if r.get("split_seed") == 42 else "shadow",
                r.get("n_test"), fnum(r.get("uar")),
                fnum(r.get("recall_C")), fnum(r.get("recall_NC")),
            ])
    write_csv(
        "multi_k_per_split.csv",
        ["method", "split_seed", "partition", "n_test", "uar", "recall_C", "recall_NC"],
        rows,
    )


# ---------------------------------------------------------------------------
# 11. ComParE-LR per-shadow (separate file for clarity in the C1 framing)
# ---------------------------------------------------------------------------
def build_compare_lr_per_shadow() -> None:
    d = load("A5b_compare_svm_k3.json")
    rows: list[list[Any]] = []
    for r in d["standalone_per_split"]:
        rows.append([
            r.get("split_seed"),
            "canonical" if r.get("split_seed") == 42 else "shadow",
            r.get("n_test"),
            fnum(r.get("uar")),
            fnum(r.get("recall_C")),
            fnum(r.get("recall_NC")),
        ])
    # Aggregate summary
    rows.append([
        "AGGREGATE_shadow", "shadow_mean",
        None,
        fnum(d["standalone_shadow_mean"]),
        None, None,
    ])
    rows.append([
        "AGGREGATE_shadow", "shadow_std",
        None,
        fnum(d.get("standalone_shadow_std"), 5),
        None, None,
    ])
    write_csv(
        "compare_lr_per_shadow.csv",
        ["split_seed", "partition", "n_test", "uar", "recall_C", "recall_NC"],
        rows,
    )


# ---------------------------------------------------------------------------
# 12. COPY EXISTING LAYER-AUDIT CSVs verbatim
# ---------------------------------------------------------------------------
def copy_layer_audit_csvs() -> None:
    pairs = [
        ("A5d_layer_honesty.csv", "layer_audit_wavlm.csv"),
        ("A5d_grouped_layer_honesty.csv", "layer_audit_wavlm_grouped.csv"),
        ("A5d_hubert_layer_honesty.csv", "layer_audit_hubert.csv"),
        ("A5a_honesty.csv", "handcrafted_group_honesty.csv"),
    ]
    for src, dst in pairs:
        s = RESULTS / src
        if s.exists():
            shutil.copy(s, OUT / dst)
            print(f"  copied {s.name} -> {dst}")
        else:
            print(f"  MISSING {s.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Project root: {ROOT}")
    print(f"Writing to: {OUT}")
    print()
    print("[1/12] cumulative_stack.csv")
    build_cumulative_stack()
    print("[2/12] shadow_distributions_long + shadow_summary + paired_lift + canonical_z")
    build_shadow_distributions()
    print("[3/12] standalone_uar_predictor.csv")
    build_standalone_uar_predictor()
    print("[4/12] methodology_table.csv")
    build_methodology_table()
    print("[5/12] speaker_smoothing_alpha_sweep.csv")
    build_speaker_smoothing_sweep()
    print("[6/12] beta_sweep_k2.csv")
    build_beta_sweep_k2()
    print("[7/12] ablations_calibration_tta.csv")
    build_ablations_calibration_tta()
    print("[8/12] per_seed_locked.csv")
    build_per_seed_locked()
    print("[9/12] hubert_a25_layer_audit + per_seed + final_layer_weights")
    build_hubert_a25()
    print("[10/12] multi_k_per_split.csv")
    build_multi_k_per_split()
    print("[11/12] compare_lr_per_shadow.csv")
    build_compare_lr_per_shadow()
    print("[12/12] copy layer-audit CSVs verbatim")
    copy_layer_audit_csvs()
    print()
    print("DONE.")
