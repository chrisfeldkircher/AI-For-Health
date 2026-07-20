"""Quantify the garden-of-forking-paths optimism on devel_test.

Multiple candidate systems were admitted/rejected by comparing their devel_test
UAR to a bar. Each individual number is honest, but the SELECTED winner is
optimistically biased (winner's curse): with N candidates whose means carry
seed-level noise, the observed max overshoots the true mean of the argmax.

Method:
  1. Collect every candidate config with per-seed locked devel_test UARs from
     results/*.json (recursive walk for runs = [{seed, locked:{devel_test:{uar}}}]
     patterns plus the K1/K2 per_seed lock structure).
  2. Winner's-curse Monte Carlo: model each candidate mean as
     N(observed_mean, SE^2) with SE = seed_std/sqrt(n). Draw a noisy observed
     mean per candidate, pick the argmax, record (observed max - true mean of
     the argmax) where 'true mean' is the observed mean (best available
     estimate). Average over draws = expected selection optimism.
  3. Report alongside: N candidates, spread of candidate means, per-candidate SEs.

Output: results/audit_forking_paths.json
Run from repo root:  python audit_forking_paths.py   (seconds)
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
RESULTS = ROOT / "results"
OUT = RESULTS / "audit_forking_paths.json"
RNG = np.random.default_rng(42)
N_MC = 200_000

candidates: dict[str, list[float]] = {}


def walk(obj, path, fname):
    """Find per-seed locked devel_test UAR lists."""
    if isinstance(obj, dict):
        # pattern A: {"runs": [{"seed":..., "locked": {"devel_test": {"uar": x}}}]}
        runs = obj.get("runs")
        if isinstance(runs, list) and runs and isinstance(runs[0], dict) \
           and "locked" in runs[0] and "seed" in runs[0]:
            uars = []
            for r in runs:
                dt = r.get("locked", {}).get("devel_test")
                if isinstance(dt, dict) and "uar" in dt:
                    uars.append(float(dt["uar"]))
            if len(uars) >= 3:
                candidates[f"{fname}:{path}"] = uars
        # pattern B: per_seed lock structure {seed: {k1_locked/k2_locked: {devel_test...}}}
        ps = obj.get("per_seed")
        if isinstance(ps, dict):
            for lockkey in ("k1_locked", "k2_locked"):
                uars = []
                for s, v in ps.items():
                    dt = v.get(lockkey, {}).get("devel_test") if isinstance(v, dict) else None
                    if isinstance(dt, dict) and "uar" in dt:
                        uars.append(float(dt["uar"]))
                if len(uars) >= 3:
                    candidates[f"{fname}:per_seed.{lockkey}"] = uars
        for k, v in obj.items():
            if k not in ("runs", "per_seed"):
                walk(v, f"{path}.{k}" if path else k, fname)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:20]):
            walk(v, f"{path}[{i}]", fname)


for f in sorted(RESULTS.glob("A5b*.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    walk(d, "", f.stem)

print(f"[collect] {len(candidates)} candidate configs with per-seed devel_test UARs:")
rows = []
for name, uars in sorted(candidates.items(), key=lambda kv: -st.mean(kv[1])):
    m = st.mean(uars)
    sd = st.stdev(uars) if len(uars) > 1 else 0.0
    se = sd / np.sqrt(len(uars))
    rows.append({"candidate": name, "n_seeds": len(uars),
                 "mean": m, "seed_std": sd, "se": float(se)})
    print(f"  {m:.4f} +/- {sd:.4f} (SE {se:.4f}, n={len(uars)})  {name}")

means = np.array([r["mean"] for r in rows])
ses = np.array([max(r["se"], 1e-6) for r in rows])
N = len(rows)

# Winner's-curse Monte Carlo
noisy = means[None, :] + RNG.standard_normal((N_MC, N)) * ses[None, :]
argmax = noisy.argmax(axis=1)
observed_max = noisy.max(axis=1)
true_of_winner = means[argmax]
optimism = observed_max - true_of_winner
sel_opt_mean = float(optimism.mean())
sel_opt_p95 = float(np.quantile(optimism, 0.95))

# How often does the noisy argmax differ from the true argmax?
flip_rate = float((argmax != int(means.argmax())).mean())

print(f"\n[winner's curse] N={N} candidates")
print(f"  expected selection optimism  = +{sel_opt_mean:.4f} UAR")
print(f"  95th percentile              = +{sel_opt_p95:.4f} UAR")
print(f"  argmax flip rate under noise = {flip_rate:.2%}")
print(f"  candidate mean spread        = {means.min():.4f} .. {means.max():.4f}")

out = {
    "rung_id": "audit_forking_paths",
    "description": (
        "Winner's-curse estimate for candidate selection on devel_test. Each "
        "candidate mean is modeled N(observed_mean, SE^2), SE from per-seed "
        "spread; Monte Carlo picks the argmax of noisy means and records "
        "observed max minus the (best-estimate) true mean of the picked "
        "candidate. This bounds how much the SELECTED system's reported "
        "devel_test UAR overshoots due to selection alone, not leakage. "
        "Caveats: candidates share seeds and features so their noise is "
        "positively correlated, which SHRINKS true selection optimism; treat "
        "the estimate as an upper-bound-flavored magnitude, not a precise "
        "correction. Assumes the candidate list found in results/A5b*.json is "
        "the full set of devel_test comparisons (undocumented discarded "
        "attempts would raise it)."),
    "n_candidates": N,
    "n_mc": N_MC,
    "candidates": rows,
    "expected_selection_optimism_uar": sel_opt_mean,
    "selection_optimism_p95_uar": sel_opt_p95,
    "argmax_flip_rate": flip_rate,
    "context": {
        "k2_locked_reference_mean": 0.7037,
        "shadow_sigma": 0.0157,
        "midterm_devel_to_test_gap": -0.0906,
    },
    "reading": (
        "If expected optimism is a few thousandths of UAR, forking paths is a "
        "minor contributor relative to the 0.09 devel->test gap; the dominant "
        "term remains pool identity."),
}
OUT.write_text(json.dumps(out, indent=2))
print(f"\n[wrote] {OUT.relative_to(ROOT)}")
