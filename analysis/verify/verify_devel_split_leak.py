"""
Verify the new audit: the OLD WavLM devel_val/devel_test split (grouped on the
shipped train-only-fit k210 IDs) was not speaker-disjoint, because those IDs
fragment devel speakers. Reproduce the headline (groups crossing the boundary)
AND contrast the loose 'in a crossing group' metric with the strict 'NN on the
other side' metric, so the 98% isn't misread as the size of the UAR leak.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
OUT = ROOT / "results" / "devel_split_leak_verification.json"
sys.path.insert(0, str(ROOT / "model"))
from data.cached_dataset import stratified_grouped_split, load_labels
from speakers.cluster import load_pseudo_speakers

NPZ = ROOT / "cache" / "ecapa-voxceleb" / "ecapa_embeddings.npz"
d = np.load(NPZ, allow_pickle=True)
stems = d["stems"].astype(str); split = d["split"].astype(str); emb = d["embeddings"].astype(np.float32)
shipped = load_pseudo_speakers(ROOT / "cache" / "pseudo_speakers" / "k210_seed42.tsv")
labels = load_labels(str(ROOT / "dataset" / "ComParE2017_Cold_4students"))

m = split == "devel"
Xdv = normalize(emb[m], axis=1)
sdv = stems[m]
idx = {s: i for i, s in enumerate(sdv)}
# corrected devel-LOCAL cohesive speaker proxies (KMeans on devel only)
devlocal = KMeans(210, n_init=10, random_state=42).fit_predict(Xdv)
dl = {s: int(devlocal[i]) for i, s in enumerate(sdv)}

print("=" * 78)
print("OLD devel_val/devel_test split (grouped on shipped train-only k210 IDs):")
print("  is it speaker-disjoint under CORRECTED devel-local groups?")
print("=" * 78)
rows = []
for seed in (42, 123, 7):
    files = sorted(f for f in labels if f.startswith("devel_"))
    val, test = stratified_grouped_split(files, labels, shipped, val_frac=0.5, seed=seed)
    val_st = set(f[:-4] for f in val); test_st = set(f[:-4] for f in test)
    side = {s: (0 if (s + ".wav") in set(val) else 1) for s in sdv}
    # groups (corrected) that appear on BOTH sides
    from collections import defaultdict
    g_sides = defaultdict(set)
    for s in sdv:
        g_sides[dl[s]].add(side[s])
    crossing = {g for g, sd in g_sides.items() if len(sd) > 1}
    n_cross_groups = len(crossing)
    n_rec_in_cross = sum(1 for s in sdv if dl[s] in crossing)
    frac_rec = n_rec_in_cross / len(sdv)
    # strict metric: NN on other side
    sarr = np.array([side[s] for s in sdv], np.int8)
    nn_other = 0; B = 2048
    for i0 in range(0, len(sdv), B):
        sim = Xdv[i0:i0+B] @ Xdv.T
        for r in range(sim.shape[0]):
            sim[r, i0+r] = -2.0
        nn_other += int((sarr[sim.argmax(1)] != sarr[i0:i0+B]).sum())
    nn_other_fraction = nn_other / len(sdv)
    rows.append({
        "seed": seed,
        "crossing_groups": n_cross_groups,
        "total_groups": 210,
        "recordings_in_crossing_groups": n_rec_in_cross,
        "recording_fraction_in_crossing_groups": frac_rec,
        "cross_boundary_nearest_neighbor_fraction": nn_other_fraction,
    })
    print(f"  seed {seed:<5} crossing groups {n_cross_groups:3d}/210  "
          f"recordings in crossing groups {frac_rec*100:5.2f}%  |  "
          f"cross-boundary nearest-neighbor {nn_other_fraction*100:5.2f}%")

# sanity: split grouped on the corrected devel-local groups -> zero crossing
files = sorted(f for f in labels if f.startswith("devel_"))
val2, test2 = stratified_grouped_split(files, labels, dl, val_frac=0.5, seed=42)
gv = {dl[f[:-4]] for f in val2}; gt = {dl[f[:-4]] for f in test2}
print(f"\n  CORRECTED devel-local grouped split: overlap groups = {len(gv & gt)} (expect 0)")

report = {
    "question": "Was the historical Development sub-split disjoint under a Development-local ECAPA proxy?",
    "grouping_used_to_construct_historical_split": "Train-fitted shipped k210 labels",
    "independent_audit_partition": "Development-local raw-L2 ECAPA KMeans k=210 seed=42",
    "seeds": rows,
    "corrected_side_local_seed42_overlapping_groups": len(gv & gt),
    "interpretation": (
        "The group-crossing and cross-boundary-nearest-neighbor diagnostics show "
        "that the historical boundary is not disjoint under the Development-local "
        "speaker proxy. Neither statistic estimates or bounds UAR optimism. A "
        "separate downstream comparison is required for score impact."
    ),
}
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nREADING: crossing-groups ~201/210 and ~98% recordings confirm that the old")
print("split was NOT disjoint under the Development-local proxy. The ~27%")
print("cross-boundary nearest-neighbor rate is a second connectivity diagnostic;")
print("it is not a mathematical bound or causal estimate of UAR optimism.")
print(f"[wrote] {OUT.relative_to(ROOT)}")
