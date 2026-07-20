"""Create a cold-label-free, blinded URTIC same/different-speaker audit.

The 300-pair set is balanced across official sides and five method-agreement
strata. Pair selection uses only ECAPA/TRILLsson embeddings and speaker-proxy
partitions. The public CSV/HTML omit all model predictions; the key must remain
hidden until both annotators have submitted independent labels.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
ECAPA = ROOT / "cache/ecapa-voxceleb/ecapa_embeddings.npz"
ECAPA_LABELS = ROOT / "results/speaker_proxy_method_labels.npz"
OUT_CSV = ROOT / "results/speaker_pair_annotation_blinded.csv"
OUT_KEY = ROOT / "results/speaker_pair_annotation_key.json"
OUT_HTML = ROOT / "results/speaker_pair_annotation.html"
OUT_INSTRUCTIONS = ROOT / "results/speaker_pair_annotation_instructions.md"
SIDES = ("train", "devel", "test")
METHODS = ("kmeans", "agglomerative", "spectral")
STRATA = (
    "ecapa_same_trillsson_different",
    "trillsson_same_ecapa_different",
    "ecapa_internal_disagreement",
    "all_methods_same",
    "all_methods_different",
)
PER_SIDE_STRATUM = 20
NEIGHBOURS_PER_VIEW = 30
SEED = 20260720


def aligned_side(side: str, ecapa_archive, label_archive) -> dict:
    stems = label_archive[f"{side}__stems"].astype(str)

    e_stems = ecapa_archive["stems"].astype(str)
    e_lookup = {stem: i for i, stem in enumerate(e_stems)}
    missing = [stem for stem in stems if stem not in e_lookup]
    if missing:
        raise RuntimeError(f"{side}: ECAPA missing {len(missing)} stems")
    ecapa_x = normalize(
        ecapa_archive["embeddings"][[e_lookup[s] for s in stems]].astype(np.float32)
    ).astype(np.float32)

    trill_archive = np.load(
        ROOT / f"cache/trillsson1/urtic_{side}_w3.npz", allow_pickle=False
    )
    t_stems = trill_archive["stems"].astype(str)
    t_lookup = {stem: i for i, stem in enumerate(t_stems)}
    missing = [stem for stem in stems if stem not in t_lookup]
    if missing:
        raise RuntimeError(f"{side}: TRILLsson embedding cache missing {len(missing)} stems")
    trill_x = normalize(
        trill_archive["embeddings"][[t_lookup[s] for s in stems]].astype(np.float32)
    ).astype(np.float32)

    trill_labels_archive = np.load(
        ROOT / f"results/trillsson1_labels_{side}.npz", allow_pickle=True
    )
    tl_stems = trill_labels_archive["stems"].astype(str)
    tl_lookup = {stem: i for i, stem in enumerate(tl_stems)}
    trill_labels = trill_labels_archive["seed42"][[tl_lookup[s] for s in stems]]

    return {
        "stems": stems,
        "ecapa_x": ecapa_x,
        "trill_x": trill_x,
        "ecapa_labels": {
            method: label_archive[f"{side}__{method}"].astype(np.int32)
            for method in METHODS
        },
        "trill_labels": trill_labels.astype(np.int32),
    }


def candidate_pairs(x_views: tuple[np.ndarray, ...]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for x in x_views:
        idx = NearestNeighbors(
            n_neighbors=NEIGHBOURS_PER_VIEW + 1,
            metric="euclidean",
            n_jobs=-1,
        ).fit(x).kneighbors(x, return_distance=False)[:, 1:]
        for i, neighbours in enumerate(idx):
            for j in neighbours:
                a, b = sorted((i, int(j)))
                pairs.add((a, b))
    return pairs


def classify_pair(data: dict, i: int, j: int) -> dict:
    ecapa_same = {
        method: bool(labels[i] == labels[j])
        for method, labels in data["ecapa_labels"].items()
    }
    trill_same = bool(data["trill_labels"][i] == data["trill_labels"][j])
    n_ecapa_same = sum(ecapa_same.values())
    if n_ecapa_same == 3 and trill_same:
        stratum = "all_methods_same"
    elif n_ecapa_same == 3 and not trill_same:
        stratum = "ecapa_same_trillsson_different"
    elif n_ecapa_same == 0 and trill_same:
        stratum = "trillsson_same_ecapa_different"
    elif 0 < n_ecapa_same < 3:
        stratum = "ecapa_internal_disagreement"
    else:
        stratum = "all_methods_different"
    sim_ecapa = float(np.dot(data["ecapa_x"][i], data["ecapa_x"][j]))
    sim_trill = float(np.dot(data["trill_x"][i], data["trill_x"][j]))
    return {
        "i": i,
        "j": j,
        "stratum": stratum,
        "ecapa_same": ecapa_same,
        "trillsson_same": trill_same,
        "cosine_ecapa": sim_ecapa,
        "cosine_trillsson": sim_trill,
        "hardness": max(sim_ecapa, sim_trill),
    }


def select_unique(candidates: list[dict], count: int, used: set[int], rng) -> list[dict]:
    # Random jitter prevents choosing an arbitrary lexicographic slice while
    # retaining hard/high-similarity disagreement cases.
    ranked = sorted(
        candidates,
        key=lambda row: row["hardness"] + float(rng.uniform(0, 0.01)),
        reverse=True,
    )
    chosen = []
    for row in ranked:
        if row["i"] in used or row["j"] in used:
            continue
        chosen.append(row)
        used.update((row["i"], row["j"]))
        if len(chosen) == count:
            return chosen
    raise RuntimeError(f"could select only {len(chosen)}/{count} unique pairs")


def html_document(public_pairs: list[dict]) -> str:
    payload = json.dumps(public_pairs, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded URTIC speaker-pair annotation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#17202a}}
.card{{border:1px solid #ccd1d1;border-radius:12px;padding:1.2rem;box-shadow:0 2px 8px #0001}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}} audio{{width:100%}}
button{{padding:.75rem 1rem;margin:.35rem;border:1px solid #777;border-radius:8px;background:#fff;cursor:pointer}}
button.selected{{background:#1f618d;color:white}} input,textarea{{padding:.5rem;margin:.3rem}}
#progress{{margin:1rem 0}} .muted{{color:#626567;font-size:.92rem}}
</style></head><body>
<h1>Blinded same/different-speaker annotation</h1>
<p class="muted">Listen independently. Judge identity only; ignore cold symptoms and words. Use Unsure when evidence is insufficient.</p>
<label>Annotator ID <input id="annotator" placeholder="initials or code"></label>
<div id="progress"></div><div class="card">
<h2 id="pairid"></h2><div class="row">
<div><p>Recording A</p><audio id="audioA" controls></audio></div>
<div><p>Recording B</p><audio id="audioB" controls></audio></div></div>
<div><button data-label="same">Same speaker</button><button data-label="different">Different speakers</button><button data-label="unsure">Unsure</button></div>
<label>Confidence <select id="confidence"><option value="">--</option><option value="1">1 low</option><option value="2">2 medium</option><option value="3">3 high</option></select></label><br>
<label>Notes<br><textarea id="notes" rows="2" cols="70"></textarea></label><br>
<button id="prev">Previous</button><button id="next">Save & next</button><button id="download">Download CSV</button>
</div>
<script>
const pairs={payload}; let index=0; const answers={{}};
const q=s=>document.querySelector(s);
function save(){{const p=pairs[index]; if(!p)return; answers[p.pair_id]={{same_speaker:q('button.selected')?.dataset.label||'',confidence:q('#confidence').value,notes:q('#notes').value}};}}
function render(){{const p=pairs[index],a=answers[p.pair_id]||{{}};q('#pairid').textContent=p.pair_id;q('#audioA').src='../'+p.audio_a;q('#audioB').src='../'+p.audio_b;q('#confidence').value=a.confidence||'';q('#notes').value=a.notes||'';document.querySelectorAll('button[data-label]').forEach(b=>b.classList.toggle('selected',b.dataset.label===a.same_speaker));q('#progress').textContent=`Pair ${{index+1}} / ${{pairs.length}} · answered ${{Object.values(answers).filter(x=>x.same_speaker).length}}`;}}
document.querySelectorAll('button[data-label]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button[data-label]').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');}});
q('#next').onclick=()=>{{save();if(index<pairs.length-1)index++;render();}};q('#prev').onclick=()=>{{save();if(index>0)index--;render();}};
q('#download').onclick=()=>{{save();const esc=x=>'"'+String(x??'').replaceAll('"','""')+'"';let csv='pair_id,audio_a,audio_b,same_speaker,confidence_1_3,notes,annotator_id\\n';for(const p of pairs){{const a=answers[p.pair_id]||{{}};csv+=[p.pair_id,p.audio_a,p.audio_b,a.same_speaker||'',a.confidence||'',a.notes||'',q('#annotator').value].map(esc).join(',')+'\\n';}}const blob=new Blob([csv],{{type:'text/csv'}}),u=URL.createObjectURL(blob),link=document.createElement('a');link.href=u;link.download='speaker_pairs_'+(q('#annotator').value||'annotator')+'.csv';link.click();URL.revokeObjectURL(u);}};render();
</script></body></html>"""


def main() -> None:
    rng = np.random.default_rng(SEED)
    ecapa_archive = np.load(ECAPA, allow_pickle=True)
    label_archive = np.load(ECAPA_LABELS, allow_pickle=True)
    selected = []
    candidate_counts = {}

    # Rare disagreement categories go first so the no-recording-reuse rule
    # cannot consume their clips in easier control strata.
    selection_order = (
        "trillsson_same_ecapa_different",
        "ecapa_same_trillsson_different",
        "ecapa_internal_disagreement",
        "all_methods_same",
        "all_methods_different",
    )
    for side in SIDES:
        print(f"[{side}] align embeddings and build two-view neighbour union", flush=True)
        data = aligned_side(side, ecapa_archive, label_archive)
        candidates = [
            classify_pair(data, i, j)
            for i, j in candidate_pairs((data["ecapa_x"], data["trill_x"]))
        ]
        buckets = {name: [] for name in STRATA}
        for row in candidates:
            buckets[row["stratum"]].append(row)
        candidate_counts[side] = {name: len(rows) for name, rows in buckets.items()}
        used: set[int] = set()
        for stratum in selection_order:
            rows = select_unique(buckets[stratum], PER_SIDE_STRATUM, used, rng)
            for row in rows:
                row.update(
                    side=side,
                    stem_a=str(data["stems"][row.pop("i")]),
                    stem_b=str(data["stems"][row.pop("j")]),
                )
                selected.append(row)

    rng.shuffle(selected)
    public_pairs = []
    key_pairs = []
    for number, row in enumerate(selected, start=1):
        pair_id = f"SP{number:03d}"
        # Randomize A/B orientation independently of ranking and method outputs.
        stems = [row["stem_a"], row["stem_b"]]
        rng.shuffle(stems)
        public = {
            "pair_id": pair_id,
            "audio_a": f"dataset/ComParE2017_Cold_4students/wav/{stems[0]}.wav",
            "audio_b": f"dataset/ComParE2017_Cold_4students/wav/{stems[1]}.wav",
        }
        public_pairs.append(public)
        key_pairs.append({"pair_id": pair_id, **row})

    public_df = pd.DataFrame(public_pairs)
    public_df["same_speaker"] = ""
    public_df["confidence_1_3"] = ""
    public_df["notes"] = ""
    public_df["annotator_id"] = ""
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    public_df.to_csv(OUT_CSV, index=False)
    public_sha = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    key = {
        "protocol": {
            "cold_labels_loaded": False,
            "n_pairs": len(key_pairs),
            "sides": list(SIDES),
            "pairs_per_side_per_stratum": PER_SIDE_STRATUM,
            "strata": list(STRATA),
            "nearest_neighbours_per_embedding_view": NEIGHBOURS_PER_VIEW,
            "recording_reuse": "none within each side; selected pairs use 600 unique recordings",
            "selection_note": (
                "Disagreement-enriched case-control sample. Method accuracy on this set is not "
                "a population prevalence estimate."
            ),
            "seed": SEED,
            "blinded_csv_sha256": public_sha,
        },
        "candidate_counts": candidate_counts,
        "pairs": key_pairs,
    }
    OUT_KEY.write_text(json.dumps(key, indent=2), encoding="utf-8")
    OUT_HTML.write_text(html_document(public_pairs), encoding="utf-8")
    instructions = """# Blinded URTIC speaker-pair annotation

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
"""
    OUT_INSTRUCTIONS.write_text(instructions, encoding="utf-8")
    print(json.dumps({
        "n_pairs": len(key_pairs),
        "unique_recordings": len({p["stem_a"] for p in key_pairs} | {p["stem_b"] for p in key_pairs}),
        "candidate_counts": candidate_counts,
        "blinded_csv_sha256": public_sha,
        "outputs": [str(p.relative_to(ROOT)) for p in (OUT_CSV, OUT_HTML, OUT_KEY, OUT_INSTRUCTIONS)],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
