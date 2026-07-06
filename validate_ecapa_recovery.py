"""External validation of the ECAPA pseudo-speaker pipeline against GROUND-TRUTH
speaker labels (the test our instructors asked for).

Our URTIC pipeline has no true speaker IDs, so we validate the identical pipeline
on a labeled corpus: extract ECAPA (speechbrain/spkrec-ecapa-voxceleb, 192-d) ->
L2-normalise -> KMeans(n_init=10, random_state=42) -> compare clusters to true
speakers. Measures cluster PURITY (the property our grouping needs: a cluster that
is ~1 speaker prevents leakage), speaker FRAGMENTATION (how many clusters one
speaker spans -- the failure we found on URTIC devel), plus ARI / NMI / homogeneity
/ completeness, a k-misspecification sweep (0.5x / 1x / 2x true count, since on
URTIC we never know true k), and a label-shuffle negative control.

Pipeline fidelity: reuses model/speakers/ecapa.load_ecapa_encoder and the exact
encode_batch call + the exact clustering settings from model/speakers/cluster.py.
The only difference vs URTIC is audio decode (librosa.load handles flac/mp3/wav
uniformly to the same 16 kHz mono float), which yields the same PCM the encoder
sees. Clustering is per-utterance (URTIC clusters per-chunk); the true label is the
utterance's speaker.

Usage:
  # smoke-test the metrics (no audio, no model):
  python validate_ecapa_recovery.py --smoke
  # real run from a manifest TSV of "audio_path<TAB>speaker_id" (one per line):
  python validate_ecapa_recovery.py --manifest path/to/manifest.tsv --name commonvoice_de
  # or a built-in adapter:
  python validate_ecapa_recovery.py --corpus librispeech --root path/to/train-clean-100 --name libri100
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))

# This box sits behind a TLS-intercepting proxy: requests/huggingface_hub reject
# the cert (certifi lacks the corporate root CA) while the Windows store has it.
# truststore routes Python SSL through the OS trust store so HF downloads work.
try:
    import truststore as _ts
    _ts.inject_into_ssl()
except Exception:
    pass

# The speechbrain ECAPA model is already cached from the URTIC extraction; point
# at it so validation extraction is fully offline for the model itself.
ECAPA_SAVEDIR = str(ROOT / "model" / "cache" / "speechbrain" / "spkrec-ecapa-voxceleb")

# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def recovery_metrics(true_labels: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """All the ground-truth recovery metrics for one clustering."""
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 homogeneity_completeness_v_measure)
    n = len(true_labels)
    ari = float(adjusted_rand_score(true_labels, cluster_labels))
    nmi = float(normalized_mutual_info_score(true_labels, cluster_labels))
    hom, com, vms = homogeneity_completeness_v_measure(true_labels, cluster_labels)

    # PURITY: sum over clusters of (majority true-speaker count) / N.
    # This is the property the grouping needs: high => each cluster is ~1 speaker.
    clusters = np.unique(cluster_labels)
    maj = 0
    cluster_sizes = []
    for c in clusters:
        m = cluster_labels == c
        _, cnt = np.unique(true_labels[m], return_counts=True)
        maj += int(cnt.max())
        cluster_sizes.append(int(m.sum()))
    purity = maj / n

    # FRAGMENTATION: for each true speaker, how many distinct clusters its
    # utterances span. 1.0 = perfect (speaker stays in one cluster). This is the
    # exact failure we measured on URTIC devel (speakers spread across clusters).
    speakers = np.unique(true_labels)
    frag = []
    for s in speakers:
        frag.append(int(np.unique(cluster_labels[true_labels == s]).size))
    frag = np.array(frag)

    return {
        "n_items": int(n),
        "n_true_speakers": int(len(speakers)),
        "n_clusters": int(len(clusters)),
        "ARI": ari, "NMI": nmi,
        "homogeneity": float(hom), "completeness": float(com), "v_measure": float(vms),
        "purity": float(purity),
        "fragmentation_mean": float(frag.mean()),
        "fragmentation_median": float(np.median(frag)),
        "fragmentation_max": int(frag.max()),
        "frac_speakers_single_cluster": float((frag == 1).mean()),
        "cluster_size_min": int(min(cluster_sizes)),
        "cluster_size_median": int(np.median(cluster_sizes)),
        "cluster_size_max": int(max(cluster_sizes)),
    }


def cluster_and_score(X: np.ndarray, true_labels: np.ndarray, k: int, seed: int = 42) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    Xn = normalize(X, axis=1)                        # matches cluster.py
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)  # matches cluster.py
    lab = km.fit_predict(Xn)
    return recovery_metrics(true_labels, lab)


# --------------------------------------------------------------------------- #
# corpus adapters -> list of (audio_path, speaker_id)
# --------------------------------------------------------------------------- #
def manifest_from_tsv(tsv: Path) -> list[tuple[Path, str]]:
    out = []
    for line in Path(tsv).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        out.append((Path(parts[0]), parts[1]))
    return out


def manifest_librispeech(root: Path) -> list[tuple[Path, str]]:
    # LibriSpeech: <root>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
    root = Path(root)
    out = []
    for p in root.rglob("*.flac"):
        spk = p.name.split("-")[0]
        out.append((p, spk))
    return out


def manifest_commonvoice(root: Path) -> list[tuple[Path, str]]:
    # Common Voice: <root>/clips/*.mp3 + <root>/validated.tsv with path,client_id.
    import csv as _csv
    root = Path(root)
    tsv = root / "validated.tsv"
    clips = root / "clips"
    out = []
    with tsv.open(encoding="utf-8") as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            out.append((clips / r["path"], r["client_id"]))
    return out


def manifest_svd(root: Path) -> list[tuple[Path, str]]:
    # Saarbruecken Voice Database export: speaker id is the numeric session id,
    # typically the leading digits of the filename (e.g. 1-a_n.wav -> speaker 1).
    root = Path(root)
    out = []
    for p in list(root.rglob("*.wav")) + list(root.rglob("*.nsp")):
        spk = p.stem.split("-")[0].split("_")[0]
        out.append((p, spk))
    return out


# Tuda-De multi-mic channels, cleanest first. IMPORTANT: use ONE channel only,
# else KMeans clusters partly on microphone identity, not speaker (workflow flag).
TUDA_CHANNELS = ["Samson", "Kinect-Beam", "Yamaha", "Kinect-RAW", "Realtek"]


def manifest_tuda(root: Path, channel: str = "Samson") -> list[tuple[Path, str]]:
    """Tuda-De German Distant Speech Corpus. Each recording = a timestamp-prefixed
    XML (speaker id/gender/age) + one WAV per microphone channel. We keep ONE
    clean channel so recovery measures speaker identity, not mic identity."""
    import xml.etree.ElementTree as ET
    root = Path(root)
    out = []
    missing_spk = 0
    for xml in root.rglob("*.xml"):
        try:
            r = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        spk = None
        for el in r.iter():
            if el.tag.lower() in ("speaker_id", "speakerid", "speaker") and (el.text or "").strip():
                spk = el.text.strip(); break
        if spk is None:
            missing_spk += 1
            continue
        if channel.upper() == "ALL":
            # keep EVERY mic file under the same speaker: run this to test whether
            # ECAPA recovery survives within-speaker recording-condition variation
            # (fragmentation_mean -> ~1 = condition-robust; -> ~5 = mic-confounded).
            for c in TUDA_CHANNELS:
                w = xml.with_name(f"{xml.stem}_{c}.wav")
                if w.exists():
                    out.append((w, spk))
            continue
        wav = xml.with_name(f"{xml.stem}_{channel}.wav")
        if not wav.exists():
            for c in TUDA_CHANNELS:
                alt = xml.with_name(f"{xml.stem}_{c}.wav")
                if alt.exists():
                    wav = alt; break
        if wav.exists():
            out.append((wav, spk))
    if missing_spk:
        print(f"[tuda][warn] {missing_spk} XML files had no recognizable speaker tag; "
              f"if this is most of them, paste one XML so I can fix the tag name.")
    return out


ADAPTERS = {
    "librispeech": manifest_librispeech,
    "commonvoice": manifest_commonvoice,
    "svd": manifest_svd,
    "tuda": manifest_tuda,
}


def manifest_from_hf(hf_id: str, split: str, out_wav_dir: Path, config: str | None = None,
                     audio_col: str | None = None, speaker_col: str | None = None,
                     speaker_from_path: bool = False, max_speakers: int | None = None,
                     max_per_speaker: int | None = None, scan_limit: int | None = None
                     ) -> list[tuple[Path, str]]:
    """STREAM a HuggingFace audio dataset (e.g. flozi00/multilingual-librispeech-
    german-labeled or facebook/multilingual_librispeech config 'german') and
    materialise a bounded, speaker-balanced subset to 16 kHz wavs + a (path,
    speaker) manifest. Decodes with soundfile (Audio(decode=False)) to avoid the
    torchcodec dependency. Speaker = an id column, else the MLS/LibriSpeech
    filename convention {speaker}_{book}_{utt} / {speaker}-{chap}-{utt}."""
    import io
    from collections import Counter
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        raise SystemExit("HuggingFace 'datasets' not installed. Run: pip install datasets")
    import soundfile as sf

    ds = (load_dataset(hf_id, config, split=split, streaming=True) if config
          else load_dataset(hf_id, split=split, streaming=True))
    feats = ds.features
    if audio_col is None:
        audio_col = next((k for k, v in feats.items() if type(v).__name__ == "Audio"), "audio")
    ds = ds.cast_column(audio_col, Audio(decode=False))
    if speaker_col is None and not speaker_from_path:
        speaker_col = next((c for c in ("speaker_id", "speaker", "reader_id", "reader",
                                        "client_id", "original_speaker_id", "spk_id")
                            if c in feats), None)
        if speaker_col is None:
            speaker_from_path = True
    print(f"[hf] {hf_id} split={split} audio_col={audio_col!r} "
          f"speaker={'<from filename>' if speaker_from_path else repr(speaker_col)} "
          f"caps: max_speakers={max_speakers} max_per_speaker={max_per_speaker}")

    out_wav_dir.mkdir(parents=True, exist_ok=True)
    per: Counter = Counter()
    seen: set = set()
    manifest = []
    scanned = 0
    for ex in ds:
        scanned += 1
        if scan_limit and scanned > scan_limit:
            break
        a = ex[audio_col]
        path = (a.get("path") if isinstance(a, dict) else None) or f"{scanned}.flac"
        stem = Path(path).stem
        spk = stem.split("_")[0].split("-")[0] if speaker_from_path else str(ex[speaker_col])
        if spk not in seen and max_speakers and len(seen) >= max_speakers:
            continue   # speaker pool full; keep scanning for more utts of known speakers
        if max_per_speaker and per[spk] >= max_per_speaker:
            continue
        seen.add(spk); per[spk] += 1
        wav = out_wav_dir / f"{spk}__{stem}.wav"
        if not wav.exists():
            data, sr = sf.read(io.BytesIO(a["bytes"]))
            sf.write(str(wav), data, sr)
        manifest.append((wav, spk))
    print(f"[hf] collected {len(manifest)} utts from {len(seen)} speakers "
          f"({scanned} rows scanned)")
    return manifest


# --------------------------------------------------------------------------- #
# extraction (identical ECAPA encoder + encode_batch)
# --------------------------------------------------------------------------- #
def extract_embeddings(manifest, cache_dir: Path, device: str, batch_size: int = 16,
                       max_seconds: float = 30.0) -> None:
    """Cache one [192] fp16 ECAPA embedding per audio file. Uses the SAME encoder
    (speechbrain/spkrec-ecapa-voxceleb) and the SAME encode_batch call as
    model/speakers/ecapa.extract_ecapa; only the audio decode is librosa (uniform
    flac/mp3/wav -> 16 kHz mono float)."""
    import torch
    import soundfile as sf
    from torch.utils.data import DataLoader, Dataset
    from speakers.ecapa import load_ecapa_encoder, TARGET_SR

    def _load16k(path):
        """wav/flac -> 16 kHz mono float32 via soundfile (+ scipy resample if
        needed). Avoids librosa's resampler (broken lazy backend in this env)."""
        a, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(a, "ndim", 1) > 1:
            a = a.mean(axis=1)
        a = np.asarray(a, dtype=np.float32)
        if sr != TARGET_SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), TARGET_SR)
            a = resample_poly(a, TARGET_SR // g, int(sr) // g).astype(np.float32)
        return a

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [(p, s) for (p, s) in manifest
            if not (cache_dir / f"{_key(p)}.pt").exists()]
    print(f"[extract] {len(manifest)} files; {len(manifest)-len(todo)} cached; {len(todo)} to encode")
    if not todo:
        return

    max_samples = int(max_seconds * TARGET_SR)

    class _DS(Dataset):
        def __init__(s, items): s.items = items
        def __len__(s): return len(s.items)
        def __getitem__(s, i):
            p, _ = s.items[i]
            a = _load16k(p)                                          # 16 kHz mono float32
            rl = len(a)
            if rl >= max_samples:
                a = a[:max_samples]; rel = 1.0
            else:
                a = np.pad(a, (0, max_samples - rl)); rel = rl / max_samples
            return {"audio": torch.from_numpy(a), "rel": float(rel), "key": _key(p)}

    def _coll(b):
        return {"audio": torch.stack([x["audio"] for x in b]),
                "rel": torch.tensor([x["rel"] for x in b], dtype=torch.float32),
                "keys": [x["key"] for x in b]}

    enc = load_ecapa_encoder(device=device, savedir=ECAPA_SAVEDIR)
    loader = DataLoader(_DS(todo), batch_size=batch_size, shuffle=False, collate_fn=_coll, num_workers=0)
    try:
        from tqdm.auto import tqdm
        loader = tqdm(loader, desc="ecapa")
    except ImportError:
        pass
    with torch.no_grad():
        for batch in loader:
            emb = enc.encode_batch(batch["audio"].to(device), batch["rel"].to(device))  # [B,1,192]
            emb = emb.squeeze(1).to(torch.float16).cpu()
            for i, k in enumerate(batch["keys"]):
                torch.save(emb[i].clone(), cache_dir / f"{k}.pt")


def _key(p: Path) -> str:
    # unique, filesystem-safe key per file (handles duplicate stems across dirs)
    p = Path(p)
    return (p.parent.name + "__" + p.stem).replace("/", "_").replace("\\", "_")


def load_matrix(manifest, cache_dir: Path):
    import torch
    cache_dir = Path(cache_dir)
    X, labels = [], []
    for p, s in manifest:
        f = cache_dir / f"{_key(p)}.pt"
        if not f.exists():
            continue
        X.append(torch.load(f, weights_only=True, map_location="cpu").to(torch.float32).numpy())
        labels.append(s)
    return np.stack(X), np.array(labels)


# --------------------------------------------------------------------------- #
def run(name: str, manifest, cache_dir: Path, device: str, out_json: Path,
        max_per_speaker: int | None, min_utts: int = 1) -> None:
    if max_per_speaker:
        by = {}
        for p, s in manifest:
            by.setdefault(s, []).append((p, s))
        manifest = [it for s in by for it in by[s][:max_per_speaker]]
        print(f"[cap] {max_per_speaker}/speaker -> {len(manifest)} files")

    extract_embeddings(manifest, cache_dir, device)
    X, labels = load_matrix(manifest, cache_dir)

    if min_utts > 1:
        # drop speakers with too few utterances (can't form a meaningful cluster)
        from collections import Counter
        cnt = Counter(labels.tolist())
        keep = np.array([cnt[s] >= min_utts for s in labels])
        X, labels = X[keep], labels[keep]
        print(f"[filter] min {min_utts} utts/speaker -> {X.shape[0]} embeddings, "
              f"{len(np.unique(labels))} speakers")
    n_true = len(np.unique(labels))
    print(f"[data] {X.shape[0]} embeddings, {n_true} true speakers")

    # k-misspecification sweep around the true count
    ks = sorted({max(2, int(round(f * n_true))) for f in (0.5, 1.0, 2.0)} | {n_true})
    results = {}
    for k in ks:
        m = cluster_and_score(X, labels, k)
        tag = "true_k" if k == n_true else f"{k/n_true:.2g}x_k"
        results[f"k{k}_{tag}"] = m
        print(f"  k={k:<5} ({tag:<7}) purity={m['purity']:.3f} ARI={m['ARI']:.3f} "
              f"NMI={m['NMI']:.3f} frag_mean={m['fragmentation_mean']:.2f} "
              f"single-cluster-spk={m['frac_speakers_single_cluster']:.2f}")

    # negative control: shuffle the true labels, cluster stays, ARI should ~ 0
    rng = np.random.default_rng(0)
    shuf = labels.copy(); rng.shuffle(shuf)
    neg = cluster_and_score(X, shuf, n_true)
    print(f"  [neg control] shuffled-label ARI={neg['ARI']:.4f} NMI={neg['NMI']:.4f} (expect ~0)")

    out = {
        "rung_id": "ecapa_recovery_validation",
        "corpus": name,
        "n_embeddings": int(X.shape[0]),
        "n_true_speakers": int(n_true),
        "pipeline": "speechbrain/spkrec-ecapa-voxceleb -> L2 -> KMeans(n_init=10, seed=42), identical to URTIC",
        "k_sweep": results,
        "negative_control_shuffled_labels": neg,
        "reading": ("purity/ARI/NMI high AND fragmentation_mean near 1 => ECAPA+kmeans "
                    "recovers real speakers, so URTIC pseudo-speakers are a trustworthy "
                    "grouping. Robustness across 0.5x/1x/2x k => the k=210 URTIC choice is "
                    "defensible despite unknown true count. Weak recovery => the grouping "
                    "only partially prevents leakage (report honestly)."),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {out_json}")


def smoke() -> None:
    """Metrics sanity check on synthetic well-separated speaker blobs (no model)."""
    rng = np.random.default_rng(0)
    n_spk, per = 50, 30
    centers = rng.standard_normal((n_spk, 192)) * 5
    X = np.repeat(centers, per, axis=0) + rng.standard_normal((n_spk * per, 192)) * 0.3
    labels = np.repeat(np.arange(n_spk), per)
    m = cluster_and_score(X, labels, n_spk)
    print("[smoke] well-separated blobs @ true k:")
    print(f"  purity={m['purity']:.3f} ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} "
          f"frag_mean={m['fragmentation_mean']:.2f} single-cluster-spk={m['frac_speakers_single_cluster']:.2f}")
    assert m["purity"] > 0.98 and m["ARI"] > 0.98 and m["fragmentation_mean"] < 1.05, m
    # negative control on the same data
    shuf = labels.copy(); rng.shuffle(shuf)
    neg = cluster_and_score(X, shuf, n_spk)
    print(f"  [neg] shuffled-label ARI={neg['ARI']:.4f} (expect ~0)")
    assert abs(neg["ARI"]) < 0.05, neg["ARI"]
    # under-clustering (0.5x): purity should drop, fragmentation stay ~1
    half = cluster_and_score(X, labels, n_spk // 2)
    print(f"  [0.5x k] purity={half['purity']:.3f} (merges speakers) "
          f"frag_mean={half['fragmentation_mean']:.2f}")
    print("SMOKE PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corpus", choices=list(ADAPTERS))
    ap.add_argument("--root")
    ap.add_argument("--manifest")
    ap.add_argument("--hf", help="HuggingFace dataset id, e.g. facebook/multilingual_librispeech")
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--hf-audio-col", default=None)
    ap.add_argument("--hf-speaker-col", default=None)
    ap.add_argument("--hf-speaker-from-path", action="store_true",
                    help="derive speaker from the MLS/LibriSpeech filename ({spk}_{book}_{utt})")
    ap.add_argument("--hf-max-speakers", type=int, default=None)
    ap.add_argument("--hf-max-per-speaker", type=int, default=None)
    ap.add_argument("--hf-scan-limit", type=int, default=6000,
                    help="max streamed rows to scan (bounds the download)")
    ap.add_argument("--min-utts", type=int, default=1,
                    help="drop speakers with fewer than this many utterances before clustering")
    ap.add_argument("--name", default="corpus")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-per-speaker", type=int, default=None)
    ap.add_argument("--channel", default="Samson", help="tuda: which mic channel (use ONE)")
    a = ap.parse_args()

    if a.smoke:
        smoke(); return

    if a.manifest:
        man = manifest_from_tsv(Path(a.manifest))
    elif a.hf:
        man = manifest_from_hf(a.hf, a.hf_split, ROOT / "cache" / "ecapa_validation" / a.name / "wav",
                               config=a.hf_config, audio_col=a.hf_audio_col, speaker_col=a.hf_speaker_col,
                               speaker_from_path=a.hf_speaker_from_path,
                               max_speakers=a.hf_max_speakers, max_per_speaker=a.hf_max_per_speaker,
                               scan_limit=a.hf_scan_limit)
    elif a.corpus == "tuda" and a.root:
        man = manifest_tuda(Path(a.root), channel=a.channel)
    elif a.corpus and a.root:
        man = ADAPTERS[a.corpus](Path(a.root))
    else:
        ap.error("give --smoke, or --manifest, or --hf <id>, or (--corpus and --root)")

    if not man:
        ap.error("empty manifest -- check the path/adapter")
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / "cache" / "ecapa_validation" / a.name
    out = ROOT / "results" / f"ecapa_recovery_{a.name}.json"
    run(a.name, man, cache, device, out, a.max_per_speaker, min_utts=a.min_utts)


if __name__ == "__main__":
    main()
