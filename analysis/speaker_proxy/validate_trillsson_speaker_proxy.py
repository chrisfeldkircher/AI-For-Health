"""Independent TRILLsson validation of the URTIC pseudo-speaker proxy.

This script deliberately does not read the ComParE cold labels.  It first
validates Google TRILLsson1 on cached, labeled LibriSpeech/MLS control audio,
then compares side-local TRILLsson clusters with the existing ECAPA clusters.

TRILLsson1 is a general paralinguistic representation, not a dedicated speaker
verification model.  Its role here is an independent *view* of the audio: high
known-speaker recovery and non-trivial agreement with ECAPA would support the
speaker-proxy hypothesis; weak recovery means it should not be fused.

Examples (run in the isolated TensorFlow environment):
  python validate_trillsson_speaker_proxy.py --control libri_en_dev
  python validate_trillsson_speaker_proxy.py --control mls_de
  python validate_trillsson_speaker_proxy.py --urtic-side train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Keep TensorFlow's startup chatter out of benchmark logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if int(os.environ.get("OPENBLAS_NUM_THREADS", "1")) > 24:
    os.environ["OPENBLAS_NUM_THREADS"] = "16"

import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "cache").is_dir())
def default_model_path(variant: int) -> Path:
    return (
        Path.home()
        / f".cache/kagglehub/models/google/trillsson/tensorFlow2/{variant}/1"
    )


WAV_ROOT = ROOT / "dataset/ComParE2017_Cold_4students/wav"
ECAPA_LABELS = ROOT / "results/speaker_proxy_method_labels.npz"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 32_000  # 2 s, matching the released TRILLsson protocol.
K_URTIC = 210
KMEANS_SEEDS = (42, 1, 2)


def audio_windows(path: Path, max_windows: int) -> np.ndarray:
    """Load mono audio and return deterministic, evenly spaced 2-second windows."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    wav = np.nan_to_num(wav, copy=False)
    if len(wav) <= WINDOW_SAMPLES:
        return np.pad(wav, (0, WINDOW_SAMPLES - len(wav)))[None].astype(np.float32)
    n = min(max_windows, int(np.ceil(len(wav) / WINDOW_SAMPLES)))
    starts = np.rint(np.linspace(0, len(wav) - WINDOW_SAMPLES, n)).astype(int)
    return np.stack([wav[s : s + WINDOW_SAMPLES] for s in starts]).astype(np.float32)


def load_model(path: Path):
    if not (path / "saved_model.pb").exists():
        raise SystemExit(
            f"TRILLsson SavedModel not found at {path}. Download the official "
            "Google/Kaggle instance google/trillsson/tensorFlow2/1/1 first."
        )
    module = tf.saved_model.load(str(path))
    return module.signatures["serving_default"]


def extract_embeddings(
    paths: list[Path], model_path: Path, max_windows: int, batch_size: int
) -> tuple[np.ndarray, dict]:
    infer = load_model(model_path)
    output = np.empty((len(paths), 1024), dtype=np.float32)
    started = time.time()
    n_windows_total = 0

    # Group a modest number of files at a time. All windows have the same length,
    # so inference can be batched without padding-dependent representations.
    file_block = max(32, batch_size * 2)
    for block_start in range(0, len(paths), file_block):
        block_paths = paths[block_start : block_start + file_block]
        windows: list[np.ndarray] = []
        owners: list[int] = []
        for local_i, path in enumerate(block_paths):
            chunks = audio_windows(path, max_windows=max_windows)
            windows.extend(chunks)
            owners.extend([local_i] * len(chunks))
        x = np.stack(windows)
        z_parts = []
        for start in range(0, len(x), batch_size):
            result = infer(audio_samples=tf.convert_to_tensor(x[start : start + batch_size]))
            # Output names differ across released variants (for example "dense"
            # in v1 and "tf.math.reduce_mean" in v5); each has one 1024-d tensor.
            z_parts.append(next(iter(result.values())).numpy().astype(np.float32))
        z = np.concatenate(z_parts, axis=0)
        owners_arr = np.asarray(owners)
        for local_i in range(len(block_paths)):
            # Average segment representations, then L2-normalize per recording.
            row = z[owners_arr == local_i].mean(axis=0)
            norm = float(np.linalg.norm(row))
            output[block_start + local_i] = row / max(norm, 1e-12)
        n_windows_total += len(x)
        done = min(block_start + file_block, len(paths))
        if done == len(paths) or done % 512 < file_block:
            elapsed = time.time() - started
            print(
                f"[extract] {done}/{len(paths)} files, {n_windows_total} windows, "
                f"{elapsed:.1f}s ({done / max(elapsed, 1e-9):.1f} files/s)",
                flush=True,
            )
    return output, {
        "elapsed_seconds": time.time() - started,
        "n_files": len(paths),
        "n_windows": n_windows_total,
        "files_per_second": len(paths) / max(time.time() - started, 1e-9),
    }


def cache_key(name: str, max_windows: int, variant: int) -> Path:
    return CACHE_DIR / f"trillsson{variant}" / f"{name}_w{max_windows}.npz"


def get_embeddings(
    name: str,
    paths: list[Path],
    model_path: Path,
    max_windows: int,
    batch_size: int,
    force: bool,
    variant: int,
) -> tuple[np.ndarray, dict]:
    target = cache_key(name, max_windows, variant)
    target.parent.mkdir(parents=True, exist_ok=True)
    stems = np.asarray([p.stem for p in paths])
    if target.exists() and not force:
        saved = np.load(target, allow_pickle=False)
        if np.array_equal(saved["stems"].astype(str), stems.astype(str)):
            print(f"[cache] {target.relative_to(ROOT)}", flush=True)
            return saved["embeddings"].astype(np.float32), json.loads(str(saved["metadata"]))
        print(f"[cache] stem mismatch; rebuilding {target.relative_to(ROOT)}", flush=True)
    embeddings, timing = extract_embeddings(paths, model_path, max_windows, batch_size)
    metadata = {
        "model": f"google/trillsson/tensorFlow2/{variant}/1",
        "model_path": str(model_path),
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE,
        "max_windows_per_file": max_windows,
        "segment_pooling": "mean then L2 normalize",
        "timing": timing,
    }
    np.savez_compressed(
        target,
        stems=stems,
        embeddings=embeddings,
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"[wrote] {target.relative_to(ROOT)}", flush=True)
    return embeddings, metadata


def recovery_metrics(true_labels: np.ndarray, cluster_labels: np.ndarray) -> dict:
    n = len(true_labels)
    majority = 0
    for cluster in np.unique(cluster_labels):
        _, counts = np.unique(true_labels[cluster_labels == cluster], return_counts=True)
        majority += int(counts.max())
    fragmentation = np.asarray(
        [len(np.unique(cluster_labels[true_labels == s])) for s in np.unique(true_labels)]
    )
    return {
        "n_items": n,
        "n_true_speakers": int(len(np.unique(true_labels))),
        "n_clusters": int(len(np.unique(cluster_labels))),
        "ARI": float(adjusted_rand_score(true_labels, cluster_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, cluster_labels)),
        "purity": float(majority / n),
        "fragmentation_mean": float(fragmentation.mean()),
        "fragmentation_median": float(np.median(fragmentation)),
        "fragmentation_max": int(fragmentation.max()),
        "frac_speakers_single_cluster": float(np.mean(fragmentation == 1)),
    }


def stable_kmeans(x: np.ndarray, k: int) -> tuple[dict[str, np.ndarray], dict]:
    labels = {}
    for seed in KMEANS_SEEDS:
        print(f"[cluster] KMeans k={k}, seed={seed}", flush=True)
        labels[str(seed)] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
    pairs = []
    keys = list(labels)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            pairs.append(
                {
                    "a": int(a),
                    "b": int(b),
                    "ARI": float(adjusted_rand_score(labels[a], labels[b])),
                    "NMI": float(normalized_mutual_info_score(labels[a], labels[b])),
                }
            )
    return labels, {
        "pairs": pairs,
        "ARI_mean": float(np.mean([p["ARI"] for p in pairs])),
        "ARI_min": float(np.min([p["ARI"] for p in pairs])),
        "NMI_mean": float(np.mean([p["NMI"] for p in pairs])),
    }


def run_control(args) -> None:
    wav_dir = ROOT / "cache/ecapa_validation" / args.control / "wav"
    paths = sorted(wav_dir.glob("*.wav"))
    if args.max_files:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"No control WAV files under {wav_dir}")
    true = np.asarray([p.stem.split("__", 1)[0] for p in paths])
    x, metadata = get_embeddings(
        args.control, paths, args.model_path, args.max_windows, args.batch_size, args.force,
        args.variant,
    )
    x = normalize(x).astype(np.float32)
    k = len(np.unique(true))
    cluster_labels, stability = stable_kmeans(x, k)
    metrics = recovery_metrics(true, cluster_labels["42"])

    # A shuffled-label negative control catches metric or alignment mistakes.
    shuffled = np.random.default_rng(42).permutation(true)
    negative = recovery_metrics(shuffled, cluster_labels["42"])
    report = {
        "question": f"Does TRILLsson{args.variant} recover known speakers on an external labeled corpus?",
        "corpus": args.control,
        "pipeline": (
            f"TRILLsson{args.variant} deterministic 2-second windows -> mean -> "
            "L2 -> KMeans(true k)"
        ),
        "metadata": metadata,
        "known_speaker_recovery": metrics,
        "kmeans_seed_stability": stability,
        "shuffled_label_negative_control": negative,
    }
    out = RESULTS_DIR / f"trillsson{args.variant}_recovery_{args.control}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[wrote] {out.relative_to(ROOT)}", flush=True)


def aligned_ecapa_labels(side: str, stems: np.ndarray) -> dict[str, np.ndarray]:
    saved = np.load(ECAPA_LABELS, allow_pickle=True)
    ecapa_stems = saved[f"{side}__stems"].astype(str)
    if np.array_equal(ecapa_stems, stems.astype(str)):
        order = np.arange(len(stems))
    else:
        lookup = {stem: i for i, stem in enumerate(ecapa_stems)}
        missing = [stem for stem in stems if stem not in lookup]
        if missing:
            raise RuntimeError(f"{len(missing)} stems absent from ECAPA labels; first={missing[0]}")
        order = np.asarray([lookup[stem] for stem in stems])
    return {
        name: saved[f"{side}__{name}"][order].astype(np.int32)
        for name in ("kmeans", "agglomerative", "spectral")
    }


def cross_view_neighbour_score(
    source_x: np.ndarray, target_labels: np.ndarray, k: int = 10
) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1).fit(source_x)
    idx = nn.kneighbors(source_x, return_distance=False)[:, 1:]
    return float(np.mean(target_labels[idx] == target_labels[:, None]))


def run_urtic(args) -> None:
    side = args.urtic_side
    paths = sorted(WAV_ROOT.glob(f"{side}_*.wav"))
    if args.max_files:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"No {side} WAV files under {WAV_ROOT}")
    stems = np.asarray([p.stem for p in paths])
    x, metadata = get_embeddings(
        f"urtic_{side}", paths, args.model_path, args.max_windows, args.batch_size, args.force,
        args.variant,
    )
    x = normalize(x).astype(np.float32)
    if len(paths) < K_URTIC:
        raise SystemExit("A URTIC cross-view run needs at least 210 files")
    trill_labels, stability = stable_kmeans(x, K_URTIC)
    reference = trill_labels["42"]
    ecapa = aligned_ecapa_labels(side, stems)
    agreements = {
        name: {
            "ARI": float(adjusted_rand_score(reference, values)),
            "NMI": float(normalized_mutual_info_score(reference, values)),
        }
        for name, values in ecapa.items()
    }
    ecapa_archive = np.load(ROOT / "cache/ecapa-voxceleb/ecapa_embeddings.npz", allow_pickle=True)
    all_stems = ecapa_archive["stems"].astype(str)
    lookup = {stem: i for i, stem in enumerate(all_stems)}
    ecapa_x = normalize(
        ecapa_archive["embeddings"][[lookup[s] for s in stems]].astype(np.float32)
    ).astype(np.float32)
    retrieval = {
        "ecapa_partition_cohesion_in_trillsson_knn10": {
            name: cross_view_neighbour_score(x, values) for name, values in ecapa.items()
        },
        "trillsson_partition_cohesion_in_ecapa_knn10": cross_view_neighbour_score(
            ecapa_x, reference
        ),
    }
    report = {
        "question": "Does an independent TRILLsson view support side-local ECAPA speaker proxies?",
        "side": side,
        "n_chunks": len(paths),
        "cold_labels_loaded": False,
        "known_speaker_count_prior": K_URTIC,
        "metadata": metadata,
        "trillsson_kmeans_seed_stability": stability,
        "agreement_vs_trillsson_seed42": agreements,
        "cross_view_retrieval": retrieval,
        "embedding_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "limitations": [
            "URTIC speaker IDs are unavailable, so cross-view agreement is not accuracy.",
            f"TRILLsson{args.variant} is a general paralinguistic model, not a dedicated speaker verifier.",
            "Cold labels are excluded and cannot be used to select or tune this partition.",
        ],
    }
    out = RESULTS_DIR / f"trillsson{args.variant}_ecapa_cross_view_{side}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    labels_out = RESULTS_DIR / f"trillsson{args.variant}_labels_{side}.npz"
    np.savez_compressed(labels_out, stems=stems, **{f"seed{k}": v for k, v in trill_labels.items()})
    print(json.dumps(report, indent=2), flush=True)
    print(f"[wrote] {out.relative_to(ROOT)} and {labels_out.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--control", choices=("libri_en_dev", "mls_de", "mls_de_small"))
    target.add_argument("--urtic-side", choices=("train", "devel", "test"))
    parser.add_argument("--variant", type=int, choices=(1, 2, 3, 4, 5), default=1)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=None, help="smoke-test prefix only")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = default_model_path(args.variant)
    if args.max_windows < 1 or args.batch_size < 1:
        parser.error("--max-windows and --batch-size must be positive")
    if args.control:
        run_control(args)
    else:
        run_urtic(args)


if __name__ == "__main__":
    main()
