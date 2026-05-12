"""Build the EDA (exploratory data analysis) bundle for the paper.

Consolidates the URTIC dataset's per-chunk embeddings + pseudo-speaker
assignments + per-chunk metadata into a compact, plot-ready bundle so
UMAP / scatter / cluster-quality figures can be produced on a separate
machine without copying the multi-GB feature caches.

Outputs (under paper_data/eda/):
    chunk_metadata.csv               — per-chunk: stem, split, cold_label, k100/k210/k420 pseudo-speakers
    ecapa_embeddings_fp16.npy        — (N, 192) ECAPA-TDNN speaker embeddings (~7 MB)
    wavlm_a25_pca128_fp16.npy        — (N, 128) PCA-128 of the WavLM-A2.5 substrate
    wavlm_a25_pca_components_fp16.npy — (128, 4096) PCA basis for re-projecting
    wavlm_a25_pca_variance.csv       — per-component explained-variance ratio
    wavlm_a25_layer_weights.csv      — honesty-prior layer-weight softmax used (T_INV=50)
    pseudo_speaker_cluster_stats.csv — per-cluster: size, intra/inter cosine, cold rate
    umap_coords_ecapa_2d.csv         — (N, 2) UMAP coordinates on raw ECAPA (192-d)
    umap_coords_wavlm_a25_2d.csv     — (N, 2) UMAP coordinates on PCA-128 of A2.5 substrate
    README.md                        — describes how to plot from these on Mac

Runtime: ~5-15 minutes (dominated by sequential torch.load + UMAP).
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
import umap

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
DATASET_LAB = ROOT / "dataset" / "ComParE2017_Cold_4students" / "lab" / "ComParE2017_Cold.tsv"
PSEUDO_DIR = CACHE / "pseudo_speakers"
WAVLM_POOLED = CACHE / "microsoft_wavlm-large" / "pooled"
ECAPA_DIR = CACHE / "ecapa-voxceleb"

OUT = ROOT / "paper_data" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Metadata: stem, split, label, pseudo-speakers
# ---------------------------------------------------------------------------
def build_metadata() -> pd.DataFrame:
    print("[1/7] building chunk_metadata.csv ...")
    # Labels
    labels = pd.read_csv(DATASET_LAB, sep="\t")
    labels.columns = ["file_name", "cold"]
    labels["file_stem"] = labels["file_name"].str.replace(".wav", "", regex=False)
    labels["split"] = labels["file_stem"].str.split("_").str[0]
    labels["cold_label"] = (labels["cold"] == "C").astype(int)
    labels = labels.drop(columns=["file_name", "cold"])

    # Pseudo-speaker assignments
    for k in [100, 210, 420]:
        ps = pd.read_csv(PSEUDO_DIR / f"k{k}_seed42.tsv", sep="\t")
        ps = ps.rename(columns={"cluster": f"k{k}_cluster"})
        labels = labels.merge(ps[["file_stem", f"k{k}_cluster"]], on="file_stem", how="left")

    # Sort by (split, file_stem) for deterministic row order — this is the row order
    # used for all .npy embedding matrices below.
    labels = labels.sort_values(["split", "file_stem"]).reset_index(drop=True)

    out_path = OUT / "chunk_metadata.csv"
    labels.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(ROOT)}  ({len(labels)} rows)")
    return labels


# ---------------------------------------------------------------------------
# 2. ECAPA embeddings (192-d): concat -> fp16 npy
# ---------------------------------------------------------------------------
def build_ecapa(meta: pd.DataFrame) -> np.ndarray:
    print("[2/7] building ecapa_embeddings_fp16.npy ...")
    N = len(meta)
    arr = np.empty((N, 192), dtype=np.float16)
    t0 = time.time()
    for i, stem in enumerate(meta["file_stem"]):
        p = ECAPA_DIR / f"{stem}.pt"
        e = torch.load(p, map_location="cpu", weights_only=False)
        arr[i] = e.cpu().numpy().astype(np.float16)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{N}  ({time.time()-t0:.1f}s)")
    out_path = OUT / "ecapa_embeddings_fp16.npy"
    np.save(out_path, arr)
    print(f"  wrote {out_path.relative_to(ROOT)}  shape={arr.shape}  size={arr.nbytes/1e6:.1f} MB")
    return arr


# ---------------------------------------------------------------------------
# 3. WavLM-A2.5 substrate: pooled (25,4096) @ softmax(T_INV * sub@1) -> (4096,)
# ---------------------------------------------------------------------------
def build_wavlm_a25(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    print("[3/7] building WavLM-A2.5 substrate + honesty-prior layer weights ...")
    audit = pd.read_csv(RESULTS / "A5d_layer_honesty.csv")
    sub = audit["subtractive_honesty_lam1"].to_numpy(dtype=np.float32)
    sub = np.clip(sub, 0.0, None)  # subtractive honesty floored at 0
    T_INV = 50.0
    logits = T_INV * sub
    logits = logits - logits.max()  # numeric stability
    layer_w = np.exp(logits) / np.exp(logits).sum()
    layer_w = layer_w.astype(np.float32)
    print(f"  T_INV=50; layer-weight shape {layer_w.shape}; max={layer_w.max():.4f}  argmax=layer{int(np.argmax(layer_w))}")

    # Save layer weights vector
    w_df = pd.DataFrame({"layer": np.arange(len(layer_w)), "sub_at_1": sub, "honesty_prior_weight": layer_w})
    w_df.to_csv(OUT / "wavlm_a25_layer_weights.csv", index=False)
    print(f"  wrote paper_data/eda/wavlm_a25_layer_weights.csv  ({len(layer_w)} layers)")

    # Per-chunk weighted sum
    N = len(meta)
    arr = np.empty((N, 4096), dtype=np.float32)
    t0 = time.time()
    for i, stem in enumerate(meta["file_stem"]):
        p = WAVLM_POOLED / f"{stem}.pt"
        pooled = torch.load(p, map_location="cpu", weights_only=False)  # (25, 4096) fp16
        pooled_f32 = pooled.cpu().numpy().astype(np.float32)
        arr[i] = layer_w @ pooled_f32  # (4096,)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{N}  ({time.time()-t0:.1f}s)")
    print(f"  full substrate shape={arr.shape} bytes={arr.nbytes/1e6:.1f} MB (fp32, in-memory)")
    return arr, layer_w


# ---------------------------------------------------------------------------
# 4. PCA-128 the substrate -> save fp16
# ---------------------------------------------------------------------------
def build_wavlm_pca(substrate: np.ndarray, n_components: int = 128) -> np.ndarray:
    print(f"[4/7] PCA-{n_components} on WavLM-A2.5 substrate ...")
    pca = PCA(n_components=n_components, random_state=42)
    pca_emb = pca.fit_transform(substrate)
    explained = pca.explained_variance_ratio_
    print(f"  PCA-{n_components} explained variance: {explained.sum()*100:.2f}% (first 16: {[round(float(x), 3) for x in explained[:16]]})")

    np.save(OUT / "wavlm_a25_pca128_fp16.npy", pca_emb.astype(np.float16))
    np.save(OUT / "wavlm_a25_pca_components_fp16.npy", pca.components_.astype(np.float16))
    pd.DataFrame({
        "component": np.arange(n_components),
        "explained_variance_ratio": explained,
        "cumulative_variance_ratio": np.cumsum(explained),
    }).to_csv(OUT / "wavlm_a25_pca_variance.csv", index=False)
    print(f"  wrote paper_data/eda/wavlm_a25_pca128_fp16.npy  shape={pca_emb.shape}")
    print(f"  wrote paper_data/eda/wavlm_a25_pca_components_fp16.npy  shape={pca.components_.shape}")
    print(f"  wrote paper_data/eda/wavlm_a25_pca_variance.csv")
    return pca_emb


# ---------------------------------------------------------------------------
# 5. UMAP on both substrates
# ---------------------------------------------------------------------------
def build_umap(ecapa: np.ndarray, wavlm_pca: np.ndarray, meta: pd.DataFrame) -> None:
    print("[5/7] UMAP-2D on ECAPA-192 ...")
    t0 = time.time()
    u_ecapa = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.10,
        metric="cosine", random_state=42,
    ).fit_transform(ecapa.astype(np.float32))
    print(f"  ECAPA UMAP done ({time.time()-t0:.1f}s)")

    print("[5/7] UMAP-2D on WavLM-A2.5 PCA-128 ...")
    t0 = time.time()
    u_wavlm = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.10,
        metric="cosine", random_state=42,
    ).fit_transform(wavlm_pca.astype(np.float32))
    print(f"  WavLM UMAP done ({time.time()-t0:.1f}s)")

    # Save with metadata join for easy plotting
    e_df = pd.DataFrame({
        "file_stem": meta["file_stem"].values,
        "split": meta["split"].values,
        "cold_label": meta["cold_label"].values,
        "k210_cluster": meta["k210_cluster"].values,
        "umap_x": u_ecapa[:, 0],
        "umap_y": u_ecapa[:, 1],
    })
    e_df.to_csv(OUT / "umap_coords_ecapa_2d.csv", index=False)
    print(f"  wrote paper_data/eda/umap_coords_ecapa_2d.csv  ({len(e_df)} rows)")

    w_df = pd.DataFrame({
        "file_stem": meta["file_stem"].values,
        "split": meta["split"].values,
        "cold_label": meta["cold_label"].values,
        "k210_cluster": meta["k210_cluster"].values,
        "umap_x": u_wavlm[:, 0],
        "umap_y": u_wavlm[:, 1],
    })
    w_df.to_csv(OUT / "umap_coords_wavlm_a25_2d.csv", index=False)
    print(f"  wrote paper_data/eda/umap_coords_wavlm_a25_2d.csv  ({len(w_df)} rows)")


# ---------------------------------------------------------------------------
# 6. Pseudo-speaker cluster quality: size + intra/inter cosine + cold rate
# ---------------------------------------------------------------------------
def build_cluster_stats(ecapa: np.ndarray, meta: pd.DataFrame, k: int = 210) -> None:
    print(f"[6/7] pseudo-speaker cluster stats (k={k}) ...")
    cl_col = f"k{k}_cluster"
    # Normalise ECAPA for cosine similarity
    norms = np.linalg.norm(ecapa.astype(np.float32), axis=1, keepdims=True) + 1e-12
    ecapa_n = (ecapa.astype(np.float32) / norms).astype(np.float32)

    # Cluster centroids (in normalised space)
    cluster_ids = sorted(meta[cl_col].unique())
    centroids = np.zeros((len(cluster_ids), ecapa.shape[1]), dtype=np.float32)
    for i, c in enumerate(cluster_ids):
        idx = (meta[cl_col].values == c)
        centroids[i] = ecapa_n[idx].mean(axis=0)
    centroids_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    # Pairwise centroid-centroid cosine -> nearest-other-centroid cos
    cent_sim = centroids_n @ centroids_n.T
    np.fill_diagonal(cent_sim, -np.inf)
    nearest_other_cos = cent_sim.max(axis=1)

    rows: list[list] = []
    for i, c in enumerate(cluster_ids):
        idx = (meta[cl_col].values == c)
        n = int(idx.sum())
        cold_rate = float(meta.loc[idx, "cold_label"].mean())
        # train/devel split share
        train_share = float((meta.loc[idx, "split"].values == "train").mean())
        # intra-cluster mean cosine = mean(centroid . embedding)
        intra_cos = float((ecapa_n[idx] @ centroids_n[i]).mean())
        rows.append([c, n, cold_rate, train_share, intra_cos, float(nearest_other_cos[i])])

    df = pd.DataFrame(rows, columns=[
        "cluster_id", "n_chunks", "cold_rate", "train_share",
        "intra_cluster_mean_cosine", "nearest_other_cluster_cosine",
    ])
    df["intra_minus_nearest_other"] = df["intra_cluster_mean_cosine"] - df["nearest_other_cluster_cosine"]
    df.to_csv(OUT / "pseudo_speaker_cluster_stats.csv", index=False)
    print(f"  wrote paper_data/eda/pseudo_speaker_cluster_stats.csv  ({len(df)} clusters)")
    print(f"  cluster size: median={int(df['n_chunks'].median())} min={int(df['n_chunks'].min())} max={int(df['n_chunks'].max())}")
    print(f"  intra cosine: mean={df['intra_cluster_mean_cosine'].mean():.3f}; nearest-other: mean={df['nearest_other_cluster_cosine'].mean():.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    print(f"Project root: {ROOT}")
    print(f"Writing to:   {OUT}")
    print()

    meta = build_metadata()
    ecapa = build_ecapa(meta)
    substrate, _layer_w = build_wavlm_a25(meta)
    wavlm_pca = build_wavlm_pca(substrate, n_components=128)
    build_umap(ecapa, wavlm_pca, meta)
    build_cluster_stats(ecapa, meta, k=210)

    print(f"\n[7/7] DONE in {(time.time()-t0)/60:.1f} min")
