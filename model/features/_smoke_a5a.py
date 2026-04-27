"""Smoke test for A5a: G1, G4, G8 extractors + honesty audit on a small subset."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from data.cached_dataset import PooledCacheDataset, load_labels, stratified_split
from features import extract_g1, extract_g4, extract_g8, G1_NAMES, G4_NAMES
from honesty import audit_group
from speakers import load_pseudo_speakers


DATA_DIR   = "../dataset/ComParE2017_Cold_4students"
CACHE_ROOT = "../cache"
WAV_DIR    = f"{DATA_DIR}/wav"
BACKBONE   = "microsoft_wavlm-large"


def _stems(files):
    return [f[:-4] if f.endswith(".wav") else f for f in files]


def main():
    full_train = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, split="train")
    full_devel = PooledCacheDataset(DATA_DIR, CACHE_ROOT, BACKBONE, split="devel")
    labels_map = load_labels(DATA_DIR)

    train_fit_files, _ = stratified_split(full_train.files, labels_map, val_frac=0.10, seed=42)
    devel_val_files, _ = stratified_split(full_devel.files, labels_map, val_frac=0.50, seed=42)

    pseudo = load_pseudo_speakers(Path(CACHE_ROOT) / "pseudo_speakers" / "k210_seed42.tsv")

    # Subsample for smoke
    train_sub = train_fit_files[:64]
    devel_sub = devel_val_files[:32]
    print(f"smoke: train={len(train_sub)}  devel={len(devel_sub)}")

    train_stems = _stems(train_sub)
    devel_stems = _stems(devel_sub)
    train_nc_stems = [s for s, fn in zip(train_stems, train_sub) if labels_map[fn] == 0]

    y_cold_tr = np.array([labels_map[f] for f in train_sub], dtype=np.int64)
    y_cold_de = np.array([labels_map[f] for f in devel_sub], dtype=np.int64)
    y_pseudo_tr = np.array([pseudo[s] for s in train_stems], dtype=np.int64)
    y_pseudo_de = np.array([pseudo[s] for s in devel_stems], dtype=np.int64)
    print(f"  cold rate train={y_cold_tr.mean():.3f}  devel={y_cold_de.mean():.3f}")

    # G1 — voicing scalars (free)
    print("\n--- G1 ---")
    Xtr_g1 = extract_g1(train_stems, CACHE_ROOT)
    Xde_g1 = extract_g1(devel_stems, CACHE_ROOT)
    print(f"G1 X_train={Xtr_g1.shape}  any_nan={np.isnan(Xtr_g1).any()}")
    print(f"G1 col 0 (voiced_frac):  mean={Xtr_g1[:,0].mean():.3f}  std={Xtr_g1[:,0].std():.3f}")
    print(f"G1 col 8 (long_silence): mean={Xtr_g1[:,8].mean():.3f}  std={Xtr_g1[:,8].std():.3f}")

    # G4 — energy/pause/breath
    print("\n--- G4 ---")
    Xtr_g4 = extract_g4(train_stems, CACHE_ROOT, WAV_DIR, progress=False)
    Xde_g4 = extract_g4(devel_stems, CACHE_ROOT, WAV_DIR, progress=False)
    print(f"G4 X_train={Xtr_g4.shape}  any_nan={np.isnan(Xtr_g4).any()}")
    print(f"G4 col 2 (rms_db_mean): mean={Xtr_g4[:,2].mean():.2f}  std={Xtr_g4[:,2].std():.2f}")
    print(f"G4 col 6 (v-s db):       mean={Xtr_g4[:,6].mean():.2f}  std={Xtr_g4[:,6].std():.2f}")

    # G8 — OOD
    print("\n--- G8 ---")
    print(f"  fit set (train_fit non-cold): {len(train_nc_stems)}")
    Xtr_g8 = extract_g8(train_stems, train_nc_stems, CACHE_ROOT, BACKBONE, device="cpu")
    Xde_g8 = extract_g8(devel_stems, train_nc_stems, CACHE_ROOT, BACKBONE, device="cpu")
    print(f"G8 X_train={Xtr_g8.shape}  any_nan={np.isnan(Xtr_g8).any()}  "
          f"range=[{Xtr_g8.min():.1f}, {Xtr_g8.max():.1f}]")

    # Honesty audit (skip — k=210 won't fit on 64 train; just check column shapes)
    print("\n--- shapes look OK; honesty audit needs the full split. ---")
    print("smoke OK")


if __name__ == "__main__":
    main()
