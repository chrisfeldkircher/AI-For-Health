"""Smoke test for features/scalar_g9_cqt.py. Run from model/:
  python smoke_g9_cqt.py
Checks: import through the package, feature shape/dtype on synthetic audio,
degenerate short-input path, cache write + extract_g9 roundtrip.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from features import cqt_features, extract_g9, G9_DIM, G9_NAMES

assert G9_DIM == 168, G9_DIM
assert len(G9_NAMES) == G9_DIM
assert G9_NAMES[0] == "cqt_b00_amean" and G9_NAMES[-1] == "cqt_b83_stddev"

# 1. shape/dtype on a synthetic 8 s mixture (matches the pipeline clip length)
sr = 16000
t = np.arange(8 * sr) / sr
audio = (0.4 * np.sin(2 * np.pi * 220 * t)
         + 0.2 * np.sin(2 * np.pi * 880 * t)
         + 0.02 * np.random.RandomState(0).randn(t.size)).astype(np.float32)
v = cqt_features(audio, sr=sr)
assert v.shape == (G9_DIM,), v.shape
assert v.dtype == np.float32, v.dtype
assert np.isfinite(v).all()
assert v.std() > 0.0, "flat output on real signal"
print(f"[ok] cqt_features: shape {v.shape}, mean {v.mean():.2f} dB, std {v.std():.2f}")

# 2. degenerate short input returns zeros, right shape
v0 = cqt_features(np.zeros(100, dtype=np.float32), sr=sr)
assert v0.shape == (G9_DIM,) and (v0 == 0).all()
print("[ok] short-input degenerate path")

# 3. cache write + extract_g9 roundtrip
with tempfile.TemporaryDirectory() as td:
    cache_root = Path(td)
    out_dir = cache_root / "handcrafted" / "cqt"
    out_dir.mkdir(parents=True)
    np.save(out_dir / "train_0001.npy", v)
    np.save(out_dir / "train_0002.npy", v * 0.5)
    X = extract_g9(["train_0001", "train_0002"], cache_root)
    assert X.shape == (2, G9_DIM)
    assert np.allclose(X[0], v) and np.allclose(X[1], v * 0.5)
print("[ok] extract_g9 roundtrip")

print("SMOKE PASS")
