"""Smoke test for features/signature.py, incl. the log-signature MATH.
Run from model/:  python smoke_signature.py
"""
import numpy as np
from features import (logsig_depth2, acoustic_path, signature_features,
                      extract_g10, G10_DIM, G10_NAMES)

# --- 1. dimensions ---
assert G10_DIM == 10, G10_DIM
assert len(G10_NAMES) == 10

# --- 2. Levy area math: the canonical order-sensitivity test ---
# 2-D path. right-then-up vs up-then-right must give OPPOSITE-sign Levy areas.
right_then_up = np.array([[0., 0.], [1., 0.], [1., 1.]])
up_then_right = np.array([[0., 0.], [0., 1.], [1., 1.]])
s1 = logsig_depth2(right_then_up)   # [incr_x, incr_y, area_xy]
s2 = logsig_depth2(up_then_right)
assert s1.shape == (3,) and s2.shape == (3,), (s1.shape, s2.shape)
# increments identical (both end at (1,1) from (0,0))
assert np.allclose(s1[:2], [1., 1.]) and np.allclose(s2[:2], [1., 1.])
# Levy area: unit square half-areas, opposite signs. With the standard
# A^xy = 1/2 integral(x dy - y dx) convention, right-then-up = +0.5,
# up-then-right = -0.5.
assert np.isclose(s1[2], +0.5) and np.isclose(s2[2], -0.5), (s1[2], s2[2])
print(f"[ok] Levy area order-sensitivity: right-then-up={s1[2]:+.3f}, up-then-right={s2[2]:+.3f}")

# --- 3. translation invariance of the log-signature ---
P = np.random.RandomState(0).randn(50, 4)
assert np.allclose(logsig_depth2(P), logsig_depth2(P + 7.3)), "not translation-invariant"
print("[ok] translation invariance")

# --- 4. straight-line path has zero Levy area (no enclosed area) ---
t = np.linspace(0, 1, 40)[:, None]
line = np.hstack([t, 2 * t])            # collinear 2-D path
assert abs(logsig_depth2(line)[2]) < 1e-9, logsig_depth2(line)[2]
print("[ok] straight-line path -> zero Levy area")

# --- 5. degenerate short input ---
assert signature_features(np.zeros(100, dtype=np.float32)).shape == (G10_DIM,)
print("[ok] short-input degenerate path")

# --- 6. real audio path + feature vector finite ---
sr = 16000
tt = np.arange(4 * sr) / sr
audio = (0.4 * np.sin(2 * np.pi * 150 * tt)
         + 0.1 * np.random.RandomState(1).randn(tt.size)).astype(np.float32)
P = acoustic_path(audio, sr=sr)
assert P.shape[1] == 4 and P.shape[0] > 10, P.shape
v = signature_features(audio, sr=sr)
assert v.shape == (G10_DIM,) and np.isfinite(v).all() and v.std() > 0
print(f"[ok] real-audio signature: shape {v.shape}, std {v.std():.3f}")

# --- 7. cache roundtrip ---
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as td:
    root = Path(td); od = root / "handcrafted" / "signature"; od.mkdir(parents=True)
    np.save(od / "train_0001.npy", v)
    X = extract_g10(["train_0001"], root)
    assert X.shape == (1, G10_DIM) and np.allclose(X[0], v)
print("[ok] extract_g10 roundtrip")
print("SMOKE PASS")
