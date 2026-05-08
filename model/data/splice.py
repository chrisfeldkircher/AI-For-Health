"""Cross-speaker splicing primitives for A5.5 augmentation.

Implements plan.md §6 (A5.5) recipe: replace a fraction r of an anchor chunk
with a partner segment, splicing on silence boundaries when possible (unvoiced
fallback), with equal-power crossfade and RMS-match. Output is the same length
as the anchor so WavLM frame counts are preserved.

Boundary picker reads from cached manner labels (cache/manner_labels/{stem}.pt,
50 Hz int8 in {0=silence, 1=voiced, 2=unvoiced}). Frame i corresponds to
audio samples [i*HOP, (i+1)*HOP] at SR=16000 (HOP=320).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
HOP = 320              # WavLM stride: 50 Hz frame rate
MIN_RUN_FRAMES = 2     # 40 ms minimum silence/unvoiced run
CROSSFADE_SIL_MS = 150
CROSSFADE_UV_MS  = 250
CAT_SILENCE, CAT_VOICED, CAT_UNVOICED = 0, 1, 2


# ---------------------------------------------------------------------------
# Frame ↔ sample conversion (manner labels are at 50 Hz)
# ---------------------------------------------------------------------------
def frames_to_samples(f: int) -> int:
    return int(f) * HOP

def samples_to_frames(s: int) -> int:
    return int(s) // HOP


# ---------------------------------------------------------------------------
# Manner-label run finders
# ---------------------------------------------------------------------------
def find_runs(manner: np.ndarray, label: int, min_frames: int) -> list[tuple[int, int]]:
    """Return list of (start_frame, end_frame_exclusive) where manner == label
    and the run length >= min_frames."""
    if len(manner) == 0:
        return []
    is_target = (manner == label)
    diffs = np.diff(np.concatenate(([False], is_target, [False])).astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_frames]


def voiced_fraction_in_window(manner: np.ndarray, s0_sample: int, s1_sample: int) -> float:
    """Fraction of frames in audio window [s0_sample, s1_sample] labelled as voiced."""
    f0 = samples_to_frames(s0_sample)
    f1 = samples_to_frames(s1_sample)
    if f1 <= f0:
        return 0.0
    sub = manner[f0:f1]
    if len(sub) == 0:
        return 0.0
    return float((sub == CAT_VOICED).mean())


# ---------------------------------------------------------------------------
# Audio primitives
# ---------------------------------------------------------------------------
def equal_power_crossfade(left_tail: np.ndarray, right_head: np.ndarray) -> np.ndarray:
    """Equal-power blend: out[i] = left[i]*cos(θ) + right[i]*sin(θ), θ ∈ [0, π/2].

    Both inputs are length n; returns length-n blend. cos/sin keep total power
    flat across the fade (a linear blend has a -6 dB perceptual dip mid-fade)."""
    n = min(len(left_tail), len(right_head))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    theta = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    fade_out = np.cos(theta).astype(np.float32)
    fade_in  = np.sin(theta).astype(np.float32)
    return (left_tail[:n].astype(np.float32) * fade_out
            + right_head[:n].astype(np.float32) * fade_in)


def local_rms(segment: np.ndarray) -> float:
    return float(np.sqrt((segment.astype(np.float32) ** 2).mean() + 1e-12))


def rms_match(segment: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale segment so its RMS matches target_rms. Preserves sign/phase."""
    s_rms = local_rms(segment)
    if s_rms < 1e-8:
        return segment.astype(np.float32, copy=False)
    return (segment.astype(np.float32) * (target_rms / s_rms)).astype(np.float32)


# ---------------------------------------------------------------------------
# Boundary pickers
# ---------------------------------------------------------------------------
@dataclass
class AnchorBoundaries:
    t0_sample: int
    t1_sample: int
    boundary_kind: str   # "silence" | "unvoiced"


@dataclass
class PartnerWindow:
    s0_sample: int
    s1_sample: int
    boundary_kind: str   # "silence" | "unvoiced"
    voiced_fraction: float


def _pick_frame_in_run(run: tuple[int, int], rng: np.random.Generator) -> int:
    """Pick a random frame inside a run [start, end_exclusive)."""
    s, e = run
    return int(rng.integers(s, e))


def pick_anchor_boundaries(
    anchor_manner: np.ndarray,
    valid_samples: int,
    r: float,
    rng: np.random.Generator,
    attempts: int = 20,
    slack_frac: float = 0.20,
) -> Optional[AnchorBoundaries]:
    """Pick splice boundaries (t0, t1) in audio samples such that t1-t0 ≈ r*valid_samples.

    Tries silence-on-silence first; falls back to unvoiced-on-unvoiced. The two
    boundaries must come from runs of the same kind. Always validates against
    the LARGER (unvoiced) crossfade window so the splicer can safely downgrade
    silence→unvoiced if the partner has no silence boundaries. Constrains the
    boundaries to leave a cf_max margin from both audio endpoints. Returns None
    if no valid pair found in `attempts` tries per kind.
    """
    target_n = int(r * valid_samples)
    slack = int(slack_frac * target_n)
    cf_max = int(CROSSFADE_UV_MS * SAMPLE_RATE / 1000)
    safe_lo = cf_max
    safe_hi = valid_samples - cf_max
    if safe_hi <= safe_lo + 2 * cf_max + 1:
        return None  # audio too short for any safe splice

    for kind, label in (("silence", CAT_SILENCE), ("unvoiced", CAT_UNVOICED)):
        # Accept runs that overlap the safe range at all; clip to range later.
        runs = [
            (s, e) for (s, e) in find_runs(anchor_manner, label, MIN_RUN_FRAMES)
            if frames_to_samples(e) > safe_lo and frames_to_samples(s) < safe_hi
        ]
        if len(runs) < 2:
            continue
        for _ in range(attempts):
            r1 = runs[rng.integers(0, len(runs))]
            t0_sample = frames_to_samples(_pick_frame_in_run(r1, rng))
            t0_sample = min(max(t0_sample, safe_lo), safe_hi - 2 * cf_max - 1)
            t1_sample_target = t0_sample + target_n
            best = None
            best_dist = slack + 1
            for run in runs:
                s_sample = frames_to_samples(run[0])
                e_sample = frames_to_samples(run[1])
                if e_sample <= t0_sample + 2 * cf_max + 1:
                    continue   # must leave room for both crossfades
                if s_sample <= t1_sample_target <= e_sample:
                    candidate = t1_sample_target
                    dist = 0
                elif e_sample < t1_sample_target:
                    candidate = e_sample - 1
                    dist = t1_sample_target - candidate
                else:
                    candidate = s_sample
                    dist = candidate - t1_sample_target
                # Clip candidate to safe range
                candidate = min(candidate, safe_hi)
                if candidate <= t0_sample + 2 * cf_max:
                    continue
                if dist < best_dist:
                    best, best_dist = candidate, dist
            if best is not None and (best - t0_sample) >= 2 * cf_max + 1:
                return AnchorBoundaries(t0_sample=t0_sample,
                                        t1_sample=int(best),
                                        boundary_kind=kind)
    return None


def pick_partner_window(
    partner_manner: np.ndarray,
    partner_valid_samples: int,
    target_samples: int,
    rng: np.random.Generator,
    attempts: int = 20,
    voiced_floor: float = 0.50,
) -> Optional[PartnerWindow]:
    """Pick partner window [s0, s1] of length `target_samples`. Both endpoints
    must land in silence (or unvoiced fallback) runs of ≥ MIN_RUN_FRAMES; the
    window's voiced-fraction must be ≥ voiced_floor; both endpoints must leave
    cf_max margin from the audio bounds."""
    cf_max = int(CROSSFADE_UV_MS * SAMPLE_RATE / 1000)
    safe_lo = cf_max
    safe_hi = partner_valid_samples - cf_max
    if target_samples > safe_hi - safe_lo:
        return None

    for kind, label in (("silence", CAT_SILENCE), ("unvoiced", CAT_UNVOICED)):
        runs = [
            (s, e) for (s, e) in find_runs(partner_manner, label, MIN_RUN_FRAMES)
            if frames_to_samples(e) > safe_lo and frames_to_samples(s) < safe_hi
        ]
        if not runs:
            continue
        for _ in range(attempts):
            r0 = runs[rng.integers(0, len(runs))]
            s0_sample = min(max(frames_to_samples(_pick_frame_in_run(r0, rng)), safe_lo),
                            safe_hi - target_samples)
            s1_sample_target = s0_sample + target_samples
            if s1_sample_target > safe_hi:
                continue
            ok = False
            for run in runs:
                s_sample = frames_to_samples(run[0])
                e_sample = frames_to_samples(run[1])
                if s_sample <= s1_sample_target <= e_sample:
                    ok = True
                    break
            if not ok:
                continue
            vf = voiced_fraction_in_window(partner_manner, s0_sample, s1_sample_target)
            if vf >= voiced_floor:
                return PartnerWindow(s0_sample=s0_sample,
                                     s1_sample=s1_sample_target,
                                     boundary_kind=kind,
                                     voiced_fraction=vf)
    return None


# ---------------------------------------------------------------------------
# Splice composer
# ---------------------------------------------------------------------------
@dataclass
class SpliceResult:
    audio: np.ndarray
    skipped: bool
    boundary_kind: Optional[str]   # "silence" | "unvoiced" | None
    crossfade_samples: int
    t0_sample: Optional[int]
    t1_sample: Optional[int]
    s0_sample: Optional[int]
    s1_sample: Optional[int]
    partner_voiced_fraction: Optional[float]
    skip_reason: Optional[str]


def splice_chunk(
    anchor_audio: np.ndarray,
    anchor_manner: np.ndarray,
    anchor_valid_samples: int,
    partner_audio: np.ndarray,
    partner_manner: np.ndarray,
    partner_valid_samples: int,
    r: float,
    rng: np.random.Generator,
) -> SpliceResult:
    """Plan §6 splice: replace anchor[t0:t1] with rms-matched, crossfaded
    partner[s0:s1]. Both segments are the same length (t1-t0). Output is the
    same length as anchor; padding (if any) is preserved.
    """
    out_len = len(anchor_audio)
    out = anchor_audio.astype(np.float32, copy=True)

    ab = pick_anchor_boundaries(anchor_manner, anchor_valid_samples, r, rng)
    if ab is None:
        return SpliceResult(out, True, None, 0, None, None, None, None, None,
                            "no_valid_anchor_boundaries")
    target_n = ab.t1_sample - ab.t0_sample
    pw = pick_partner_window(partner_manner, partner_valid_samples, target_n, rng)
    if pw is None:
        return SpliceResult(out, True, None, 0, None, None, None, None, None,
                            "no_valid_partner_window")
    # Boundary kind for the crossfade window
    kind = ab.boundary_kind if ab.boundary_kind == pw.boundary_kind else "unvoiced"
    cf_ms = CROSSFADE_SIL_MS if kind == "silence" else CROSSFADE_UV_MS
    cf = int(cf_ms * SAMPLE_RATE / 1000)
    if 2 * cf >= target_n:
        return SpliceResult(out, True, None, 0, None, None, None, None, None,
                            "segment_too_short_for_crossfade")
    # Defensive bounds: every read/write region must fit inside the audio arrays.
    if (ab.t0_sample < 0 or ab.t1_sample > len(anchor_audio)
            or ab.t1_sample > anchor_valid_samples
            or pw.s0_sample < 0 or pw.s1_sample > len(partner_audio)
            or pw.s1_sample > partner_valid_samples):
        return SpliceResult(out, True, None, 0, None, None, None, None, None,
                            "boundaries_out_of_audio_bounds")

    # Carve segments
    seg_partner = partner_audio[pw.s0_sample : pw.s1_sample].astype(np.float32, copy=True)
    target_rms = local_rms(anchor_audio[ab.t0_sample : ab.t1_sample])
    seg_partner = rms_match(seg_partner, target_rms)

    # Compose: anchor[:t0] | xfade(anchor[t0:t0+cf], partner[:cf]) |
    #          partner[cf:N-cf] | xfade(partner[N-cf:], anchor[t1-cf:t1]) | anchor[t1:]
    out[: ab.t0_sample] = anchor_audio[: ab.t0_sample]
    cf_left  = equal_power_crossfade(
        anchor_audio[ab.t0_sample : ab.t0_sample + cf].astype(np.float32),
        seg_partner[:cf],
    )
    out[ab.t0_sample : ab.t0_sample + cf] = cf_left
    middle_n = target_n - 2 * cf
    out[ab.t0_sample + cf : ab.t1_sample - cf] = seg_partner[cf : cf + middle_n]
    cf_right = equal_power_crossfade(
        seg_partner[target_n - cf : target_n],
        anchor_audio[ab.t1_sample - cf : ab.t1_sample].astype(np.float32),
    )
    out[ab.t1_sample - cf : ab.t1_sample] = cf_right
    out[ab.t1_sample :] = anchor_audio[ab.t1_sample :]

    return SpliceResult(
        audio=out,
        skipped=False,
        boundary_kind=kind,
        crossfade_samples=cf,
        t0_sample=ab.t0_sample,
        t1_sample=ab.t1_sample,
        s0_sample=pw.s0_sample,
        s1_sample=pw.s1_sample,
        partner_voiced_fraction=pw.voiced_fraction,
        skip_reason=None,
    )


# ---------------------------------------------------------------------------
# Partner sampling (same Cold label, different pseudo-speaker)
# ---------------------------------------------------------------------------
def build_partner_pool(
    files: list[str],
    labels: dict[str, int],
    pseudo: dict[str, int],
) -> dict[str, list[str]]:
    """For each anchor file, list partner files with same Cold label and
    different pseudo-speaker. Keys/values are file names with .wav suffix as
    they appear in URTIC's TSVs; pseudo-speaker lookup strips the suffix."""
    def _stem(f: str) -> str:
        return f[:-4] if f.endswith(".wav") else f

    pool: dict[str, list[str]] = {}
    for f in files:
        if f not in labels:
            pool[f] = []
            continue
        f_label = labels[f]
        f_spk   = pseudo.get(_stem(f))
        partners = [
            g for g in files
            if g != f
            and labels.get(g) == f_label
            and (f_spk is None or pseudo.get(_stem(g)) != f_spk)
        ]
        pool[f] = partners
    return pool


def sample_partner(
    anchor: str,
    partner_pool: dict[str, list[str]],
    rng: np.random.Generator,
) -> Optional[str]:
    pool = partner_pool.get(anchor, [])
    if not pool:
        return None
    return pool[int(rng.integers(0, len(pool)))]
