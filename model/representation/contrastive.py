"""A6 Phase 1 head-only PoC: supervised contrastive with speaker-masked positives.

Trains a fresh projection MLP (4096 -> 512 -> 128, L2-normalised output) on top
of frozen A2.5 standardiser + layer-weights. Loss is SupCon (Khosla et al.
2020) with the positive set restricted to same-Cold + different-pseudo-speaker
pairs; same-speaker same-class pairs are masked out -- that exclusion is the
de-confounding lever.

Design notes:
  - A2.5's own classifier head has a 256-d projection trained under cold loss;
    reusing it would entangle the contrastive verdict with classifier-shaped
    structure. Fresh projection isolates "did the contrastive recipe move the
    speaker probe?" from "did A2.5's projection already encode it?".
  - Layer mix and standardiser stay frozen at A2.5's converged values: per M5
    the layer-weight subspace needs lr x10 to actually move; keeping it frozen
    here is consistent with the head-only PoC scope.
  - Batches are 8 pseudo-speakers x 8 chunks. The sampler enforces "every
    anchor has at least one cross-speaker same-class positive" by construction
    when possible; rejection rate is logged as a diagnostic.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# projection MLP
# ---------------------------------------------------------------------------


class ContrastiveProjection(nn.Module):
    """4096 -> 512 -> 128 projection with GELU + LayerNorm, L2-normalised output.

    LayerNorm (not BatchNorm) because contrastive batches are small (64) and the
    SupCon paper finds LN/GroupNorm more stable for projection heads. L2-norm
    on output puts z on the unit hypersphere -- standard SupCon convention.
    """

    def __init__(self, in_dim: int = 4096, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


# ---------------------------------------------------------------------------
# frozen A2.5 layer mix + standardiser
# ---------------------------------------------------------------------------


@torch.no_grad()
def fuse_with_a25(
    pooled: torch.Tensor,
    scaler: nn.Module,
    layer_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply A2.5's frozen standardiser + softmax layer-mix to pooled stats.

    pooled        : [B, n_layers, stat_dim] fp16/fp32
    scaler        : FeatureStandardiser (state copied from A2.5 checkpoint)
    layer_weights : [n_layers] raw logits from A2.5 (pre-softmax)
    -> fused      : [B, stat_dim] fp32
    """
    x = scaler(pooled.to(torch.float32))
    w = torch.softmax(layer_weights, dim=0).view(1, -1, 1)
    return (x * w).sum(dim=1)


# ---------------------------------------------------------------------------
# SupCon with speaker-masked positives
# ---------------------------------------------------------------------------


def supcon_speaker_masked_loss(
    z: torch.Tensor,
    cold_labels: torch.Tensor,
    pseudo_speakers: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-12,
    mask_speakers: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Supervised contrastive loss with same-speaker same-class positives masked out.

    z               : [B, D] L2-normalised projection output
    cold_labels     : [B] long  (0 NC, 1 C)
    pseudo_speakers : [B] long  (k=210 cluster assignments)
    temperature     : float

    Returns (loss, stats). stats contains diagnostic counts.

    Positive set for anchor i:
        same_class[i, j] = (cold_labels[i] == cold_labels[j])
        diff_speaker[i, j] = (pseudo_speakers[i] != pseudo_speakers[j])
        positive[i, j] = same_class[i, j] AND diff_speaker[i, j]
        (and i != j)

    Negative set for anchor i:
        negative[i, j] = NOT same_class[i, j]   (and i != j)
        # Different-class pairs are negatives regardless of speaker.

    Same-class same-speaker pairs are NEITHER positives NOR negatives -- they
    are excluded from the denominator entirely. Including them as implicit
    negatives (the standard SupCon denominator) would push same-speaker chunks
    apart, which is the OPPOSITE of speaker-invariance and anti-de-confounding.

    Set `mask_speakers=False` for the vanilla SupCon control: positive set =
    same_class & ~eye (regardless of speaker), denominator = ~eye. This is
    used to disambiguate "speaker-masking activates de-confounding" from
    "any contrastive class-pressure activates de-confounding" -- if vanilla
    SupCon gives the same probe drop, the speaker-masking lever does no
    additional work.

    Anchors with empty positive sets are excluded from the loss (their
    contribution would be 0/0). Count of excluded anchors is reported in stats
    as `n_anchors_excluded`.
    """
    B = z.shape[0]
    device = z.device

    # Pairwise cosine similarities scaled by temperature.
    sim = z @ z.T / temperature                                            # [B, B]

    # Numerical stability: subtract per-row max before exp.
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    sim = sim - sim_max

    # Build positive, negative, and denominator masks.
    same_class = cold_labels.unsqueeze(0) == cold_labels.unsqueeze(1)      # [B, B]
    diff_spk   = pseudo_speakers.unsqueeze(0) != pseudo_speakers.unsqueeze(1)
    eye        = torch.eye(B, dtype=torch.bool, device=device)

    if mask_speakers:
        pos_mask = same_class & diff_spk & ~eye                            # [B, B]
    else:
        # Vanilla SupCon: positives are any same-class non-self pair.
        pos_mask = same_class & ~eye                                       # [B, B]
    neg_mask   = (~same_class) & ~eye                                      # [B, B]
    denom_mask = pos_mask | neg_mask                                       # [B, B]

    # Speaker-masked denominator: positives (same-class diff-speaker) +
    # true negatives (different-class). Same-class same-speaker pairs are
    # absent from the denominator -- the loss never sees them, so the gradient
    # neither pulls them together (would re-introduce speaker structure) nor
    # pushes them apart (would be anti-de-confounding).
    exp_sim   = torch.exp(sim) * denom_mask
    log_denom = torch.log(exp_sim.sum(dim=1) + eps)                        # [B]

    # log( exp(sim) ) = sim.  Per-anchor log-prob over its positive set:
    #   L_i = -1/|P(i)| * sum_{p in P(i)} ( sim[i, p] - log_denom[i] )
    n_pos      = pos_mask.sum(dim=1).float()                               # [B]
    valid      = n_pos > 0
    n_excluded = int((~valid).sum().item())

    if not valid.any():
        # Pathological batch: no anchor has a same-class different-speaker
        # partner. Loss is undefined; return zero with valid grad path so the
        # optimizer step is a no-op.
        zero = z.sum() * 0.0
        return zero, {
            "loss":               0.0,
            "n_anchors":          B,
            "n_anchors_excluded": B,
            "n_positives_mean":   0.0,
            "n_positives_min":    0,
        }

    pos_term = (sim * pos_mask).sum(dim=1)                                 # [B]
    log_prob = (pos_term / n_pos.clamp(min=1)) - log_denom                 # [B]
    loss = -log_prob[valid].mean()

    return loss, {
        "loss":               float(loss.detach().item()),
        "n_anchors":          B,
        "n_anchors_excluded": n_excluded,
        "n_positives_mean":   float(n_pos[valid].mean().item()),
        "n_positives_min":    int(n_pos[valid].min().item()),
    }


# ---------------------------------------------------------------------------
# speaker-block batch sampler
# ---------------------------------------------------------------------------


@dataclass
class _SamplerStats:
    n_batches_yielded: int
    n_batches_rejected: int
    batch_size_actual: list[int]
    n_speakers_per_batch_actual: list[int]


class SpeakerBlockSampler(Sampler[list[int]]):
    """Yield batches of (n_speakers x chunks_per_speaker) chunk indices.

    Default 8 x 8 = 64 chunks per batch. Each batch contains chunks from
    `n_speakers` distinct pseudo-speakers (sampled without replacement per
    epoch); within each speaker-block, `chunks_per_speaker` chunks are drawn
    without replacement from that speaker's chunks.

    The class-balance heuristic: prefer speakers whose chunk pool contains
    BOTH cold and non-cold so each batch has class diversity. With URTIC's
    9.5% cold rate, most speakers are NC-only, so the sampler oversamples
    speakers that have at least one cold chunk to reach a target cold-fraction
    per batch (default 0.20, well above corpus rate so each anchor has
    cross-speaker same-class candidates with high probability).

    Batches where any anchor has 0 valid same-class different-speaker positives
    are rejected and a replacement batch is drawn (logged in stats).
    """

    def __init__(
        self,
        labels: list[int],
        pseudo_speakers: list[int],
        n_speakers: int = 8,
        chunks_per_speaker: int = 8,
        target_cold_fraction: float = 0.20,
        n_batches_per_epoch: Optional[int] = None,
        max_rejections: int = 5,
        seed: int = 42,
    ):
        if len(labels) != len(pseudo_speakers):
            raise ValueError("labels and pseudo_speakers must have the same length")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.pseudo_speakers = np.asarray(pseudo_speakers, dtype=np.int64)
        self.n_speakers = n_speakers
        self.chunks_per_speaker = chunks_per_speaker
        self.target_cold_fraction = target_cold_fraction
        self.max_rejections = max_rejections
        self.seed = seed
        self.epoch = 0

        # Index speaker -> list of chunk indices, split by class.
        spk_to_idx_cold: dict[int, list[int]] = defaultdict(list)
        spk_to_idx_nc:   dict[int, list[int]] = defaultdict(list)
        for i, (lab, spk) in enumerate(zip(self.labels, self.pseudo_speakers)):
            if lab == 1:
                spk_to_idx_cold[int(spk)].append(i)
            elif lab == 0:
                spk_to_idx_nc[int(spk)].append(i)
            # ignore -1 (unlabelled)

        # Speakers usable as cold-anchors (have >= 1 cold chunk).
        self.cold_speakers = sorted(spk_to_idx_cold.keys())
        # Speakers usable as nc-anchors (have >= 1 nc chunk).
        self.nc_speakers   = sorted(spk_to_idx_nc.keys())
        self.spk_to_idx_cold = dict(spk_to_idx_cold)
        self.spk_to_idx_nc   = dict(spk_to_idx_nc)
        # All-class chunks per speaker (for filling).
        self.spk_to_idx_all: dict[int, list[int]] = defaultdict(list)
        for i, spk in enumerate(self.pseudo_speakers):
            if self.labels[i] >= 0:
                self.spk_to_idx_all[int(spk)].append(i)

        # Default: enough batches that each cold chunk is seen ~once per epoch.
        n_cold = int((self.labels == 1).sum())
        target_cold_per_batch = max(1, int(round(n_speakers * chunks_per_speaker
                                                  * target_cold_fraction)))
        if n_batches_per_epoch is None:
            n_batches_per_epoch = max(1, n_cold // target_cold_per_batch)
        self.n_batches_per_epoch = n_batches_per_epoch

        self.stats = _SamplerStats(
            n_batches_yielded=0,
            n_batches_rejected=0,
            batch_size_actual=[],
            n_speakers_per_batch_actual=[],
        )

    def __len__(self) -> int:
        return self.n_batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _draw_batch(self, rng: np.random.Generator) -> Optional[list[int]]:
        """Draw one batch. Returns None if construction failed (caller may retry)."""
        # Decide split between cold-anchored and nc-anchored speakers.
        n_cold_speakers = max(1, int(round(self.n_speakers * self.target_cold_fraction
                                           * (self.chunks_per_speaker / self.chunks_per_speaker))))
        n_cold_speakers = min(n_cold_speakers, self.n_speakers, len(self.cold_speakers))
        n_nc_speakers = self.n_speakers - n_cold_speakers
        n_nc_speakers = min(n_nc_speakers, len(self.nc_speakers))

        if n_cold_speakers + n_nc_speakers < 2:
            return None

        cold_pick = list(rng.choice(self.cold_speakers, size=n_cold_speakers, replace=False))
        # NC pool excludes any speaker already picked as cold (a speaker may
        # appear in both lists; keep the cold-picked ones out of the nc pool).
        nc_pool = [s for s in self.nc_speakers if s not in set(cold_pick)]
        if len(nc_pool) < n_nc_speakers:
            n_nc_speakers = len(nc_pool)
        nc_pick = list(rng.choice(nc_pool, size=n_nc_speakers, replace=False)) if n_nc_speakers else []

        batch: list[int] = []
        for spk in cold_pick:
            cold_chunks = self.spk_to_idx_cold.get(int(spk), [])
            nc_chunks   = self.spk_to_idx_nc.get(int(spk), [])
            # Take all cold chunks first (up to chunks_per_speaker), fill rest with nc.
            n_take_cold = min(len(cold_chunks), self.chunks_per_speaker)
            if n_take_cold:
                batch.extend(rng.choice(cold_chunks, size=n_take_cold, replace=False).tolist())
            n_fill = self.chunks_per_speaker - n_take_cold
            if n_fill > 0 and nc_chunks:
                n_fill = min(n_fill, len(nc_chunks))
                batch.extend(rng.choice(nc_chunks, size=n_fill, replace=False).tolist())
        for spk in nc_pick:
            nc_chunks = self.spk_to_idx_nc.get(int(spk), [])
            n_take = min(len(nc_chunks), self.chunks_per_speaker)
            if n_take:
                batch.extend(rng.choice(nc_chunks, size=n_take, replace=False).tolist())

        if len(batch) < 4:                           # too small to be meaningful
            return None

        # Validate: every anchor must have at least one same-class
        # different-speaker partner in this batch.
        labels = self.labels[batch]
        speakers = self.pseudo_speakers[batch]
        for i in range(len(batch)):
            mask = (labels == labels[i]) & (speakers != speakers[i])
            mask[i] = False
            if not mask.any():
                return None                          # rejected; retry

        self.stats.n_speakers_per_batch_actual.append(
            int(np.unique(speakers).size)
        )
        self.stats.batch_size_actual.append(len(batch))
        return batch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n_batches_per_epoch):
            for attempt in range(self.max_rejections):
                batch = self._draw_batch(rng)
                if batch is not None:
                    self.stats.n_batches_yielded += 1
                    yield batch
                    break
                self.stats.n_batches_rejected += 1
            else:
                # Exhausted retries -- yield best-effort batch (may have a few
                # invalid anchors, which the loss will exclude from the mean).
                batch = self._draw_batch(rng) or []
                if batch:
                    self.stats.n_batches_yielded += 1
                    yield batch
