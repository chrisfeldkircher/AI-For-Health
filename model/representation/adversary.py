"""A7 — gradient-reversal speaker adversary (DANN; Ganin 2015).

Phase 1 PoC primitive: a gradient reversal layer + a pseudo-speaker
discriminator that the projection MLP must defeat. Forward pass is identity;
backward pass multiplies the gradient by `-lambda_adv`, so the projection
sees the discriminator's gradient as a push to make speaker harder to predict
(the discriminator itself trains normally to predict speaker -- arms race).

Design notes:
  - Lambda follows the standard Ganin sigmoid ramp:
        lambda_adv(p) = lambda_max * (2/(1+exp(-10p)) - 1)
    where p = epoch/total_epochs. Avoids hitting full adversarial pressure
    from step 1 (training instability mitigation).
  - Discriminator is intentionally medium-capacity (128 -> 256 -> n_speakers).
    Too-shallow = can't learn speaker, no adversary signal.
    Too-deep = overfits the 8.5k train chunks, noisy adversary signal.
  - Training loop accumulates per-epoch discriminator train accuracy as the
    health diagnostic that makes the verdict interpretable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverseFn(Function):
    """Forward = identity. Backward = -lambda_adv * grad."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.lambda_adv = float(lambda_adv)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
    """Functional API. Use at the projection -> discriminator boundary."""
    return GradReverseFn.apply(x, lambda_adv)


class SpeakerDiscriminator(nn.Module):
    """Medium-capacity MLP: z (proj_dim) -> hidden -> n_speakers logits.

    Defaults to a single-hidden 256-unit GELU MLP -- enough to learn linear-
    plus-lightly-nonlinear speaker structure without over-fitting 8.5k
    train chunks across k=210 pseudo-speakers (~40 chunks/speaker).
    """

    def __init__(
        self,
        proj_dim: int = 128,
        n_speakers: int = 210,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_speakers),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def lambda_adv_sigmoid(p: float, lambda_max: float, k: float = 10.0) -> float:
    """Ganin sigmoid ramp.

    p ∈ [0, 1] is training progress (e.g., epoch/total_epochs).
    Returns 0 at p=0, ~lambda_max at p=1; smoothly ramps via
        lambda_adv(p) = lambda_max * (2 / (1 + exp(-k*p)) - 1).
    Default k=10 matches Ganin et al. 2015.
    """
    if lambda_max == 0.0:
        return 0.0
    return float(lambda_max * (2.0 / (1.0 + math.exp(-k * p)) - 1.0))
