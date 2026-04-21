from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .backbone import Backbone


@dataclass
class CacheManifest:
    backbone_id: str
    checkpoint_hash: str
    transformers_version: str
    torch_version: str
    n_layers: int
    hidden_dim: int
    stat_dim: int
    dtype: str
    n_chunks: int
    created_at: str

    @classmethod
    def create(cls, backbone: "Backbone", stat_dim: int, n_chunks: int) -> "CacheManifest":
        import transformers
        return cls(
            backbone_id=backbone.backbone_id,
            checkpoint_hash=backbone.checkpoint_hash,
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            n_layers=backbone.n_layers,
            hidden_dim=backbone.hidden_dim,
            stat_dim=stat_dim,
            dtype="float16",
            n_chunks=n_chunks,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path) -> "CacheManifest":
        with open(path, "r") as f:
            return cls(**json.load(f))

    def is_compatible(self, other: "CacheManifest") -> bool:
        """Invalidation key: checkpoint_hash + n_layers + hidden_dim must agree."""
        return (
            self.checkpoint_hash == other.checkpoint_hash
            and self.n_layers == other.n_layers
            and self.hidden_dim == other.hidden_dim
        )


def save_pooled(path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def load_pooled(path) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu", weights_only=True)
