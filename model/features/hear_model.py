# HeAR (Google Health Acoustic Representations) embedder for URTIC cold detection.
#
# WHY THIS IS NOT JUST AutoModel.from_pretrained: HeAR's config declares a
# non-standard pooler (Linear 1024->512, linear activation; keys pooler_output_size
# / pooler_act / pooled_dim) and the repo ships NO custom modeling file. The env's
# transformers is 4.48.0, whose ViTPooler is hardcoded Linear(1024->1024)+Tanh, so
# AutoModel.from_pretrained would build the WRONG pooler and emit a random-init
# pooler_output. HeAR pins transformers==4.50.3 for this reason. Instead of
# upgrading the shared env (which the WavLM/HuBERT pipeline depends on) we load the
# standard ViT trunk with 4.48.0 and reconstruct the real HeAR pooler from the two
# checkpoint tensors pooler.dense.{weight,bias}. Verified against the checkpoint:
# 24 layers, CLS+96 patches (image_size [192,128], patch 16, 1 channel), a final
# `layernorm`, and pooler.dense = Linear(1024->512).
#
# The 512-d HeAR embedding == pooler.dense( last_hidden_state[:, 0] ), where
# last_hidden_state is post-final-layernorm (as in transformers ViTModel) and
# pooler_act is linear (identity). This matches the upstream PyTorch quick-start,
# which reads output.pooler_output.
"""HeAR ViT embedder with reconstructed 1024->512 linear pooler."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hear_preprocess import preprocess_windows, CLIP_SAMPLES

HEAR_REPO = "google/hear-pytorch"


def make_windows(wav: torch.Tensor, hop: int = CLIP_SAMPLES,
                 min_tail: int = 8000) -> torch.Tensor:
  """Tile a 1-D 16 kHz waveform into [W, 32000] windows (last zero-padded).

  Non-overlapping by default (hop == 32000). A trailing partial window with fewer
  than `min_tail` real samples is dropped unless it is the only window, so the
  mean-pooled embedding is not diluted by a near-silent tail.
  """
  if wav.ndim != 1:
    wav = wav.reshape(-1)
  n = int(wav.shape[0])
  if n == 0:
    return torch.zeros((1, CLIP_SAMPLES), dtype=torch.float32)
  starts = list(range(0, n, hop))
  wins = []
  for s in starts:
    w = wav[s:s + CLIP_SAMPLES]
    real = int(w.shape[0])
    if real < CLIP_SAMPLES:
      if len(starts) > 1 and real < min_tail:
        continue  # drop mostly-silent trailing window
      w = F.pad(w, (0, CLIP_SAMPLES - real))
    wins.append(w)
  if not wins:  # safety (e.g. everything dropped)
    w = wav[:CLIP_SAMPLES]
    wins = [F.pad(w, (0, CLIP_SAMPLES - int(w.shape[0])))]
  return torch.stack(wins, 0)


class HearEmbedder:
  """Frozen HeAR ViT trunk + reconstructed linear pooler -> 512-d embeddings."""

  def __init__(self, vit: nn.Module, pooler: nn.Linear, device: str):
    self.vit = vit
    self.pooler = pooler
    self.device = device
    self.embed_dim = pooler.out_features  # 512

  @torch.no_grad()
  def embed_windows(self, windows: torch.Tensor) -> torch.Tensor:
    """[W, <=32000] waveform windows -> [W, 512] HeAR embeddings.

    The mel-PCEN preprocessing runs on CPU: (1) it is the exact path proven
    numerically identical (0.0) to Google's upstream reference, whereas CUDA
    FFT/interp kernels diverge slightly; and (2) torch.abs on a complex STFT
    fails to JIT-compile on this CUDA build. Only the heavy ViT forward is on GPU.
    """
    pixel_values = preprocess_windows(windows.float().cpu())   # [W, 1, 192, 128]
    pixel_values = pixel_values.to(self.device)
    seq = self.vit(pixel_values=pixel_values).last_hidden_state  # [W, 97, 1024]
    cls = seq[:, 0]                                        # [W, 1024] (post-LN)
    return self.pooler(cls)                               # [W, 512] linear

  @torch.no_grad()
  def embed_waveform(self, wav: torch.Tensor, hop: int = CLIP_SAMPLES,
                     min_tail: int = 8000, vit_batch: int = 64) -> torch.Tensor:
    """1-D 16 kHz waveform -> single [512] mean-pooled embedding."""
    windows = make_windows(wav, hop=hop, min_tail=min_tail)  # [W, 32000]
    total = None
    n = 0
    for i in range(0, windows.shape[0], vit_batch):
      emb = self.embed_windows(windows[i:i + vit_batch])      # [w, 512]
      s = emb.sum(dim=0)
      total = s if total is None else total + s
      n += int(emb.shape[0])
    return (total / n).detach().float().cpu()                 # [512]


def load_hear(device: str = "cpu", repo: str = HEAR_REPO) -> HearEmbedder:
  """Load the HeAR ViT trunk (4.48-compatible) + reconstruct the 1024->512 pooler.

  Raises if any trunk weight is missing or unexpected, or if the pooler shape is
  not (512, 1024) -- i.e. it fails loud rather than silently emitting garbage.
  """
  import truststore
  truststore.inject_into_ssl()
  from transformers import ViTModel, AutoConfig
  from huggingface_hub import hf_hub_download

  cfg = AutoConfig.from_pretrained(repo)
  bin_path = hf_hub_download(repo, "pytorch_model.bin")
  sd = torch.load(bin_path, map_location="cpu", weights_only=True)

  vit = ViTModel(cfg, add_pooling_layer=False)
  trunk = {k: v for k, v in sd.items() if not k.startswith("pooler.")}
  missing, unexpected = vit.load_state_dict(trunk, strict=False)
  if missing:
    raise RuntimeError(f"HeAR trunk is missing weights: {missing}")
  if unexpected:
    raise RuntimeError(f"HeAR trunk got unexpected weights: {unexpected}")

  w = sd["pooler.dense.weight"]
  b = sd["pooler.dense.bias"]
  if tuple(w.shape) != (512, 1024):
    raise RuntimeError(f"unexpected HeAR pooler shape {tuple(w.shape)}, want (512, 1024)")
  pooler = nn.Linear(w.shape[1], w.shape[0])
  with torch.no_grad():
    pooler.weight.copy_(w)
    pooler.bias.copy_(b)

  vit.eval().to(device)
  pooler.eval().to(device)
  return HearEmbedder(vit, pooler, device)
