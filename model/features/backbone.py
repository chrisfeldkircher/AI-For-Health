from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class Backbone(nn.Module, ABC):
    """
    Frozen audio foundation model used as a feature extractor.

    n_layers is the number of hidden states returned by forward(), not the
    number of transformer blocks. For WavLM/HuBERT this includes the CNN
    feature extractor output at index 0 (so WavLM-Large → 25, not 24).
    Whisper encoders expose only transformer blocks (no stem), so
    WavLM-Large → 25 and whisper-large → 32 by the same naming rule.
    """

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def backbone_id(self) -> str: ...

    @property
    @abstractmethod
    def checkpoint_hash(self) -> str: ...

    @abstractmethod
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        audio:          [B, T_audio] fp32 waveform at self.sample_rate
        attention_mask: [B, T_audio] long (1 valid, 0 pad), optional
        returns:
            hidden:       [B, n_layers, T_out, hidden_dim] fp16
            output_mask:  [B, T_out] bool, True for valid frames
        """


def _hash_config(cfg_dict: dict, extra: str = "") -> str:
    blob = json.dumps(cfg_dict, sort_keys=True) + extra
    return hashlib.md5(blob.encode()).hexdigest()[:16]


class WavLMBackbone(Backbone):
    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = "cuda"):
        super().__init__()
        from transformers import WavLMModel

        self._model_name = model_name
        self._device = device
        self.model = WavLMModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False

        self._n_layers = self.model.config.num_hidden_layers + 1
        self._hidden_dim = self.model.config.hidden_size
        self._checkpoint_hash = _hash_config(self.model.config.to_dict(), extra=model_name)

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def backbone_id(self) -> str:
        return self._model_name.replace("/", "_")

    @property
    def checkpoint_hash(self) -> str:
        return self._checkpoint_hash

    @torch.no_grad()
    def forward(self, audio, attention_mask=None):
        audio = audio.to(self._device, dtype=torch.float16)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device, dtype=torch.long)

        out = self.model(
            input_values=audio,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = torch.stack(out.hidden_states, dim=1).to(torch.float16)

        B, _, T_out, _ = hidden.shape
        if attention_mask is None:
            output_mask = torch.ones(B, T_out, dtype=torch.bool, device=hidden.device)
        else:
            in_lens = attention_mask.sum(dim=1)
            out_lens = self.model._get_feat_extract_output_lengths(in_lens).long()
            output_mask = torch.arange(T_out, device=hidden.device)[None, :] < out_lens[:, None]

        return hidden, output_mask


class HuBERTBackbone(Backbone):
    def __init__(self, model_name: str = "facebook/hubert-large-ll60k", device: str = "cuda"):
        super().__init__()
        from transformers import HubertModel

        self._model_name = model_name
        self._device = device
        self.model = HubertModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False

        self._n_layers = self.model.config.num_hidden_layers + 1
        self._hidden_dim = self.model.config.hidden_size
        self._checkpoint_hash = _hash_config(self.model.config.to_dict(), extra=model_name)

    @property
    def n_layers(self) -> int: return self._n_layers
    @property
    def hidden_dim(self) -> int: return self._hidden_dim
    @property
    def sample_rate(self) -> int: return 16000
    @property
    def backbone_id(self) -> str: return self._model_name.replace("/", "_")
    @property
    def checkpoint_hash(self) -> str: return self._checkpoint_hash

    @torch.no_grad()
    def forward(self, audio, attention_mask=None):
        audio = audio.to(self._device, dtype=torch.float16)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device, dtype=torch.long)

        out = self.model(
            input_values=audio,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = torch.stack(out.hidden_states, dim=1).to(torch.float16)

        B, _, T_out, _ = hidden.shape
        if attention_mask is None:
            output_mask = torch.ones(B, T_out, dtype=torch.bool, device=hidden.device)
        else:
            in_lens = attention_mask.sum(dim=1)
            out_lens = self.model._get_feat_extract_output_lengths(in_lens).long()
            output_mask = torch.arange(T_out, device=hidden.device)[None, :] < out_lens[:, None]

        return hidden, output_mask


class WhisperEncoderBackbone(Backbone):
    """
    Whisper returns only transformer block outputs (the conv stem is not
    exposed in hidden_states). n_layers here equals num_hidden_layers,
    without a +1.
    """

    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = "cuda"):
        super().__init__()
        from transformers import WhisperModel, WhisperFeatureExtractor

        self._model_name = model_name
        self._device = device
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False

        self._n_layers = self.model.config.encoder_layers
        self._hidden_dim = self.model.config.d_model
        self._checkpoint_hash = _hash_config(self.model.config.to_dict(), extra=model_name)

    @property
    def n_layers(self) -> int: return self._n_layers
    @property
    def hidden_dim(self) -> int: return self._hidden_dim
    @property
    def sample_rate(self) -> int: return 16000
    @property
    def backbone_id(self) -> str: return self._model_name.replace("/", "_")
    @property
    def checkpoint_hash(self) -> str: return self._checkpoint_hash

    @torch.no_grad()
    def forward(self, audio, attention_mask=None):
        audio_np = audio.detach().cpu().numpy()
        inputs = self.feature_extractor(
            list(audio_np),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        mel = inputs.input_features.to(self._device, dtype=torch.float16)

        encoder = self.model.encoder
        out = encoder(mel, output_hidden_states=True, return_dict=True)
        hidden = torch.stack(out.hidden_states[1:], dim=1).to(torch.float16)

        B, _, T_out, _ = hidden.shape
        output_mask = torch.ones(B, T_out, dtype=torch.bool, device=hidden.device)
        return hidden, output_mask


_REGISTRY = {
    "wavlm-base-plus": lambda device: WavLMBackbone("microsoft/wavlm-base-plus", device),
    "wavlm-large": lambda device: WavLMBackbone("microsoft/wavlm-large", device),
    "hubert-large": lambda device: HuBERTBackbone("facebook/hubert-large-ll60k", device),
    "whisper-large-v3": lambda device: WhisperEncoderBackbone("openai/whisper-large-v3", device),
}


def build_backbone(name: str, device: str = "cuda") -> Backbone:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backbone {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](device)
