import os
import csv
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
import torch

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False

LABEL_MAP = {"C": 1, "NC": 0}

Split = Literal["train", "devel", "test"]
FeatureSet = Literal["compare", "egemaps", "egemapsv02"]


def _load_audio(wav_path: str) -> tuple[np.ndarray, int]:
    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, sr


def _compute_melspectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power_to_db: bool = True,
) -> np.ndarray:
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for mel spectrograms — pip install librosa")
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    if power_to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)


def _compute_opensmile_features(
    wav_path: str,
    feature_set: FeatureSet = "egemapsv02",
) -> np.ndarray:
    if not OPENSMILE_AVAILABLE:
        raise ImportError("opensmile is required — pip install opensmile")

    smile_feature_set = {
        "compare": opensmile.FeatureSet.ComParE_2016,
        "egemaps": opensmile.FeatureSet.eGeMAPSv01a,
        "egemapsv02": opensmile.FeatureSet.eGeMAPSv02,
    }[feature_set]

    smile = opensmile.Smile(
        feature_set=smile_feature_set,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(wav_path)
    return features.values[0].astype(np.float32)


def _cache_path(cache_dir: str, wav_path: str, suffix: str) -> str:
    key = hashlib.md5(wav_path.encode()).hexdigest()
    return os.path.join(cache_dir, f"{key}_{suffix}.pkl")


def _load_or_compute(cache_dir: Optional[str], cache_key: str, compute_fn):
    if cache_dir:
        path = os.path.join(cache_dir, cache_key + ".pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    result = compute_fn()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(result, f)
    return result


class AudioDataset(Dataset):
    """
    ComParE 2017 Cold sub-challenge dataset.

    Returns per sample:
        - raw audio waveform (float32, normalised to [-1, 1])
        - mel spectrogram (n_mels x T) — requires librosa
        - opensmile functionals (1-D vector) — requires opensmile
        - label (int: 1=Cold, 0=Non-cold, -1=unlabelled test set)

    Parameters
    ----------
    data_dir : str
        Path to the ComParE2017_Cold_4students directory.
    split : 'train' | 'devel' | 'test'
        Which subset to load.
    use_mel : bool
        Whether to compute mel spectrograms.
    use_opensmile : bool
        Whether to extract OpenSMILE functionals.
    opensmile_feature_set : 'compare' | 'egemaps' | 'egemapsv02'
        Which OpenSMILE feature set to use.
    mel_config : dict, optional
        Override any mel spectrogram hyper-parameters:
        n_mels, n_fft, hop_length, fmin, fmax, power_to_db.
    pad_or_truncate_secs : float, optional
        If set, pads/truncates all clips to this length in seconds before
        computing the mel spectrogram (ensures fixed-size tensors).
    cache_dir : str, optional
        Directory to cache computed features so they are only computed once.
    transform : callable, optional
        Extra transform applied to the raw waveform.
    """

    def __init__(
        self,
        data_dir: str,
        split: Split = "train",
        use_mel: bool = True,
        use_opensmile: bool = False,
        opensmile_feature_set: FeatureSet = "egemapsv02",
        mel_config: Optional[dict] = None,
        pad_or_truncate_secs: Optional[float] = None,
        cache_dir: Optional[str] = None,
        transform=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.use_mel = use_mel
        self.use_opensmile = use_opensmile
        self.opensmile_feature_set = opensmile_feature_set
        self.mel_config = mel_config or {}
        self.pad_or_truncate_secs = pad_or_truncate_secs
        self.cache_dir = cache_dir
        self.transform = transform

        self.wav_dir = os.path.join(data_dir, "wav")
        self.label_path = os.path.join(data_dir, "lab", "ComParE2017_Cold.tsv")

        self._labels: dict[str, int] = {}
        self._load_labels()

        self.file_list = sorted(
            f for f in os.listdir(self.wav_dir)
            if f.endswith(".wav") and f.startswith(split + "_")
        )

        if use_opensmile and not OPENSMILE_AVAILABLE:
            raise ImportError("opensmile is not installed — pip install opensmile")
        if use_mel and not LIBROSA_AVAILABLE:
            raise ImportError("librosa is not installed — pip install librosa")

    def _load_labels(self):
        with open(self.label_path, "r") as fd:
            reader = csv.reader(fd, delimiter="\t", quotechar='"')
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    self._labels[row[0]] = LABEL_MAP.get(row[1].strip(), -1)

    def __len__(self) -> int:
        return len(self.file_list)

    def _pad_or_truncate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        target_len = int(self.pad_or_truncate_secs * sr)
        if len(audio) >= target_len:
            return audio[:target_len]
        return np.pad(audio, (0, target_len - len(audio)))

    def __getitem__(self, idx: int) -> dict:
        file_name = self.file_list[idx]
        wav_path = os.path.join(self.wav_dir, file_name)
        label = self._labels.get(file_name, -1)

        audio, sr = _load_audio(wav_path)

        if self.transform:
            audio = self.transform(audio)

        audio_for_features = audio
        if self.pad_or_truncate_secs is not None:
            audio_for_features = self._pad_or_truncate(audio, sr)

        sample = {
            "file_name": file_name,
            "audio": torch.from_numpy(audio_for_features),
            "sample_rate": sr,
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.use_mel:
            mel_key = f"mel_{self.split}_{file_name}_{str(self.mel_config)}"
            mel_key = hashlib.md5(mel_key.encode()).hexdigest()

            def _compute_mel():
                return _compute_melspectrogram(audio_for_features, sr, **self.mel_config)

            mel = _load_or_compute(self.cache_dir, mel_key, _compute_mel)
            sample["mel"] = torch.from_numpy(mel)

        if self.use_opensmile:
            smile_key = hashlib.md5(
                f"smile_{self.opensmile_feature_set}_{wav_path}".encode()
            ).hexdigest()

            def _compute_smile():
                return _compute_opensmile_features(wav_path, self.opensmile_feature_set)

            smile_feats = _load_or_compute(self.cache_dir, smile_key, _compute_smile)
            sample["opensmile"] = torch.from_numpy(smile_feats)

        return sample

    def get_label(self, idx: int) -> int:
        return self._labels.get(self.file_list[idx], -1)

    def get_file_name(self, idx: int) -> str:
        return self.file_list[idx]

    def get_sample_rate(self, idx: int) -> int:
        _, sr = wavfile.read(os.path.join(self.wav_dir, self.file_list[idx]))
        return sr

    @property
    def opensmile_dim(self) -> int:
        if not self.use_opensmile:
            return 0
        return {
            "compare": 6373,
            "egemaps": 88,
            "egemapsv02": 88,
        }[self.opensmile_feature_set]

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(
            [self._labels[f] for f in self.file_list if f in self._labels],
            minlength=2,
        )
        total = counts.sum()
        weights = total / (2.0 * counts.clip(min=1))
        return torch.tensor(weights, dtype=torch.float32)
