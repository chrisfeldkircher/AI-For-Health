"""
Phoneme labelling via `facebook/wav2vec2-xlsr-53-espeak-cv-ft`.

Multilingual IPA phoneme CTC, fine-tuned on CommonVoice transcripts converted
with espeak-ng. Produces per-frame argmax phoneme IDs at 50 Hz (20 ms), which
matches WavLM's frame rate exactly (same conv stride 320 on 16 kHz input).
That alignment is what makes `phoneme_labels[t]` indexable into the frame
cache `frames/L{N}/{stem}.pt[t]` without any resampling.

No CTC decoding is performed. For per-frame pooling we want the model's
per-frame belief (including blank/silence frames), not a collapsed phone
sequence — downstream, the category map folds the blank token into the
"silence" category.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


PHONEME_CATEGORIES = ("silence", "vowel", "nasal", "fricative", "plosive", "approximant")
CAT_SILENCE, CAT_VOWEL, CAT_NASAL, CAT_FRICATIVE, CAT_PLOSIVE, CAT_APPROXIMANT = range(6)

# IPA character sets (disjoint). Categorisation is by the first phonetically
# meaningful character in a multi-char token: "tʃ" → plosive via 't' (the
# stop release is the acoustically dominant landmark for affricates); "aɪ"
# → vowel via 'a'; "nd" → nasal via 'n'.
_NASALS       = set("mnŋɲɱɳɴ")
_PLOSIVES     = set("pbtdkgqʔɢcɟʈɖ")
_FRICATIVES   = set("fvszʃʒxɣhçʝðθħʕɦɸβχʁʂʐɕʑʍ")
_APPROXIMANTS = set("lrɹɾʎɭʟɺɽʋjwɥɰɻ")
_VOWELS       = set("aeiouɑɐɒæəɛɜɨɪʊʌœɔɘɵɶʉʏyɤɯøɞ")

_SPECIAL_TOKENS = {"", "<pad>", "<s>", "</s>", "<unk>", "|",
                   "[PAD]", "[UNK]", "[CLS]", "[SEP]"}


def classify_token(token: str) -> int:
    """IPA token string → category int in [0, 6)."""
    if token in _SPECIAL_TOKENS or not token.strip():
        return CAT_SILENCE
    for ch in token:
        if ch in _NASALS:       return CAT_NASAL
        if ch in _PLOSIVES:     return CAT_PLOSIVE
        if ch in _FRICATIVES:   return CAT_FRICATIVE
        if ch in _APPROXIMANTS: return CAT_APPROXIMANT
        if ch in _VOWELS:       return CAT_VOWEL
    return CAT_SILENCE


def build_category_map(vocab: dict) -> dict:
    """Build {id_to_category: [cat per token_id], names: [...]} from a HF tokenizer vocab."""
    n = max(vocab.values()) + 1
    cats = [CAT_SILENCE] * n
    for tok, tid in vocab.items():
        cats[tid] = classify_token(tok)
    return {"names": list(PHONEME_CATEGORIES), "id_to_category": cats}


@torch.no_grad()
def extract_phonemes(
    dataset: Dataset,
    cache_root: str,
    model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    device: str = "cuda",
    batch_size: int = 4,
    num_workers: int = 0,
    skip_existing: bool = True,
    progress: bool = True,
) -> dict:
    """
    Writes:
      {cache_root}/phoneme_labels/{stem}.pt        [T_valid] int16  (per-frame argmax IDs)
      {cache_root}/phoneme_labels/vocab.json       id → IPA token
      {cache_root}/phoneme_labels/categories.json  id → category int + names

    Padding is stripped per-file via the model's CNN output-length computation.
    """
    from transformers import Wav2Vec2ForCTC
    from huggingface_hub import hf_hub_download
    from .extract import _pad_collate

    out_dir = Path(cache_root) / "phoneme_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip the HF tokenizer stack entirely for this checkpoint — across the
    # versions we tested, Wav2Vec2Processor / AutoTokenizer /
    # Wav2Vec2PhonemeCTCTokenizer all mis-resolve the espeak-cv-ft config
    # (some return a bool instead of an instance). We only need the id→IPA
    # map, which is just vocab.json on the Hub.
    vocab_path = hf_hub_download(repo_id=model_name, filename="vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        vocab: dict = json.load(f)  # {token_str: id}

    model = (Wav2Vec2ForCTC
             .from_pretrained(model_name, torch_dtype=torch.float16)
             .eval()
             .to(device))
    for p in model.parameters():
        p.requires_grad = False

    # Straight argmax including CTC blank. Blank (conventionally `<pad>` id 0)
    # wins ~75% of frames — those get mapped to CAT_SILENCE via categories.json.
    # We deliberately do NOT mask blank and re-argmax: the runner-up on
    # blank-winning frames is a small set of filler tokens (`t`, `ɾ`, `j` ~50%
    # of non-blank mass on a sample stem), which would contaminate the plosive
    # and approximant buckets with non-phoneme frames. Honest silence + clean
    # per-category phoneme frames beats dense but contaminated labels.
    id2tok = {v: k for k, v in vocab.items()}
    (out_dir / "vocab.json").write_text(
        json.dumps(id2tok, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "categories.json").write_text(
        json.dumps(build_category_map(vocab), ensure_ascii=False, indent=2),
        encoding="utf-8")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_pad_collate,
    )
    if progress:
        try:
            from tqdm.auto import tqdm
            short = model_name.split("/")[-1]
            loader = tqdm(loader, desc=f"phonemes[{short}]")
        except ImportError:
            pass

    n_written = 0
    for batch in loader:
        file_names = batch["file_name"]
        remaining: list[tuple[int, str, Path]] = []
        for i, fn in enumerate(file_names):
            stem = fn[:-4] if fn.endswith(".wav") else fn
            target = out_dir / f"{stem}.pt"
            if skip_existing and target.exists():
                continue
            remaining.append((i, stem, target))
        if not remaining:
            continue

        # Normalise per-sample over valid frames in fp32 (wav2vec2 feature
        # extractor's default behaviour), then cast to fp16 for the model.
        audio = batch["audio"].to(device, dtype=torch.float32)
        attn  = batch["attention_mask"].to(device, dtype=torch.long)

        m = attn.to(torch.float32)
        n = m.sum(-1, keepdim=True).clamp(min=1.0)
        mean = (audio * m).sum(-1, keepdim=True) / n
        centered = (audio - mean) * m
        var = (centered ** 2).sum(-1, keepdim=True) / n
        audio_norm = ((audio - mean) / (var + 1e-7).sqrt()) * m
        audio_norm = audio_norm.to(torch.float16)

        logits = model(input_values=audio_norm, attention_mask=attn).logits  # [B, T_out, V]
        pred = logits.argmax(-1)                                                # [B, T_out]

        in_lens = attn.sum(-1)
        out_lens = model.wav2vec2._get_feat_extract_output_lengths(in_lens).long()

        for i, stem, target in remaining:
            T_valid = int(out_lens[i].item())
            labels = pred[i, :T_valid].to(torch.int16).cpu().contiguous()
            torch.save(labels, target)
            n_written += 1

    return {
        "n_written": n_written,
        "vocab_size": len(vocab),
        "categories": list(PHONEME_CATEGORIES),
        "out_dir": str(out_dir),
    }
