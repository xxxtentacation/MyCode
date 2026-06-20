from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Sequence, Tuple, Union
import random
import re

import torch
from torch.utils.data import Dataset

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
_KV_NOISE_RE = re.compile(r"\b(?:cmd|task|job|seq|action|id|speed|temp|noise|sensor|ts|mode)\s*=\s*[^\s]+")
_DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_PUNCT_RE = re.compile(r"[^a-z\s]")

_TYPO_MAP = {
    "relase": "release",
    "griper": "gripper",
    "rigth": "right",
    "lft": "left",
    "mov": "move",
}

_STOP_WORDS = {
    "please",
    "now",
    "quickly",
    "kindly",
    "immediately",
    "robot",
    "arm",
    "channela",
    "modex",
    "do",
    "auto",
}

# Domain-specific synonym groups for data augmentation in robotic-arm commands
_SYNONYM_GROUPS: List[List[str]] = [
    ["move", "shift", "slide", "go", "travel"],
    ["grab", "grasp", "grip", "hold", "catch", "pick"],
    ["release", "drop", "free", "let", "loosen"],
    ["left", "leftward", "leftwards"],
    ["right", "rightward", "rightwards"],
    ["up", "upward", "upwards", "raise", "lift"],
    ["down", "downward", "downwards", "lower"],
    ["object", "item", "thing", "target", "piece"],
    ["execute", "run", "perform", "do"],
    ["quickly", "fast", "rapidly", "swiftly"],
    ["slowly", "gradually", "gently"],
    ["stop", "halt", "cease", "end"],
    ["rotate", "turn", "spin", "twist"],
    ["push", "press", "shove"],
    ["pull", "drag", "draw", "tug"],
]

_SYNONYM_MAP: Dict[str, str] = {}
for _group in _SYNONYM_GROUPS:
    for _i, _src in enumerate(_group):
        _SYNONYM_MAP[_src] = random.choice([w for j, w in enumerate(_group) if j != _i]) if len(_group) > 1 else _src


def _get_synonym(token: str) -> str:
    """Return a random synonym from the same group, or the original token."""
    for group in _SYNONYM_GROUPS:
        if token in group:
            others = [w for w in group if w != token]
            if others:
                return random.choice(others)
    return token


def augment_text(text: str, alpha_sr: float = 0.15, alpha_rd: float = 0.10, alpha_rs: float = 0.05) -> str:
    """EDA-style text augmentation for short command texts.

    Args:
        text: Preprocessed text string (space-separated tokens).
        alpha_sr: Probability of replacing each token with a synonym.
        alpha_rd: Probability of randomly deleting each token.
        alpha_rs: Probability of randomly swapping two adjacent tokens (per token).

    Returns:
        Augmented text string.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text

    # Synonym replacement
    for i in range(len(tokens)):
        if random.random() < alpha_sr:
            tokens[i] = _get_synonym(tokens[i])

    # Random deletion (always keep at least 2 tokens)
    if len(tokens) > 2:
        tokens = [t for t in tokens if random.random() >= alpha_rd]
        if len(tokens) < 2:
            tokens = tokens[:2]  # shouldn't happen with low alpha_rd

    # Random swap (adjacent tokens)
    for i in range(len(tokens) - 1):
        if random.random() < alpha_rs:
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]

    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower().strip())


def preprocess_text(text: str) -> str:
    cleaned = text.lower().strip().replace("->", " ").replace("<", " ")
    cleaned = _KV_NOISE_RE.sub(" ", cleaned)

    for src, dst in _TYPO_MAP.items():
        cleaned = cleaned.replace(src, dst)

    cleaned = _DIGIT_RE.sub(" ", cleaned)
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    tokens = [tok for tok in cleaned.split() if tok and tok not in _STOP_WORDS]
    return " ".join(tokens)


def read_labeled_text(path: Union[str, Path]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            text, label = line.split("\t")
            rows.append((text, label))
    return rows


@dataclass
class VocabBundle:
    token_to_id: Dict[str, int]
    label_to_id: Dict[str, int]

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {v: k for k, v in self.label_to_id.items()}


def build_vocab(samples: Sequence[Tuple[str, str]], min_freq: int = 1, use_preprocess: bool = True) -> VocabBundle:
    token_freq: Dict[str, int] = {}
    labels = sorted({label for _, label in samples})

    for text, _ in samples:
        corpus_text = preprocess_text(text) if use_preprocess else text
        for tok in tokenize(corpus_text):
            token_freq[tok] = token_freq.get(tok, 0) + 1

    token_to_id = {"<pad>": 0, "<unk>": 1}
    for tok, freq in sorted(token_freq.items()):
        if freq >= min_freq and tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return VocabBundle(token_to_id=token_to_id, label_to_id=label_to_id)


class VocabLike(Protocol):
    token_to_id: Dict[str, int]
    label_to_id: Dict[str, int]


class TextCommandDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, str]],
        vocab: VocabLike,
        use_preprocess: bool = True,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.vocab = vocab
        self.use_preprocess = use_preprocess
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        text, label = self.samples[idx]
        encoded_text = preprocess_text(text) if self.use_preprocess else text
        if self.augment:
            encoded_text = augment_text(encoded_text)
        token_ids = [
            self.vocab.token_to_id.get(tok, self.vocab.token_to_id["<unk>"])
            for tok in tokenize(encoded_text)
        ]
        if not token_ids:
            token_ids = [self.vocab.token_to_id["<unk>"]]
        return token_ids, self.vocab.label_to_id[label]


def collate_batch(batch: Sequence[Tuple[List[int], int]]) -> Dict[str, torch.Tensor]:
    seqs, labels = zip(*batch)
    max_len = max(len(seq) for seq in seqs)

    input_ids = torch.zeros((len(seqs), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.float32)

    for i, seq in enumerate(seqs):
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :seq_len] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }

