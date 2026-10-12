"""A synthetic char-level corpus, so the pipeline runs end to end with no downloads.

Sentences come from a tiny grammar: small enough that a few-million-parameter
model reaches near-perfect reconstruction in minutes on CPU, structured enough
that prior samples are readable and their failures are easy to interpret.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from psf.config import Config
from psf.utils import dist

PAD_ID, BOB_ID, EOS_ID = 0, 1, 2
_SPECIALS = ("<pad>", "<bob>", "<eos>")

_DET = ("the", "a", "one")
_ADJ = ("red", "small", "quick", "quiet", "old", "bright")
_NOUN = ("fox", "robot", "river", "child", "garden", "engine")
_VERB = ("sees", "finds", "builds", "follows", "paints", "carries")
_PLACE = ("in the garden", "near the river", "by the wall", "under the tree")


class CharTokenizer:
    """Character vocabulary with three reserved ids: PAD, BOB, EOS."""

    def __init__(self, chars: str) -> None:
        self.itos = list(_SPECIALS) + sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(
            self.itos[int(i)] for i in ids
            if not (skip_special_tokens and int(i) < len(_SPECIALS))
        )


def _sentence(rng: random.Random) -> str:
    parts = [
        rng.choice(_DET), rng.choice(_ADJ), rng.choice(_NOUN), rng.choice(_VERB),
        rng.choice(_DET), rng.choice(_ADJ), rng.choice(_NOUN),
    ]
    text = " ".join(parts)
    if rng.random() < 0.5:
        text += " " + rng.choice(_PLACE)
    return text + " ."


def _document(rng: random.Random, budget: int) -> str:
    """Pack whole sentences until the next one would not fit in `budget` chars."""
    text = _sentence(rng)
    while rng.random() < 0.6:
        nxt = text + " " + _sentence(rng)
        if len(nxt) > budget:
            break
        text = nxt
    return text[:budget]


class _ToyDataset(Dataset):
    def __init__(self, tokens: Tensor, mask: Tensor) -> None:
        self.tokens, self.mask = tokens, mask

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.tokens[idx][self.mask[idx]]}


def build_corpus(cfg: Config) -> tuple[CharTokenizer, list[str]]:
    rng = random.Random(cfg.data.toy_seed)
    budget = cfg.max_seq_len - 1                    # one slot reserved for EOS
    total = cfg.data.toy_train_docs + cfg.data.toy_val_docs
    docs = [_document(rng, budget) for _ in range(total)]
    return CharTokenizer("".join(docs)), docs


def toy_tokenizer(cfg: Config) -> CharTokenizer:
    """Fit the char vocabulary and copy the special ids into the config."""
    tokenizer, _ = build_corpus(cfg)
    cfg.data.vocab_size = len(tokenizer)
    cfg.data.pad_token_id = PAD_ID
    cfg.data.bob_token_id = BOB_ID
    cfg.data.eos_token_id = EOS_ID
    return tokenizer


def _encode(docs: list[str], tokenizer: CharTokenizer, seq_len: int) -> _ToyDataset:
    tokens = torch.full((len(docs), seq_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros(len(docs), seq_len, dtype=torch.bool)
    for row, doc in enumerate(docs):
        ids = tokenizer.encode(doc)[: seq_len - 1] + [EOS_ID]
        tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row, : len(ids)] = True
    return _ToyDataset(tokens, mask)


def toy_dataloaders(cfg: Config, collate_fn) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders over the synthetic corpus, using the shared collate."""
    tokenizer, docs = build_corpus(cfg)
    n_train = cfg.data.toy_train_docs
    train_ds = _encode(docs[:n_train], tokenizer, cfg.max_seq_len)
    val_ds = _encode(docs[n_train:], tokenizer, cfg.max_seq_len)
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=True)
        if dist.is_distributed() else None
    )
    return (
        DataLoader(train_ds, batch_size=cfg.train.batch_size,
                   shuffle=train_sampler is None, sampler=train_sampler,
                   drop_last=True, collate_fn=collate_fn),
        DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                   collate_fn=collate_fn),
    )
