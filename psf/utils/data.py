"""Pre-tokenized HF dataset -> fixed-canvas token batches.

Run ``python -m scripts.download_data`` once, then point
``config.data.dataset_path`` at the resulting directory.  Every document is
truncated or padded to the fixed canvas ``L = K * S``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from psf.config import Config
from psf.utils import dist
from psf.utils.misc import cycle


@dataclass
class Batch:
    tokens: Tensor   # [B, L] int64, PAD after the document
    mask: Tensor     # [B, L] bool, True on real tokens

    def to(self, device) -> "Batch":
        return Batch(
            tokens=self.tokens.to(device, non_blocking=True),
            mask=self.mask.to(device, non_blocking=True),
        )


def load_split(path: str, split: Optional[str] = None):
    """Load one split of a pre-tokenized dataset from local disk or the hub."""
    from datasets import DatasetDict, load_dataset, load_from_disk

    try:
        ds = load_from_disk(path)
    except FileNotFoundError:
        ds = load_dataset(path)

    if isinstance(ds, DatasetDict):
        if split is not None:
            ds = ds.get(split)
        elif len(ds) == 1:
            ds = next(iter(ds.values()))
        else:
            raise ValueError(f"dataset has splits {list(ds)}; specify one")
    elif split not in (None, "train"):
        ds = None

    if ds is not None:
        ds.set_format(type="numpy", columns=["input_ids"])
    return ds


def collate(items: list[dict], pad_id: int, max_len: int) -> Batch:
    """Pack variable-length ``input_ids`` into the fixed [B, L] canvas."""
    tokens = torch.full((len(items), max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(items), dtype=torch.long)
    for i, item in enumerate(items):
        ids = item["input_ids"][:max_len]
        tokens[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        lengths[i] = len(ids)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    return Batch(tokens=tokens, mask=mask)


def load_tokenizer(cfg: Config):
    """Load the tokenizer and copy its vocab size / special ids into the config.

    Returns None when transformers or the weights are unavailable; training
    still runs, only decoded text turns into raw token ids.
    """
    if cfg.data.source == "toy":
        from psf.utils.toy_data import toy_tokenizer

        return toy_tokenizer(cfg)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    except Exception as exc:
        print(f"tokenizer '{cfg.data.tokenizer_name}' unavailable ({exc}); "
              "keeping the configured vocab size and printing token ids")
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.data.vocab_size = len(tokenizer)
    cfg.data.pad_token_id = tokenizer.pad_token_id
    cfg.data.eos_token_id = tokenizer.eos_token_id
    # T5 has no BOS; the pad slot doubles as the decoder's begin-of-block token.
    cfg.data.bob_token_id = (
        tokenizer.bos_token_id if tokenizer.bos_token_id is not None
        else tokenizer.pad_token_id
    )
    return tokenizer


def decode(tokenizer, ids) -> str:
    """Token ids -> text, falling back to raw ids when there is no tokenizer."""
    if tokenizer is None:
        return " ".join(str(int(i)) for i in ids)
    return tokenizer.decode(ids, skip_special_tokens=True)


def make_dataloaders(cfg: Config) -> tuple[DataLoader, Optional[DataLoader]]:
    """Train / val loaders over the fixed canvas; val is carved off train if absent.

    Under torchrun the train set is sharded across ranks, so
    ``cfg.train.batch_size`` is the per-process batch size.
    """
    collate_fn = partial(
        collate, pad_id=cfg.data.pad_token_id, max_len=cfg.max_seq_len
    )
    if cfg.data.source == "toy":
        from psf.utils.toy_data import toy_dataloaders

        return toy_dataloaders(cfg, collate_fn)

    train_ds = load_split(cfg.data.dataset_path, "train")
    val_ds = load_split(cfg.data.dataset_path, "validation")
    if train_ds is None:
        raise FileNotFoundError(
            f"no train split at {cfg.data.dataset_path!r}; "
            "run `python -m scripts.download_data` first"
        )

    if val_ds is None and cfg.data.val_frac > 0:
        import numpy as np

        # Seeded random 90/10-style split; identical on every rank, no cache files.
        perm = np.random.default_rng(cfg.data.split_seed).permutation(len(train_ds))
        n_val = min(int(len(perm) * cfg.data.val_frac), cfg.data.val_max_docs,
                    len(perm) - 1)
        if n_val > 0:
            val_ds = train_ds.select(perm[:n_val], keep_in_memory=True)
            train_ds = train_ds.select(perm[n_val:], keep_in_memory=True)

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=True)
        if dist.is_distributed() else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn,
        )
    return train_loader, val_loader


class RolloutBankDataset(Dataset):
    """Wraps an on-disk bank of rollout-generated documents (see
    ``scripts/sample_rollout_bank.py``) in the same ``Batch`` shape the real
    dataloader produces, so it's a drop-in alternative training-data source.
    """

    def __init__(self, bank_path: str) -> None:
        bank = torch.load(bank_path, map_location="cpu", mmap=True)
        self.tokens = bank["tokens"]      # [N, L] int16
        self.lengths = bank["lengths"]    # [N] int32

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        n = int(self.lengths[idx])
        return {"input_ids": self.tokens[idx, :n].to(torch.int64).numpy()}


def make_rollout_loader(bank_path: str, cfg: Config) -> DataLoader:
    """Same DDP-sharded/collated shape as the real ``train_loader``."""
    ds = RolloutBankDataset(bank_path)
    collate_fn = partial(collate, pad_id=cfg.data.pad_token_id, max_len=cfg.max_seq_len)
    sampler = (DistributedSampler(ds, shuffle=True, drop_last=True)
              if dist.is_distributed() else None)
    return DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=sampler is None,
        sampler=sampler, drop_last=True, num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )


class MixedBatchLoader:
    """Infinite iterator: yields batches from ``alt_loader`` with probability
    ``p``, else from ``base_loader`` -- both internally re-cycled (reshuffled
    each pass) as needed. Drop-in for whatever ``train_loader`` a trainer
    expects (only ``__iter__`` and ``dataset`` -- for the printed doc count --
    are read elsewhere).
    """

    def __init__(self, base_loader: DataLoader, alt_loader: DataLoader,
                p: float, seed: int = 0) -> None:
        self.dataset = base_loader.dataset   # only used for len() logging
        self._base, self._alt, self._p = base_loader, alt_loader, p
        self._rng = random.Random(seed + dist.rank())

    def __iter__(self):
        base_iter, alt_iter = cycle(self._base), cycle(self._alt)
        while True:
            yield next(alt_iter) if self._rng.random() < self._p else next(base_iter)
