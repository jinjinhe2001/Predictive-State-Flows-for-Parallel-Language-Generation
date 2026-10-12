"""Generation-quality metrics.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

REPETITION_PREFIX_LENGTH = 256
REPETITION_NGRAM_ORDERS = (2, 4, 8)
HARD_LOOP_NGRAM_ORDER = 8
HARD_LOOP_MIN_COUNT = 3


def _sample_repetition_metrics(ids: Tensor, prefix_length: int) -> dict[str, float]:
    """Measure exact n-gram repetition on one fixed-length token prefix."""
    ids = ids[:prefix_length]
    length = ids.numel()
    result: dict[str, float] = {}
    for order in REPETITION_NGRAM_ORDERS:
        if length < order:
            result[f"rep{order}"] = 0.0
            continue
        ngrams = ids.unfold(0, order, 1)
        result[f"rep{order}"] = (
            1.0 - torch.unique(ngrams, dim=0).shape[0] / ngrams.shape[0]
        )

    order = HARD_LOOP_NGRAM_ORDER
    if length < order:
        result["loop_coverage"] = 0.0
        result["has_hard_loop"] = 0.0
        return result
    ngrams = ids.unfold(0, order, 1)
    _, inverse, counts = torch.unique(
        ngrams, dim=0, return_inverse=True, return_counts=True
    )
    hard_loop_starts = counts[inverse] >= HARD_LOOP_MIN_COUNT
    coverage = torch.zeros(length, dtype=torch.bool, device=ids.device)
    for start in hard_loop_starts.nonzero(as_tuple=False).flatten().tolist():
        coverage[start : start + order] = True
    result["loop_coverage"] = coverage.float().mean().item()
    result["has_hard_loop"] = float(hard_loop_starts.any())
    return result


def diversity_metrics(tokens: Tensor, lengths: Tensor, pad_id: int) -> dict:
    """Cheap, model-free stats over generated ids.

    Args:
        tokens:  [B, L] right-padded generated ids.
        lengths: [B] true (pre-pad) length of each sample.
        pad_id:  padding id to ignore.

    Returns:
        gen_len                     mean sequence length (tokens),
        gen_empty_rate              fraction of empty samples,
        gen_distinct                mean per-sample unique/total token ratio,
        gen_unigram_entropy         corpus-level unigram entropy in nats,
        gen_rep{2,4,8}_256          repeated n-gram fractions in the first 256 tokens,
        gen_loop8_coverage_256      mean token coverage of repeated 8-grams,
        gen_loop8_sample_rate_256   fraction of samples containing a hard 8-gram loop.
    """
    B, L = tokens.shape
    device = tokens.device
    keep = torch.arange(L, device=device)[None, :] < lengths.unsqueeze(1)  # [B, L]

    mean_len = lengths.float().mean().item() if B else 0.0

    distinct = []
    repetition = []
    counts: dict[int, int] = {}
    for b in range(B):
        ids = tokens[b][keep[b]]
        if ids.numel() == 0:
            continue
        distinct.append(ids.unique().numel() / ids.numel())
        repetition.append(_sample_repetition_metrics(ids, REPETITION_PREFIX_LENGTH))
        for tid in ids.tolist():
            counts[tid] = counts.get(tid, 0) + 1

    mean_distinct = sum(distinct) / len(distinct) if distinct else 0.0
    empty_rate = 1.0 - len(repetition) / B if B else 0.0

    total = sum(counts.values())
    if total > 0:
        probs = torch.tensor([c / total for c in counts.values()])
        entropy = float(-(probs * probs.log()).sum())
    else:
        entropy = 0.0

    metrics = {
        "gen_len": mean_len,
        "gen_empty_rate": empty_rate,
        "gen_distinct": mean_distinct,
        "gen_unigram_entropy": entropy,
    }
    for order in REPETITION_NGRAM_ORDERS:
        metrics[f"gen_rep{order}_256"] = (
            sum(row[f"rep{order}"] for row in repetition) / len(repetition)
            if repetition else 0.0
        )
    metrics["gen_loop8_coverage_256"] = (
        sum(row["loop_coverage"] for row in repetition) / len(repetition)
        if repetition else 0.0
    )
    metrics["gen_loop8_sample_rate_256"] = (
        sum(row["has_hard_loop"] for row in repetition) / len(repetition)
        if repetition else 0.0
    )
    return metrics


def repetition_metrics(token_lists: list[list[int]],
                       prefix_length: int = REPETITION_PREFIX_LENGTH) -> dict[str, float]:
    """Same repetition/loop stats as `diversity_metrics`, over ragged id lists.

    For callers that already hold one token list per sample (scripts/eval_gen.py)
    rather than a right-padded [B, L] tensor. Empty samples are skipped.
    """
    rows: list[dict[str, float]] = []
    distinct: list[float] = []
    for ids in token_lists:
        if not ids:
            continue
        row = torch.tensor(ids, dtype=torch.long)
        distinct.append(row.unique().numel() / row.numel())
        rows.append(_sample_repetition_metrics(row, prefix_length))

    out = {f"rep{order}": 0.0 for order in REPETITION_NGRAM_ORDERS}
    out["loop8_coverage"] = 0.0
    out["loop8_sample_rate"] = 0.0
    out["distinct"] = 0.0
    if not rows:
        return out

    for order in REPETITION_NGRAM_ORDERS:
        out[f"rep{order}"] = sum(r[f"rep{order}"] for r in rows) / len(rows)
    out["loop8_coverage"] = sum(r["loop_coverage"] for r in rows) / len(rows)
    out["loop8_sample_rate"] = sum(r["has_hard_loop"] for r in rows) / len(rows)
    out["distinct"] = sum(distinct) / len(distinct)
    return out


class PerplexityScorer:
    """Lazy wrapper around a frozen causal LM (default GPT-2) for generative PPL."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def perplexity(self, texts: list[str], max_length: int = 256, batch_size: int = 32) -> float:
        """Corpus-level token perplexity of `texts` under the frozen LM.

        Empty strings are dropped; returns NaN if nothing is scorable.
        """
        texts = [t for t in texts if t.strip()]
        if not texts:
            return float("nan")

        total_nll = 0.0
        total_tok = 0.0
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            logits = self.model(input_ids=ids, attention_mask=attn).logits

            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = ids[:, 1:].reshape(-1)
            shift_mask = attn[:, 1:].reshape(-1).float()

            nll = F.cross_entropy(shift_logits, shift_labels, reduction="none") * shift_mask
            total_nll += float(nll.sum())
            total_tok += float(shift_mask.sum())

        if total_tok == 0:
            return float("nan")
        return math.exp(total_nll / total_tok)


def generation_metrics(
    tokens: Tensor,
    lengths: Tensor,
    pad_id: int,
    tokenizer=None,
    scorer: Optional[PerplexityScorer] = None,
    ppl_max_length: int = 256,
) -> tuple[dict, list[str]]:
    """Bundle diversity stats (+ optional Gen PPL) and decoded texts.

    Returns (metrics, texts). `texts` is empty when no tokenizer is given.
    """
    metrics = diversity_metrics(tokens, lengths, pad_id)

    texts: list[str] = []
    if tokenizer is not None:
        seqs = [tokens[b, : int(lengths[b])].tolist() for b in range(tokens.size(0))]
        texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        if scorer is not None:
            metrics["gen_ppl"] = scorer.perplexity(texts, max_length=ppl_max_length)

    return metrics, texts
