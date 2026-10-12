"""GPT-2 (AR reference) generation latency under the same protocol as our probes.

1023 new tokens from BOS (a 1024-token document), KV cache, sampling, eager tf32; batch 1 and 50;
warmup + averaged iterations with cuda.synchronize. Optionally the HF static-cache + torch.compile
(CUDA graphs) path for a fair comparison with scripts/latency_graphed.py.
  CUDA_VISIBLE_DEVICES=4 python -m scripts.latency_gpt2 --model gpt2 --batches 1 50
"""
from __future__ import annotations

import argparse
import time

import torch


def timed(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 50])
    p.add_argument("--new-tokens", type=int, default=1023)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--compile", action="store_true", help="also time static-cache + torch.compile")
    p.add_argument("--flops", action="store_true", help="count FLOPs of one batch-1 generation (FlopCounterMode)")
    p.add_argument("--graph", action="store_true",
                   help="also time a manual CUDA-graph decode (StaticCache, one captured step, replayed)")
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    nparam = sum(q.numel() for q in model.parameters()) / 1e6
    bos = tok.eos_token_id
    print(f"{a.model}: {nparam:.1f}M params, {a.new_tokens} new tokens/doc, eager tf32, KV cache")
    if a.flops:
        from torch.utils.flop_counter import FlopCounterMode
        ids1 = torch.full((1, 1), bos, dtype=torch.long, device=dev)
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            model.generate(ids1, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=a.new_tokens,
                           min_new_tokens=a.new_tokens, pad_token_id=bos)
        print(f"FLOPs/doc (measured, batch 1, KV cache): {fc.get_total_flops() / 1e9:.1f} GF", flush=True)
    print(f"{'B':>3} | {'eager ms/batch':>14} {'ms/doc':>8} | {'compiled ms/batch':>17} {'ms/doc':>8}")
    for B in a.batches:
        ids = torch.full((B, 1), bos, dtype=torch.long, device=dev)
        gen = lambda: model.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                                     max_new_tokens=a.new_tokens, min_new_tokens=a.new_tokens,
                                     pad_token_id=bos)
        with torch.no_grad():
            t_e = timed(gen, a.iters, 1)
        row = f"{B:>3} | {t_e * 1e3:>14.0f} {t_e * 1e3 / B:>8.1f} |"
        if a.graph:
            try:
                t_g = graphed_decode_time(model, B, a.new_tokens, bos, dev, a.iters)
                row += f" {t_g * 1e3:>17.0f} {t_g * 1e3 / B:>8.1f}  (manual CUDA graph)"
            except Exception as e:  # noqa: BLE001
                row += f"   graph failed: {type(e).__name__}: {str(e)[:80]}"
        elif a.compile:
            try:
                model.generation_config.cache_implementation = "static"
                model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
                with torch.no_grad():
                    t_c = timed(gen, a.iters, 2)
                row += f" {t_c * 1e3:>17.0f} {t_c * 1e3 / B:>8.1f}"
            except Exception as e:  # noqa: BLE001
                row += f"   compile failed: {type(e).__name__}"
        print(row, flush=True)


@torch.no_grad()
def graphed_decode_time(model, B: int, new_tokens: int, bos: int, dev, iters: int) -> float:
    """Manual CUDA-graph AR decode: HF StaticCache, one forward step captured, replayed per token."""
    from transformers import StaticCache
    L = new_tokens + 1
    cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=L, device=dev, dtype=model.dtype)
    inp = torch.full((B, 1), bos, dtype=torch.long, device=dev)
    pos = torch.zeros(1, dtype=torch.long, device=dev)

    def step():
        out = model(input_ids=inp, cache_position=pos, past_key_values=cache, use_cache=True)
        return out.logits[:, -1].float()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        logits_s = step()

    def run():
        cache.reset()
        inp.fill_(bos)
        for n in range(new_tokens):
            pos.fill_(n)
            g.replay()
            nxt = torch.multinomial(torch.softmax(logits_s, dim=-1), 1)
            inp.copy_(nxt)

    return timed(run, iters, 1)


if __name__ == "__main__":
    main()
