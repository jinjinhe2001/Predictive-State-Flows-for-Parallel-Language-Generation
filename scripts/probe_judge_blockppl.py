"""GPT-2-large per-block perplexity, split into first-word-of-block vs rest.

    python probe_judge_blockppl.py MODES N [CKPT] [bt]

MODES  comma-separated subset of real,recon,parallel,rollout,gtroll,seedblk1,
       priors1
       (gtroll = ground-truth rollout: every S_k from a real held-out doc,
       decoded through rollout's sequential block-AR loop instead of
       recon's parallel pass -- isolates decode scheduling from state
       quality; seedblk1 = block 1 is a real doc's verbatim text, blocks
       2+ are plain self-referential rollout -- the "real S1 start";
       priors1 = S_1 ~ flow prior (marginal single-block draw), blocks 2+
       plain self-referential rollout -- "diffusion S1" rollout;
       partial<T> (e.g. partial0.5) = SDEdit-style: corrupt the REAL state
       to time T (0=pure noise=exactly "parallel", 1=the real state=
       exactly "recon") and only run the prior over [T, 1], then decode in
       parallel -- traces the path between those two endpoints)
TAGS   may also include "z0" to decode parallel mode with z_level=0.0
       (the plan label -- the one-shot main-metric convention for
       level-cond checkpoints)

Each mode also reports unigram entropy (gpt2 + t5 token space, corpus and
per-sample) so block-interior PPL can be read at its entropy position.
N      documents per mode
CKPT   stage-2 checkpoint (default runs/prior-pci/prior.pt)
bt     pass the literal string "bt" to build the model with boundary tokens
"""
import bisect
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from psf.config import Config
from psf.models.psf import PSF
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.misc import load_checkpoint, set_seed
from scripts.eval_gen import mean_sample_entropy, unigram_entropy

MODES = sys.argv[1].split(",")
N = int(sys.argv[2])
CKPT = sys.argv[3] if len(sys.argv) > 3 else "runs/prior-pci/prior.pt"
TAGS = sys.argv[4].split(",") if len(sys.argv) > 4 else []
TEMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
TOPP = 0.95 if TEMP > 0 else 1.0
cfg = Config(); cfg.encoder_type = "prefix"; cfg.decoder.causal_latents = True
cfg.decoder.boundary_tokens = "bt" in TAGS
cfg.decoder.landing_state = "land" in TAGS or "land64" in TAGS
cfg.decoder.landing_dim = 64 if "land64" in TAGS else 0
cfg.decoder.landing_attn = "attn" in TAGS
cfg.decoder.noise_level_cond = "level" in TAGS
for _t in TAGS:
    if _t.startswith("win"): cfg.decoder.boundary_window = int(_t[3:])
for _t in TAGS:
    if _t.startswith("depth"):
        cfg.decoder.depth = int(_t[5:])
for _t in TAGS:
    if _t.startswith("dim"): cfg.latent_dim = int(_t[3:])
cfg.denoiser.num_cfg_tokens = 4; cfg.data.num_workers = 0
tok = load_tokenizer(cfg)
m = PSF(cfg).to("cuda").eval()
load_checkpoint(CKPT, m, "cuda")
print(f"# ckpt={CKPT} tags={TAGS} temp={TEMP}", flush=True)
S, K = cfg.block_span, cfg.num_blocks
jt = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
jt.pad_token = jt.eos_token
judge = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large").to("cuda").eval()

@torch.no_grad()
def get_docs(mode):
    set_seed(0)
    docs = []
    if mode in ("real", "recon", "gtroll", "seedblk1") or mode.startswith("partial"):
        _, vl = make_dataloaders(cfg)
        for b in vl:
            b = b.to("cuda")
            if mode == "real":
                toks, lens = b.tokens, b.mask.sum(1)
            elif mode == "recon":
                z = m.encode(b.tokens, b.mask).block_feats
                bt = m.decoder.generate(z, cfg.data.bob_token_id, TEMP, TOPP,
                                        z_level=0.0 if "z0" in TAGS else 1.0)
                toks, lens = m.truncate_at_eos(bt)
            elif mode.startswith("partial"):   # SDEdit-style: corrupt real state to t_start
                t_start = float(mode[len("partial"):])
                toks, lens = m.partial_denoise_generate(
                    b.tokens, b.mask, t_start, temperature=TEMP, top_p=TOPP,
                    z_level=0.0 if "z0" in TAGS else 1.0,
                )
            elif mode == "gtroll":   # real states, but rollout's sequential decode
                toks, lens = m.rollout_generate(
                    b.tokens.shape[0], temperature=TEMP, top_p=TOPP,
                    gt_tokens=b.tokens, gt_mask=b.mask,
                )
            else:   # seedblk1: real block 1 verbatim, blocks 2+ self-referential
                toks, lens = m.rollout_generate(
                    b.tokens.shape[0], temperature=TEMP, top_p=TOPP,
                    gt_block1=b.tokens[:, :S], gt_block1_mask=b.mask[:, :S],
                )
            for i in range(toks.shape[0]):
                ids = toks[i][: int(lens[i])].tolist()
                if ids: docs.append(ids)
                if len(docs) >= N: return docs
    else:
        while len(docs) < N:
            n = min(20, N - len(docs))
            if mode == "parallel":
                toks, lens = m.generate(n, temperature=TEMP, top_p=TOPP,
                                        z_level=0.0 if "z0" in TAGS else 1.0)
            elif mode == "priors1":
                toks, lens = m.rollout_generate(n, temperature=TEMP, top_p=TOPP,
                                                prior_s1=True)
            else:
                toks, lens = m.rollout_generate(n, temperature=TEMP, top_p=TOPP)
            for i in range(n):
                ids = toks[i][: int(lens[i])].tolist()
                if ids: docs.append(ids)
    return docs

@torch.no_grad()
def judge_blocks(docs):
    nll_k = torch.zeros(K); cnt_k = torch.zeros(K)
    agg = {"first": [0.0, 0], "rest": [0.0, 0]}
    for s in range(0, len(docs), 16):
        chunk = docs[s:s + 16]
        texts = [decode(tok, ids) for ids in chunk]
        bounds = [[len(decode(tok, ids[: j * S]))
                   for j in range(1, (len(ids) + S - 1) // S)] for ids in chunk]
        enc = jt(texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=1024, return_offsets_mapping=True)
        ids_j = enc.input_ids.to("cuda"); mask_j = enc.attention_mask.to("cuda")
        logits = judge(input_ids=ids_j, attention_mask=mask_j).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        nll = -logp.gather(-1, ids_j[:, 1:, None]).squeeze(-1).cpu()
        offs = enc.offset_mapping
        for bi in range(len(chunk)):
            valid = mask_j[bi, 1:].cpu().bool()
            prev = 0
            for ti in range(nll.shape[1]):
                if not valid[ti]: break
                blk = min(bisect.bisect_right(bounds[bi], int(offs[bi, ti + 1, 0])), K - 1)
                v = float(nll[bi, ti])
                nll_k[blk] += v; cnt_k[blk] += 1
                if blk >= 1:
                    key = "first" if blk != prev else "rest"
                    agg[key][0] += v; agg[key][1] += 1
                prev = blk
    ppl = torch.exp(nll_k / cnt_k.clamp(min=1)).tolist()
    tot = float(torch.exp(nll_k.sum() / cnt_k.sum().clamp(min=1)))
    fb = torch.exp(torch.tensor(agg["first"][0] / max(agg["first"][1], 1)))
    rb = torch.exp(torch.tensor(agg["rest"][0] / max(agg["rest"][1], 1)))
    return tot, ppl, float(fb), float(rb)

for mode in MODES:
    docs = get_docs(mode)
    tot, ppl, fb, rb = judge_blocks(docs)
    gpt2_ids = [jt(decode(tok, ids), truncation=True, max_length=1024).input_ids
                for ids in docs]
    mean_len = sum(len(d) for d in docs) / max(len(docs), 1)
    print(f"{mode}: overall {tot:.2f} | first-token-of-block {fb:.2f} vs rest-of-block {rb:.2f}", flush=True)
    print(f"  entropy: gpt2 corpus={unigram_entropy(gpt2_ids):.3f} "
          f"/sample={mean_sample_entropy(gpt2_ids):.3f} | "
          f"t5 corpus={unigram_entropy(docs):.3f} "
          f"/sample={mean_sample_entropy(docs):.3f} | "
          f"mean len={mean_len:.0f} t5 tok", flush=True)
    for row in range(0, K, 16):
        print("  k=%2d..%2d: " % (row + 1, row + 16)
              + " ".join(f"{v:7.0f}" for v in ppl[row:row + 16]), flush=True)
