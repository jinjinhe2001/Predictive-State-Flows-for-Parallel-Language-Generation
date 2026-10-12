"""Measured FLOPs (torch FlopCounterMode) for ELF uncond generation, per size.

Wraps the REAL sampling loop (_generate_samples_single_batch) + decode
(_dlm_decode_batch) at batch=1 -> per-sample FLOPs. Random weights: FLOPs are
shape-determined, so no checkpoint is needed. Config replicates the ELF-B eval
(elf_eval.log): sde, 32 steps, cfg=1, self_cond_cfg_scale=3, sde_gamma=1.5,
logit_normal; t5-small latent dim 512, canvas 1024, vocab 32128.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from types import SimpleNamespace
import torch
from torch.utils.flop_counter import FlopCounterMode
from modules.model import ELF_models
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch

GF = 1e9
device = torch.device('cuda')
TXT_DIM, MAXLEN, VOCAB = 512, 1024, 32128

cfg = SimpleNamespace(
    t_eps=0.05, self_cond_prob=0.5, num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4, denoiser_noise_scale=2.0,
    denoiser_p_mean=-1.5, denoiser_p_std=0.8, max_length=MAXLEN, use_bf16=True,
)
sc = SimpleNamespace(sampling_method='sde', sde_gamma=1.5, time_schedule='logit_normal')
NSTEPS, CFG_SCALE, SCCFG = int(os.environ.get('NSTEPS', 32)), 1.0, 3.0
sc.sde_gamma = 1.5 if NSTEPS == 32 else 1.0

def build(name):
    return ELF_models[name](
        text_encoder_dim=TXT_DIM, max_length=MAXLEN, bottleneck_dim=128,
        num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        vocab_size=VOCAB,
    ).to(device).eval()

def count(fn):
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        fn()
    return fc.get_total_flops() / GF

print(f'device={device}  batch=1 (per-sample FLOPs, GF=1e9 true-FLOPs)')
print(f'sampling: {sc.sampling_method} {NSTEPS} steps  cfg={CFG_SCALE} sccfg={SCCFG} gamma={sc.sde_gamma}')
print()
hdr = f"{'model':>7} {'H':>5} {'L':>3} {'params(M)':>10} | {'1 fwd GF':>9} {'gen GF':>9} {'decode GF':>10} | {'TOTAL GF':>9}"
print(hdr); print('-'*len(hdr))
for name in ['ELF-B','ELF-M','ELF-L']:
    torch.manual_seed(0)
    m = build(name)
    nparam = sum(p.numel() for p in m.parameters())/1e6
    gen = torch.Generator()
    z = (torch.randn(1, MAXLEN, TXT_DIM, device=device)*cfg.denoiser_noise_scale)
    t_steps = get_sampling_steps(NSTEPS, 'logit_normal', P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std, device=device, dtype=torch.float32)
    # single forward (self-cond input = 2*dim) for sanity
    zc = torch.cat([z, torch.zeros_like(z)], dim=-1)
    tb = torch.full((1,), 0.5, device=device)
    scb = torch.full((1,), SCCFG, device=device)
    g_fwd = count(lambda: m(zc, tb, deterministic=True, self_cond_cfg_scale=scb))
    g_gen = count(lambda: _generate_samples_single_batch(m, gen, z, t_steps, None, None, cfg, sc, CFG_SCALE, SCCFG))
    latent = _generate_samples_single_batch(m, gen, z, t_steps, None, None, cfg, sc, CFG_SCALE, SCCFG)
    g_dec = count(lambda: _dlm_decode_batch(latent, m, t_steps[-1].item(), cfg, SCCFG))
    H = m.hidden_size; L = m.depth
    print(f'{name:>7} {H:>5} {L:>3} {nparam:>10.1f} | {g_fwd:>9.1f} {g_gen:>9.1f} {g_dec:>10.1f} | {g_gen+g_dec:>9.1f}')
    del m, z, latent
    torch.cuda.empty_cache()
print()
print('gen = 32-step sde rollout (1 fwd/step, self-cond CFG via prefix tokens);')
print('decode = 1 fwd + factored unembed to vocab. TOTAL = gen+decode (per sample).')
