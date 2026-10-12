"""Wall-clock latency for ELF uncond generation (B/M/L), same protocol as our
scripts/latency_probe.py: single GPU, eager (no torch.compile), batch 50 and 1,
warmup + averaged iters, torch.cuda.synchronize. Random weights (latency is
shape-determined). gen = full sde rollout; dec = latent->token decode forward.
bf16 = ELF's own eval setting (autocast); tf32 = same precision as our probe.
"""
import os, sys, time
def _argl(flag, default, cast):
    if flag in sys.argv:
        i = sys.argv.index(flag); vals = []
        for v in sys.argv[i + 1:]:
            if v.startswith('--'): break
            vals.append(cast(v))
        return vals or default
    return default
BATCHES = _argl('--batches', [50, 1], int)
STEPS = _argl('--steps', [32, 64], int)
PRECS = _argl('--prec', ['bf16', 'tf32'], str)
MODELS = _argl('--models', ['ELF-B', 'ELF-M', 'ELF-L'], str)
COMPILE = '--compile' in sys.argv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from types import SimpleNamespace
import torch
from modules.model import ELF_models
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch

device = torch.device('cuda')
torch.set_float32_matmul_precision('high')
TXT_DIM, MAXLEN, VOCAB = 512, 1024, 32128
CFG_SCALE, SCCFG = 1.0, 3.0

def mkcfg(bf16):
    return SimpleNamespace(t_eps=0.05, self_cond_prob=0.5, num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4, denoiser_noise_scale=2.0, denoiser_p_mean=-1.5,
        denoiser_p_std=0.8, max_length=MAXLEN, use_bf16=bf16)

def build(name):
    return ELF_models[name](text_encoder_dim=TXT_DIM, max_length=MAXLEN, bottleneck_dim=128,
        num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        vocab_size=VOCAB).to(device).eval()

def cuda_time(fn, iters, warmup):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

print(f'device={torch.cuda.get_device_name(0)}  ' + ('torch.compile' if COMPILE else 'eager, no compile'))
hdr = f"{'model':>6} {'prec':>5} {'steps':>5} {'B':>3} | {'gen ms':>9} {'dec ms':>8} {'total ms':>9} | {'ms/sample':>9}"
print(hdr); print('-'*len(hdr)); sys.stdout.flush()
for name in MODELS:
    torch.manual_seed(0)
    m = build(name)
    if COMPILE:
        m = torch.compile(m)
    gen = torch.Generator()
    for bf16 in [pr == 'bf16' for pr in PRECS]:
        cfg = mkcfg(bf16)
        for B in BATCHES:
            iters, warmup = (2, 1) if B >= 16 else (5, 2)
            for nsteps in STEPS:
                gamma = 1.5 if nsteps == 32 else 1.0
                sc = SimpleNamespace(sampling_method='sde', sde_gamma=gamma, time_schedule='logit_normal')
                z = torch.randn(B, MAXLEN, TXT_DIM, device=device) * cfg.denoiser_noise_scale
                t_steps = get_sampling_steps(nsteps, 'logit_normal', P_mean=cfg.denoiser_p_mean,
                    P_std=cfg.denoiser_p_std, device=device, dtype=torch.float32)
                with torch.no_grad():
                    t_gen = cuda_time(lambda: _generate_samples_single_batch(m, gen, z, t_steps, None, None, cfg, sc, CFG_SCALE, SCCFG), iters, warmup)
                    lat = _generate_samples_single_batch(m, gen, z, t_steps, None, None, cfg, sc, CFG_SCALE, SCCFG)
                    t_dec = cuda_time(lambda: _dlm_decode_batch(lat, m, t_steps[-1].item(), cfg, SCCFG), iters, warmup)
                tot = t_gen + t_dec
                print(f'{name:>6} {"bf16" if bf16 else "tf32":>5} {nsteps:>5} {B:>3} | {t_gen*1e3:>9.1f} {t_dec*1e3:>8.1f} {tot*1e3:>9.1f} | {tot*1e3/B:>9.2f}')
                sys.stdout.flush()
                del z, lat; torch.cuda.empty_cache()
    del m; torch.cuda.empty_cache()
print('done')
