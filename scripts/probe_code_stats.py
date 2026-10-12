"""Where do the prior's realization codes sit relative to the encoder's?

Zeroing the codes at inference lowered perplexity at equal entropy (eval_r2_abl_rczero_*), and shuffled
codes matched the prior's own, so the question is whether the prior's codes are off the distribution the
writer was trained on. This prints, for the code part of the plan: per-dimension mean and std averaged
over dimensions, the mean L2 norm per block, and the mean absolute correlation between neighbouring
blocks' codes, for (a) the encoder's posterior means on held-out documents (what the prior was fit to),
(b) posterior samples (what the writer saw in training), (c) the prior's generated codes.

  CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_code_stats --ckpt runs/prior-srpl-rc64-look-k8/prior.pt --num-blocks 8 <arch flags>
"""
from __future__ import annotations

import argparse

import torch

from psf.diffusion.flow import sample as flow_sample
from psf.models.psf import PSF
from psf.utils.data import load_tokenizer, make_dataloaders
from psf.utils.misc import get_device, load_checkpoint, set_seed
from scripts.eval_gen import config_from_args, parse_args as eval_parse_args


def stats(name: str, r: torch.Tensor) -> None:
    r = r.float()
    B, K, R = r.shape
    flat = r.reshape(-1, R)
    norm = r.norm(dim=-1).mean().item()
    corr = torch.nn.functional.cosine_similarity(r[:, :-1].reshape(-1, R), r[:, 1:].reshape(-1, R), dim=-1).abs().mean().item()
    print(f"{name:<34} mean={flat.mean().item():+.3f} std/dim={flat.std(dim=0).mean().item():.3f} "
          f"|r|={norm:.2f} (sqrt(R)={R ** 0.5:.1f}) |cos(r_k, r_k+1)|={corr:.3f} max|r|={flat.abs().max().item():.1f}",
          flush=True)


@torch.no_grad()
def main() -> None:
    args = eval_parse_args()
    device = get_device(args.device)
    set_seed(args.seed)
    cfg, _ = config_from_args(args)
    cfg.train.batch_size = 50
    cfg.data.num_workers = 0
    load_tokenizer(cfg)
    model = PSF(cfg).to(device).eval()
    load_checkpoint(args.ckpt, model, device)
    n = args.num_samples
    nlat = cfg.num_blocks * cfg.num_latent_slots + cfg.num_global_latents
    z = flow_sample(model.denoiser, n, nlat, model.prior_dim, cfg.flow, device, cfg_scale=args.cfg_scale)
    _, rc_prior = model.split_latents(model.denoiser.denormalize(z))
    _, val = make_dataloaders(cfg)
    mus, samples = [], []
    got = 0
    for batch in val:
        batch = batch.to(device)
        enc = model.encode(batch.tokens, batch.mask)
        mus.append(enc.rc_mu.float())
        samples.append(model.sample_rc(enc, "sample").float())
        got += batch.tokens.shape[0]
        if got >= n:
            break
    stats("encoder posterior mean (real docs)", torch.cat(mus)[:n])
    stats("encoder posterior sample (real)", torch.cat(samples)[:n])
    stats("prior codes (generated plans)", rc_prior)
    stats("N(0, I)", torch.randn_like(rc_prior))


if __name__ == "__main__":
    main()
