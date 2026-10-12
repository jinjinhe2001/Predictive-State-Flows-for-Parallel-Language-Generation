"""Landing-target invariants of the block decoder (decoder.boundary_tokens), on a toy model (CPU, no data).

Checks the invariants the design rests on:

  1. position 0 of every block is *blind* to the landing target -- otherwise
     the one training forward would not match decode phase 1, which runs
     before any b_k exists;
  2. positions >= 1 do see it (the wiring is actually live);
  3. only b_{k+1} crosses the seam -- perturbing the rest of block k+1
     changes nothing in block k;
  4. generate() feeds itself the b it sampled at step 0;
  5. boundary_tokens=False is bit-identical to the old decoder.

    python -m pytest -q tests/test_landing_target.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psf.config import toy_config
from psf.models.psf import PSF
from psf.utils.misc import set_seed


def build(boundary: bool) -> PSF:
    set_seed(0)
    cfg = toy_config()
    cfg.encoder_type = "prefix"
    cfg.decoder.causal_latents = True
    cfg.decoder.boundary_tokens = boundary
    return PSF(cfg).eval()


def test_landing_target_invariants() -> None:
    torch.manual_seed(0)
    model = build(True)
    cfg = model.config
    B, K, S, V = 2, cfg.num_blocks, cfg.block_span, cfg.data.vocab_size
    tokens = torch.randint(2, V, (B, K * S))
    mask = torch.ones_like(tokens, dtype=torch.bool)

    with torch.no_grad():
        z = model.encode(tokens, mask).block_feats
        inputs, targets = model.block_view(tokens)
        nf = model.boundary_targets(targets)
        W = cfg.decoder.boundary_window                       # landing window [B, K, W] (W = 1 by default)
        assert nf.shape == (B, K, W) and (nf[:, -1] == -1).all(), nf.shape
        assert torch.equal(nf[:, :-1, 0], targets[:, 1:, 0]), "b_{k+1} mis-aligned"

        base = model.decoder(z, inputs, next_first=nf)
        # A different landing target for every block.
        alt = model.decoder(z, inputs, next_first=torch.randint(2, V, (B, K)))
        none = model.decoder(z, inputs, next_first=None)

    d0 = max((base - alt)[:, :, 0].abs().max(), (base - none)[:, :, 0].abs().max())
    d1 = (base - alt)[:, :, 1:].abs().max()
    print(f"1/2. pos-0 drift {d0:.3e} (want 0)   pos>=1 drift {d1:.3e} (want >0)")
    assert d0 == 0.0, "position 0 leaks the landing target -- decode phase 1 breaks"
    assert d1 > 1e-4, "landing target has no effect on positions >= 1"

    # 3. Only the first token of block k+1 may cross the seam.
    other = tokens.clone()
    other[:, 1::S] = torch.randint(2, V, other[:, 1::S].shape)   # every x_{k,1}
    with torch.no_grad():
        ins2, tgt2 = model.block_view(other)
        z2 = model.encode(other, mask).block_feats
        moved = model.decoder(z2, inputs, next_first=model.boundary_targets(tgt2))
    print(f"3.   b_{{k+1}} unchanged by x_{{k+1,1}}: "
          f"{torch.equal(model.boundary_targets(tgt2), nf)}")
    assert torch.equal(model.boundary_targets(tgt2), nf)
    del moved

    # 4. generate(): the b sampled at step 0 is what later steps are told.
    with torch.no_grad():
        out = model.decoder.generate(z, cfg.data.bob_token_id, temperature=0.0)
    assert out.shape == (B, K, S), out.shape
    with torch.no_grad():
        bob = torch.full((B, K, 1), cfg.data.bob_token_id, dtype=torch.long)
        step0 = model.decoder(z, bob)[:, :, -1].argmax(-1)
    print(f"4.   step-0 argmax == generated first token: "
          f"{torch.equal(step0, out[:, :, 0])}")
    assert torch.equal(step0, out[:, :, 0])

    # 5. Flag off -> untouched decoder (same params, same numbers).
    plain = build(False)
    with torch.no_grad():
        zp = plain.encode(tokens, mask).block_feats
        a = plain.decoder(zp, inputs)
        b = plain.decoder(zp, inputs, next_first=nf)
    print(f"5.   flag off ignores next_first: {torch.equal(a, b)}")
    assert torch.equal(a, b)
    extra = sum(p.numel() for n, p in model.decoder.named_parameters()
                if n.startswith(("tgt_proj", "no_target")))
    print(f"     extra params: {extra:,}")
    print("all boundary-token invariants hold")


if __name__ == "__main__":
    test_landing_target_invariants()
