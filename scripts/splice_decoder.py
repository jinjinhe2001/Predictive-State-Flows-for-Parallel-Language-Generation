"""Graft a decoder-only finetune onto an existing stage-2 checkpoint.

The boundary-token fix touches the decoder only. Trained with
``--freeze-encoder`` the latents stay bit-identical, so the prior already
fitted on them remains valid and stage 2 need not be rerun. This swaps the
decoder tensors in, leaves encoder and denoiser alone, and *verifies* the
frozen-encoder claim by comparing every encoder tensor rather than trusting
the flag.

    python -m scripts.splice_decoder --prior runs/prior-pci/prior.pt \
        --ae runs/ae-bt/ae.pt --out runs/prior-bt/prior.pt
"""

from __future__ import annotations

import argparse
import os

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="swap a finetuned decoder into a prior ckpt")
    p.add_argument("--prior", required=True, help="stage-2 checkpoint to graft onto")
    p.add_argument("--ae", required=True, help="stage-1 checkpoint holding the new decoder")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prior = torch.load(args.prior, map_location="cpu", weights_only=False)
    ae = torch.load(args.ae, map_location="cpu", weights_only=False)
    pm, am = prior["model"], ae["model"]

    drift = [
        k for k, v in pm.items()
        if k.startswith("encoder.")
        and (k not in am or v.shape != am[k].shape or not torch.equal(v, am[k]))
    ]
    if drift:
        raise SystemExit(
            f"encoder differs in {len(drift)} tensors (e.g. {drift[:3]}) — the AE run "
            f"was not --freeze-encoder, so the prior no longer matches its latents"
        )

    kept = {k: v for k, v in pm.items() if not k.startswith("decoder.")}
    new_dec = {k: v for k, v in am.items() if k.startswith("decoder.")}
    added = sorted(set(new_dec) - set(pm))
    dropped = sorted(k for k in pm if k.startswith("decoder.") and k not in new_dec)

    n_enc = sum(1 for k in pm if k.startswith("encoder."))
    print(f"encoder verified identical across {n_enc} tensors")
    print(f"decoder: {len(new_dec)} tensors in, {len(dropped)} dropped, new keys {added}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": {**kept, **new_dec}, "steps": ae.get("steps", 0)}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
