"""End-to-end smoke test on the synthetic toy corpus (CPU, no downloads, about a minute).

Runs the real entry points with the PSF architecture flags for a few steps each:
block-decoder training with landing targets, self-realized prefixes, source labels and realization
codes -> Stage III prior -> a warm-started states-only prior, then checks that every checkpoint is a
plain state dict that loads with ``weights_only=True``.
"""

import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = ["--encoder", "prefix", "--causal-latents", "--boundary-tokens", "--level-cond",
        "--rc-dim", "16", "--rc-lookahead"]
PRIOR = ["--sc-cfg", "--objective", "v", "--time-sampler", "logit_normal", "--time-mean", "-1.5",
         "--time-std", "0.8", "--steps", "10", "--batch-size", "4", "--device", "cpu"]


def run(*args: str) -> None:
    res = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-4000:]


def load_state(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert set(ckpt) >= {"model"} and all(torch.is_tensor(v) for v in ckpt["model"].values())
    return ckpt["model"]


@pytest.mark.slow
def test_toy_pipeline(tmp_path):
    ae, prior, so = (str(tmp_path / n) for n in ("ae", "prior", "prior-so"))
    run("scripts.train_ae", "--toy", *ARCH, "--decoder-noise", "--srp-p", "0.5", "--srp-temp", "0.65",
        "--srp-top-p", "0.95", "--rc-beta", "0.1", "--rc-kl-target", "4", "--rc-warmup", "2",
        "--lr-schedule", "cosine", "--wd-exclude", "--steps", "10", "--batch-size", "4",
        "--device", "cpu", "--output-dir", ae)
    run("scripts.train_prior", "--toy", "--ae-ckpt", f"{ae}/ae.pt", *ARCH, *PRIOR,
        "--rc-source", "mean", "--output-dir", prior)
    run("scripts.train_prior", "--toy", "--ae-ckpt", f"{ae}/ae.pt", "--resume", f"{prior}/prior.pt",
        *ARCH, *PRIOR, "--rc-source", "zero", "--output-dir", so)

    ae_sd, prior_sd = load_state(f"{ae}/ae.pt"), load_state(f"{prior}/prior.pt")
    load_state(f"{so}/prior.pt")
    # the prior checkpoint carries the frozen encoder and block decoder unchanged
    shared = [k for k in ae_sd if not k.startswith("denoiser.")]
    assert shared and all(torch.equal(ae_sd[k], prior_sd[k]) for k in shared)
