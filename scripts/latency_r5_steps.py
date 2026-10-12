"""scripts.latency_r4 with the flow prior's Euler step count set from the command line.

latency_r4 builds its models through scripts.latency_probe.build_model (record) or scripts.bench_ours.build (big),
both of which hard-code cfg.flow.num_sampling_steps = 50; GraphedPrior.sample reads that value at run time. This
wrapper patches both builders to return the config with the requested step count, then calls latency_r4.main()
unchanged. No existing file is modified.

Run from the repo root:
  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python latency_r5_steps.py --prior-steps 32 --arch record --blocks 64 32 16 8 \
      --batches 1 32 --samplers temp:0.7 temp:0.75 --out bench/latency_r5_steps_record_nfe32.jsonl
All arguments other than --prior-steps go to latency_r4.
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())   # the repo root

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--prior-steps", type=int, required=True)
args, rest = pre.parse_known_args()
N = args.prior_steps

import scripts.bench_ours as bo      # noqa: E402
import scripts.latency_r4 as r4      # noqa: E402


def with_steps(build):
    def build_n(*a, **k):
        model, cfg = build(*a, **k)
        cfg.flow.num_sampling_steps = N
        return model, cfg
    return build_n


r4.build_model = with_steps(r4.build_model)   # latency_r4 calls its module-level name for --arch record
bo.build = with_steps(bo.build)               # imported inside latency_r4.main() for --arch big
print(f"# prior Euler steps = {N}", flush=True)
sys.argv = ["latency_r4"] + rest
r4.main()
