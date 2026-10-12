#!/bin/bash
# Same-data autoregressive baseline: 197M GPT-style LM (d = 1024, 13 layers, tied embeddings) on the same
# T5-tokenized OpenWebText, 103k steps at 512 documents per step (54B tokens). Released as ar-197m.
set -euo pipefail
cd "$(dirname "$0")/../.."
NPROC=${NPROC:-8}
TORCHRUN=${TORCHRUN:-"torchrun --standalone --nproc_per_node $NPROC --max-restarts 0"}
RUNS=${RUNS:-runs}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$TORCHRUN -m scripts.train_ar --output-dir $RUNS/ar-197m --dim 1024 --depth 13 --heads 16 --batch-size 64 --steps 103000 \
  --lr 3e-4 --warmup 1000 --wd 0.01 --ema-decay 0.999 --log-every 100 --val-every 2000 --save-every 5000
