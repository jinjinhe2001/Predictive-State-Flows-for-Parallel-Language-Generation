#!/bin/bash
# Temperature sweep for one released model, for the matched-entropy ("@5.16") comparisons:
#   bash configs/eval/sweep.sh psf_k8 0.65:0.95,0.7:0.95,0.75:0.95,0.8:0.95
cd "$(dirname "$0")/../.."
set -euo pipefail
PYTHON=${PYTHON:-python}
CKPT_DIR=${CKPT_DIR:-checkpoints}
OUT=${OUT:-outputs/sweeps}
M=$1; SETTINGS=$2
mkdir -p $OUT
EXTRA=""
case $M in *states-only) EXTRA="--rc-mode zero";; esac
$PYTHON -m scripts.eval_gen --ckpt $CKPT_DIR/$M.pt $(cat configs/models/$M.args) $EXTRA \
  --z-level 0.0 --sampling-steps 50 --num-samples 1000 --gen-batch 50 --settings $SETTINGS \
  --save-texts $OUT/texts_$M 2>&1 | tee $OUT/eval_${M}_sweep.log
$PYTHON -m scripts.isoentropy $OUT/eval_${M}_sweep.log --targets 5.16
