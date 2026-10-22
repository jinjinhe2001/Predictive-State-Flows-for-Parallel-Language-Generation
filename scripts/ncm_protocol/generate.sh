#!/bin/bash
# PSF samples for the NCM-protocol comparison: NCM's 5 seeds (42 0 1 2 3) x 400 documents per sampling setting, enough
# to pack 256 windows of 1024 GPT-2 tokens per seed. Run from the repo root:
#   bash scripts/ncm_protocol/generate.sh psf_k8 "" 0.75:0.95,0.85:0.95,0.95:0.95,1.0:1.0
#   bash scripts/ncm_protocol/generate.sh psf_k8_states-only "--rc-mode zero --no-repeat-ngram 4" 1.0:0.55,1.0:0.65,1.0:0.75
#   CKPT_DIR   directory holding the downloaded *.pt files (default: checkpoints)
#   OUT        output directory (default: outputs/ncm_protocol)
set -euo pipefail
PYTHON=${PYTHON:-python}
CKPT_DIR=${CKPT_DIR:-checkpoints}
OUT=${OUT:-outputs/ncm_protocol}
M=$1; EXTRA=$2; SETTINGS=$3
mkdir -p $OUT
for S in 42 0 1 2 3; do
  $PYTHON -m scripts.eval_gen --ckpt $CKPT_DIR/$M.pt $(cat configs/models/$M.args) $EXTRA \
    --z-level 0.0 --sampling-steps 50 --num-samples 400 --gen-batch 50 --settings $SETTINGS --seed $S \
    --save-texts $OUT/${M}_s$S 2>&1 | tee $OUT/gen_${M}_s$S.log
done
