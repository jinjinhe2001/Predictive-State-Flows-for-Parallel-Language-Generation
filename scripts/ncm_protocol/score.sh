#!/bin/bash
# Score generate.sh's samples with NCM's protocol, both as 1024-token packed windows and one document per sample:
#   bash scripts/ncm_protocol/score.sh <ncm checkout> psf_k8 psf_k8_states-only ...
# then interpolate to NCM's entropy:
#   python scripts/ncm_protocol/summarize.py outputs/ncm_protocol/score_packed.log --targets 5.42
set -euo pipefail
PYTHON=${PYTHON:-python}
OUT=${OUT:-outputs/ncm_protocol}
NCM=$1; shift
RUNS=()
for M in "$@"; do
  for f in $OUT/${M}_s42_nfe50_*_R0.jsonl; do
    b=$(basename $f); T=${b#${M}_s42_}; T=${T%.jsonl}
    RUNS+=("${M}_${T}=$OUT/${M}_s42_$T.jsonl,$OUT/${M}_s0_$T.jsonl,$OUT/${M}_s1_$T.jsonl,$OUT/${M}_s2_$T.jsonl,$OUT/${M}_s3_$T.jsonl")
  done
done
for MODE in packed doc; do
  $PYTHON scripts/ncm_protocol/ncm_score.py --ncm $NCM --mode $MODE "${RUNS[@]}" | tee $OUT/score_$MODE.log
done
