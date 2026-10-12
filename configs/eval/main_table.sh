#!/bin/bash
# Reproduce the quality columns of Table 1 from the released checkpoints: 1000 unconditional samples per row,
# generative perplexity and unigram entropy under GPT-2 Large, MAUVE@512 against 1000 held-out documents.
#   CKPT_DIR   directory holding the downloaded *.pt files (default: checkpoints)
#   OUT        output directory (default: outputs/main_table)
#   MODELS     space-separated subset of the rows below
# Table 1 reports PSF K=32's perplexity from the --gen-batch 125 samples (configs/sampling/psf_k32.args) and
# its MAUVE from a second 1000-sample set drawn with --gen-batch 50.
cd "$(dirname "$0")/../.."
set -euo pipefail
PYTHON=${PYTHON:-python}
CKPT_DIR=${CKPT_DIR:-checkpoints}
OUT=${OUT:-outputs/main_table}
MODELS=${MODELS:-"psf_k64 psf_k32 psf_k16 psf_k8 psf_k8_states-only psf-l_k16 psf-l_k8 psf-l_k8_states-only"}
mkdir -p $OUT

for M in $MODELS; do
  $PYTHON -m scripts.eval_gen --ckpt $CKPT_DIR/$M.pt $(cat configs/models/$M.args) $(cat configs/sampling/$M.args) \
    --save-texts $OUT/texts_$M 2>&1 | tee $OUT/eval_$M.log
done

# same-data autoregressive baseline: sample, then score with the same judge. sample_ar seeds once, so the
# paper's row is the second setting of this exact command (the 1.0:0.86 dump is a by-product).
if [ -f $CKPT_DIR/ar-197m.pt ]; then
  $PYTHON -m scripts.sample_ar --ckpt $CKPT_DIR/ar-197m.pt --num-samples 1000 --gen-batch 125 --val-batches 0 \
    --settings 1.0:0.86,1.0:0.88 --save-texts $OUT/texts_ar-197m
  $PYTHON -m scripts.eval_gen --texts $OUT/texts_ar-197m_t1.00_p0.88.jsonl --num-samples 1000 2>&1 | tee $OUT/eval_ar-197m.log
fi

# MAUVE@512 against held-out OpenWebText (needs the dataset from scripts.download_data)
[ -f $OUT/real_owt_1000.jsonl ] || $PYTHON -m scripts.dump_real_texts --num 1000 --out $OUT/real_owt_1000.jsonl
DUMPS=()
for f in $OUT/texts_*.jsonl; do DUMPS+=("$(basename $f .jsonl)=$f"); done
$PYTHON -m scripts.mauve_eval --real $OUT/real_owt_1000.jsonl --max-len 512 "${DUMPS[@]}" 2>&1 | tee $OUT/mauve512.log

# perplexity interpolated to entropy 5.16 (only meaningful with a temperature sweep; see configs/eval/sweep.sh)
$PYTHON -m scripts.isoentropy $OUT/eval_*.log --targets 5.16 | tee $OUT/isoentropy.log
