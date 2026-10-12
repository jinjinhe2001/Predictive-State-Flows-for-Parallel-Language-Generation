#!/bin/bash
# PSF-L (d = 1024): the paper's training recipe for the larger model, in order (Appendix A.2).
#
#   bash configs/train/psf-l.sh      # all steps
#   bash configs/train/psf-l.sh 4    # resume from step 4
#
# Per-GPU batch sizes assume 8 GPUs: block decoder 24 x 8 = 192 documents per step, prior 48 x 8 = 384.
# Released checkpoints (each prior holds the whole model):
#   psf-l_k64/k32/k16/k8 = $RUNS/big-prior-k{64,32,16,8}/prior.pt (steps 4-7)
#   psf-l_k8_states-only = $RUNS/big-prior-k8-zerocode/prior.pt    (step 8)
# Steps 1-3 are RECONSTRUCTED from the run record; their launch scripts were not archived.
set -euo pipefail
cd "$(dirname "$0")/../.."
FROM=${1:-1}
NPROC=${NPROC:-8}
TORCHRUN=${TORCHRUN:-"torchrun --standalone --nproc_per_node $NPROC --max-restarts 0"}
PYTHON=${PYTHON:-python}
RUNS=${RUNS:-runs}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENC="--encoder prefix --causal-latents --latent-dim 1024"
FT="--freeze-encoder $ENC --boundary-tokens --decoder-noise --ema-decay 0.999 --bf16 --batch-size 24 --show 2 --decoder-depth 18 --lr-schedule cosine --wd-exclude --srp-p 0.5 --srp-temp 0.65 --srp-top-p 0.95 --level-cond --rc-dim 64 --rc-beta 0.1 --rc-lookahead"
DENOISER="--denoiser-hidden 1024 --denoiser-depth 24 --denoiser-heads 16"
PRIOR_ARCH="$ENC --boundary-tokens --level-cond --decoder-depth 18 --rc-dim 64 --rc-lookahead $DENOISER --denoiser-bottleneck 256"
PRIOR_OPT="--sc-cfg --objective v --time-sampler logit_normal --time-mean -1.5 --time-std 0.8 --bf16 --steps 30000 --ema-decay 0.9999 --batch-size 48"
step() { [ "$1" -ge "$FROM" ]; }

# 1. Stage I: encoder + 18-layer block decoder from scratch at K = 64, with landing targets from the start
#    (batch 32 per GPU, about 39.5k steps).
if step 1; then
  $TORCHRUN -m scripts.train_ae $ENC --decoder-depth 18 --boundary-tokens --decoder-noise --ema-decay 0.999 --bf16 \
    --batch-size 32 --steps 39500 --output-dir $RUNS/ae-d1024-base
fi

# 2. Self-realized prefixes with the source label (12k steps, batch 24 per GPU). The states are detached, so the
#    encoder stays bit-identical to step 1 (scripts/compare_ckpt_tensors.py checks this).
if step 2; then
  $TORCHRUN -m scripts.train_ae --init-ckpt $RUNS/ae-d1024-base/ae.pt $ENC --decoder-depth 18 --boundary-tokens \
    --decoder-noise --ema-decay 0.999 --bf16 --batch-size 24 --srp-p 0.5 --srp-temp 0.65 --srp-top-p 0.95 --level-cond \
    --steps 12000 --output-dir $RUNS/ae-d1024-srpl
fi

# 3. Warm-start prior (308M): 30k steps on a 0.3 mix of real documents and a 200k-document self-rollout bank.
if step 3; then
  $PYTHON -m scripts.sample_rollout_bank --ckpt $RUNS/ae-d1024-srpl/ae.pt $ENC --boundary-tokens --sc-cfg \
    --decoder-depth 18 --level-cond --num 200000 --temperature 0.9 --out $RUNS/rollout_bank/bank_d1024_200k.pt
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/ae-d1024-srpl/ae.pt $ENC --boundary-tokens --level-cond --decoder-depth 18 \
    $DENOISER --sc-cfg --objective v --time-mean 0.8 --time-std 0.8 --ema-decay 0.9999 --bf16 --batch-size 48 --steps 30000 \
    --rollout-bank $RUNS/rollout_bank/bank_d1024_200k.pt --rollout-mix-p 0.3 --output-dir $RUNS/prior-d1024-srpl-big-rollmix
fi

# 4-7. Block-length chain K = 64 -> 32 -> 16 -> 8: realization codes + block decoder (KL budget = 1 nat per token),
#      then a 30k-step prior warm-started from the previous one.
chain() {  # chain <K> <steps> <KL budget> <init block decoder> <warm-start prior>
  local K=$1 steps=$2 kl=$3 init=$4 resume=$5
  $TORCHRUN -m scripts.train_ae --init-ckpt $init --num-blocks $K --steps $steps --rc-kl-target $kl --rc-warmup 1000 \
    $FT --output-dir $RUNS/big-ae-k$K
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/big-ae-k$K/ae.pt --resume $resume --num-blocks $K \
    $PRIOR_ARCH --rc-source mean $PRIOR_OPT --output-dir $RUNS/big-prior-k$K
}
if step 4; then chain 64 10000 16  $RUNS/ae-d1024-srpl/ae.pt $RUNS/prior-d1024-srpl-big-rollmix/prior.pt; fi
if step 5; then chain 32 8000  32  $RUNS/big-ae-k64/ae.pt    $RUNS/big-prior-k64/prior.pt; fi
if step 6; then chain 16 6000  64  $RUNS/big-ae-k32/ae.pt    $RUNS/big-prior-k32/prior.pt; fi
if step 7; then chain 8  6000  128 $RUNS/big-ae-k16/ae.pt    $RUNS/big-prior-k16/prior.pt; fi

# 8. States-only prior at K = 8 (code targets = 0), 30k steps from the K = 8 prior.
if step 8; then
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/big-ae-k8/ae.pt --resume $RUNS/big-prior-k8/prior.pt --num-blocks 8 \
    $PRIOR_ARCH --rc-source zero $PRIOR_OPT --output-dir $RUNS/big-prior-k8-zerocode
fi
