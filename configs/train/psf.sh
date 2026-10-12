#!/bin/bash
# PSF (d = 512): the paper's full training recipe, in order (Section 3, Appendix A.2).
#
#   bash configs/train/psf.sh        # all steps
#   bash configs/train/psf.sh 7      # resume from step 7
#
# Per-GPU batch sizes assume 8 GPUs (8 x 64 = 512 documents per step); scale --batch-size for other GPU counts.
# Each prior checkpoint holds the whole model (encoder, block decoder, prior). Released checkpoints:
#   psf_k64            = $RUNS/prior-srpl-rc64-look-tlow/prior.pt      (step 6)
#   psf_k32/k16/k8     = $RUNS/prior-srpl-rc64-look-k{32,16,8}/prior.pt (steps 7-9)
#   psf_k8_states-only = $RUNS/prior-k8-zerocode/prior.pt               (step 10)
set -euo pipefail
cd "$(dirname "$0")/../.."
FROM=${1:-1}
NPROC=${NPROC:-8}
TORCHRUN=${TORCHRUN:-"torchrun --standalone --nproc_per_node $NPROC --max-restarts 0"}
PYTHON=${PYTHON:-python}
RUNS=${RUNS:-runs}
K64=${K64:-1}                     # step 6 only produces psf_k64; K64=0 skips it (K = 32/16/8 do not need it)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENC="--encoder prefix --causal-latents"
# block-decoder fine-tuning on the frozen encoder (every step after step 1)
FT="--freeze-encoder $ENC --boundary-tokens --decoder-noise --ema-decay 0.999 --bf16 --batch-size 64 --show 2"
RECIPE="--decoder-depth 18 --lr-schedule cosine --wd-exclude"
SRP="--srp-p 0.5 --srp-temp 0.65 --srp-top-p 0.95 --level-cond"
CODES="--rc-dim 64 --rc-beta 0.1 --rc-lookahead"
# Stage III flow prior on the frozen encoder + block decoder
PRIOR_ARCH="$ENC --boundary-tokens --level-cond --decoder-depth 18 --rc-dim 64 --rc-lookahead --denoiser-bottleneck 256"
PRIOR_OPT="--sc-cfg --objective v --time-sampler logit_normal --time-mean -1.5 --time-std 0.8 --ema-decay 0.9999 --bf16 --batch-size 64 --steps 30000"
step() { [ "$1" -ge "$FROM" ]; }

# 1. Stage I: encoder + block decoder from scratch at K = 64, then the base prior.
#    RECONSTRUCTED from the run record (30k decoder steps, 150k prior steps); the launch script was not archived.
if step 1; then
  $TORCHRUN -m scripts.train_ae $ENC --decoder-noise --ema-decay 0.9999 --bf16 \
    --batch-size 64 --steps 30000 --output-dir $RUNS/ae-pci
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/ae-pci/ae.pt $ENC \
    --sc-cfg --objective v --time-mean 0.8 --time-std 0.8 --ema-decay 0.9999 --bf16 \
    --batch-size 64 --steps 150000 --output-dir $RUNS/prior-pci
fi

# 2. Landing target (10k steps). The encoder is frozen from here on, so the prior stays valid: splice the new
#    block decoder into it.
if step 2; then
  $TORCHRUN -m scripts.train_ae --init-ckpt $RUNS/ae-pci/ae.pt $FT \
    --steps 10000 --output-dir $RUNS/ae-bt
  $PYTHON -m scripts.splice_decoder --prior $RUNS/prior-pci/prior.pt --ae $RUNS/ae-bt/ae.pt \
    --out $RUNS/prior-bt/prior.pt
fi

# 3. Full recipe: 18-layer block decoder, cosine schedule, no weight decay on norms/biases, scaled init (30k steps).
if step 3; then
  $TORCHRUN -m scripts.train_ae --init-ckpt $RUNS/ae-bt/ae.pt $FT $RECIPE --scaled-init \
    --steps 30000 --output-dir $RUNS/ae-full
  $PYTHON -m scripts.splice_decoder --prior $RUNS/prior-bt/prior.pt --ae $RUNS/ae-full/ae.pt \
    --out $RUNS/prior-full/prior.pt
fi

# 4. Self-realized prefixes with the source label (10k steps). The splice is RECONSTRUCTED (not archived).
if step 4; then
  $TORCHRUN -m scripts.train_ae --init-ckpt $RUNS/ae-full/ae.pt $FT $RECIPE $SRP \
    --steps 10000 --output-dir $RUNS/ae-srpl
  $PYTHON -m scripts.splice_decoder --prior $RUNS/prior-full/prior.pt --ae $RUNS/ae-srpl/ae.pt \
    --out $RUNS/prior-srpl/prior.pt
fi

# 5. Realization codes (d_r = 64) with lookahead; KL budget 16 nats per block (10k steps).
if step 5; then
  $TORCHRUN -m scripts.train_ae --init-ckpt $RUNS/ae-srpl/ae.pt $FT $RECIPE $SRP $CODES \
    --rc-kl-target 16 --rc-warmup 1000 --steps 10000 --output-dir $RUNS/ae-srpl-rc64-look
fi

# 6. Stage III prior at K = 64 (released as psf_k64), warm-started from the spliced prior of step 4.
if step 6 && [ "$K64" = 1 ]; then
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/ae-srpl-rc64-look/ae.pt --resume $RUNS/prior-srpl/prior.pt \
    $PRIOR_ARCH --rc-source mean $PRIOR_OPT --output-dir $RUNS/prior-srpl-rc64-look-tlow
fi

# 7-9. Block-length chain K = 32 -> 16 -> 8: block decoder (encoder frozen, KL budget = 1 nat per token), then a
#      30k-step prior warm-started from the previous one.
if step 7; then
  $TORCHRUN -m scripts.train_ae --init-ckpt $RUNS/ae-srpl-rc64-look/ae.pt --num-blocks 32 $FT $RECIPE $SRP $CODES \
    --rc-kl-target 32 --rc-warmup 500 --steps 8000 --output-dir $RUNS/ae-srpl-rc64-look-k32
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/ae-srpl-rc64-look-k32/ae.pt --resume $RUNS/prior-srpl/prior.pt \
    --num-blocks 32 $PRIOR_ARCH --rc-source mean $PRIOR_OPT --output-dir $RUNS/prior-srpl-rc64-look-k32
fi
if step 8; then
  $TORCHRUN -m scripts.train_ae --init-ckpt $RUNS/ae-srpl-rc64-look-k32/ae.pt --num-blocks 16 $FT $RECIPE $SRP $CODES \
    --rc-kl-target 64 --rc-warmup 500 --steps 6000 --output-dir $RUNS/ae-srpl-rc64-look-k16
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/ae-srpl-rc64-look-k16/ae.pt --resume $RUNS/prior-srpl-rc64-look-k32/prior.pt \
    --num-blocks 16 $PRIOR_ARCH --rc-source mean $PRIOR_OPT --output-dir $RUNS/prior-srpl-rc64-look-k16
fi
if step 9; then
  $TORCHRUN -m scripts.train_ae --init-ckpt $RUNS/ae-srpl-rc64-look-k16/ae.pt --num-blocks 8 $FT $RECIPE $SRP $CODES \
    --rc-kl-target 128 --rc-warmup 500 --steps 6000 --output-dir $RUNS/ae-srpl-rc64-look-k8
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/ae-srpl-rc64-look-k8/ae.pt --resume $RUNS/prior-srpl-rc64-look-k16/prior.pt \
    --num-blocks 8 $PRIOR_ARCH --rc-source mean $PRIOR_OPT --output-dir $RUNS/prior-srpl-rc64-look-k8
fi

# 10. States-only prior at K = 8 (code targets = 0), 30k steps from the K = 8 prior.
if step 10; then
  $TORCHRUN -m scripts.train_prior --ae-ckpt $RUNS/ae-srpl-rc64-look-k8/ae.pt --resume $RUNS/prior-srpl-rc64-look-k8/prior.pt \
    --num-blocks 8 $PRIOR_ARCH --rc-source zero $PRIOR_OPT --output-dir $RUNS/prior-k8-zerocode
fi
