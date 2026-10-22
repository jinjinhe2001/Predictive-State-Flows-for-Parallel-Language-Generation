# PSF under NCM's OpenWebText protocol

Scripts behind the [NCM](https://github.com/HazyResearch/ncm) comparison in the top-level README. They evaluate the
released PSF checkpoints with NCM's own scorer and timing setup; NCM's numbers are the ones its paper reports.

## How the two evaluations differ

| | PSF paper (Table 1) | NCM protocol (this folder) |
|---|---|---|
| sample | one document, cut at its EOS (~860 GPT-2 tokens) | 1024 GPT-2 tokens, running across documents (NCM's `cut_at_eos: false`) |
| samples | 1,000, one seed | 5 seeds (42, 0, 1, 2, 3) x 256 |
| scorer | `scripts.eval_gen` (GPT-2 Large) | NCM's `evaluation.gen_ppl.generation_perplexity` (GPT-2 Large) |
| matched entropy | H = 5.16 | H = 5.42 (NCM's entropy; data 5.44) |

PSF documents are concatenated without a separator (NCM's decoder drops EOS) and cut into 256 windows of 1024 tokens
per seed. Reaching H = 5.42 needs a higher sampling temperature than the paper's settings. As a check, the same scorer
gives 15.1 @ H = 5.45 on NCM's own OpenWebText validation stream (NCM reports 14.15 @ 5.44).

## Results

Gen. PPL interpolated to H = 5.42 (log-linear between the two nearest temperatures; for K=16 and K=32 the lowest
measured entropy is just above 5.42, so their values are extrapolated by at most 0.03); time per 1024-token sequence on
one H100 at batch 16, FP32 with TF32 off and compiled graphs, the setup that reproduces NCM's reported times (below).

| model | Gen. PPL @ 5.42 | nearest measured point | ms / sequence |
|---|---|---|---|
| NCM (reported) | 83.12 | 83.12 @ 5.42 | 80 |
| PSF K=32 | 43.3 | 46.7 @ 5.45 | 38.7 |
| PSF K=16 | 35.9 | 36.6 @ 5.43 | 44.5 |
| PSF K=8 | 32.9 | 35.5 @ 5.45 | 68.2 |
| PSF K=8, states-only | 30.0 | 33.4 @ 5.46 | not timed in FP32 (TF32: 44.5 vs 46.3 for K=8) |
| PSF-L K=8, states-only | 25.8 | 31.8 @ 5.49 | 112.9 |

Scoring each PSF document on its own instead (`--mode doc`, lower entropy at a given temperature) gives 47.5 for K=8
at H = 5.42.

NCM's paper times "1,024 sequences at batch size 16 on a single H100". `calib_ncm.py` times NCM's own generation
code (random weights; the cost depends only on shapes) under several setups against two of its reported numbers.
Only FP32 + `torch.compile` matches both: 5.90 s for TinyStories (paper: 5.65 s) and 75 ms per OpenWebText sequence
(paper: 80 ms). In BF16 + `torch.compile` NCM takes 25 ms per sequence and PSF (TF32) 24 / 29 / 46 ms for K = 32 / 16 / 8.

## Reproduce

Needs the checkpoints (see the top-level README) and a checkout of [NCM](https://github.com/HazyResearch/ncm)
for its scorer and model code (`pip install omegaconf hydra-core` on top of this repo's requirements).

```bash
# 1. samples: 5 seeds x 400 documents per setting
bash scripts/ncm_protocol/generate.sh psf_k8 "" 0.75:0.95,0.85:0.95,0.95:0.95,1.0:1.0
bash scripts/ncm_protocol/generate.sh psf_k8_states-only "--rc-mode zero --no-repeat-ngram 4" 1.0:0.55,1.0:0.65,1.0:0.75
bash scripts/ncm_protocol/generate.sh psf_k16 "" 0.80:0.95,0.90:0.95,1.0:0.95
bash scripts/ncm_protocol/generate.sh psf_k32 "" 0.80:0.95,0.90:0.95,1.0:0.95

# 2. NCM's scorer, packed 1024-token windows and one document per sample; interpolate to H = 5.42
bash scripts/ncm_protocol/score.sh ../ncm psf_k8 psf_k8_states-only psf_k16 psf_k32
python scripts/ncm_protocol/summarize.py outputs/ncm_protocol/score_packed.log outputs/ncm_protocol/score_doc.log --targets 5.42

# 3. timing: PSF in FP32 (TF32 off), then NCM's generator under each setup
python scripts/ncm_protocol/time_psf.py --fp32-highest --ckpt-dir checkpoints --blocks 8 16 32 --batches 16 \
    --iters 64 --samplers tp:0.95:0.95 --out outputs/ncm_protocol/time_psf_fp32.jsonl
PSF=$PWD; cd ../ncm && for M in fp32 bf16 bf16+compile fp32+compile; do
  python $PSF/scripts/ncm_protocol/calib_ncm.py --task owt --mode $M; done
```

`ncm_score.py --data <NCM's owt/valid_gpt2.bin> x` scores the Data row; `plot_owt_speed.py` draws the figure from
NCM's Figure 5 SVG and the results above.
