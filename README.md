# Efficient Corrective Sampling by Over-Relaxation

**Blog post: [Efficient Corrective Sampling by Over-Relaxation](https://mhh0318.github.io/blog/efficient-corrective-sampling/)**

Sampling-only code for the released checkpoints of
[Correction Training for Diffusion Models](https://arxiv.org/abs/2610.14136) (Ahn et al., 2026). Nothing is retrained:
we use the released corrector with a rescaled gate and reach the same FID with 30% fewer network evaluations. This
repository is a fork of [their code](https://github.com/sunovivid/correction-training) at `475e214`; see the upstream
README for training, checkpoints and the original sampler.

## Samplers

Inside the correction interval, with the learned per-channel gate γ and corrector proposal C:

| Sampler | Update | Corrector calls per active step |
| :--- | :--- | :---: |
| corrective (original) | K corrections `v_k = v_{k-1} + γ ⊙ (C(x̂0(v_{k-1})) - v_{k-1})` | K |
| `extrap` | K corrective steps, total move scaled by w: `v + w (v_K - v)` | K |
| `relax` | K corrections with the gate scaled by s: `v_k = v_{k-1} + s γ ⊙ (C(x̂0(v_{k-1})) - v_{k-1})` | K |

At unit scale, relax (s = 1) reduces bit-exactly to corrective sampling, and extrap (w = 1) agrees with it up to
floating-point rounding (`src/tests/test_efficient.py`, tolerance 1e-7).

## Result

SiT-REPA-XL/2 (`sit-repa-50k`), class-conditional ImageNet 256×256, no CFG, 123 sampling steps, corrections for
t ∈ [0.5, 0.7]. NFE counts denoiser and corrector forward passes per image.

| Sampler | NFE | FID-50K | Difference to corrective, K = 5 |
| :--- | :---: | :---: | :---: |
| corrective, K = 5 | 248 | 1.291 | — |
| `relax`, K = 2, s = 3 | 173 | **1.286** | −0.005 ± 0.007 |
| `extrap`, K = 1, w = 6 | 148 | 1.312 | +0.021 ± 0.009 |

Differences are point differences of the FIDs; ± is the paired bootstrap standard error over images (30 resamples).
Over-relaxation matches five-step corrective sampling with 30% fewer network evaluations. The analysis, the
strength sweeps and the ablations are in the [blog post](https://mhh0318.github.io/blog/efficient-corrective-sampling/).

## Usage

Install a PyTorch build for your GPU (we used PyTorch 2.14 on NVIDIA B300; the upstream pin, 2.5.1, does not support
Blackwell GPUs), then:

```bash
pip install numpy pillow einops safetensors tqdm pyyaml scipy diffusers huggingface_hub accelerate pytorch-fid==0.3.0
python src/tools/prepare_evaluation.py --model sit-repa      # FID reference statistics
python -m unittest src.tests.test_efficient                  # unit-scale equivalence, CPU
```

FID-50K in four shards per run (one GPU each). The checkpoint downloads on first use (`CORRECTION_CHECKPOINT_DIR` sets
where), and the last shard to finish writes `result.json`. `--method paper` is corrective sampling:

```bash
for r in 0 1 2 3; do
  python src/efficient/evaluate.py --method paper  --k 5           --nshards 4 --shard $r --output runs/paper-k5
  python src/efficient/evaluate.py --method relax  --k 2 --scale 3 --nshards 4 --shard $r --output runs/relax-k2-s3
  python src/efficient/evaluate.py --method extrap --k 1 --scale 6 --nshards 4 --shard $r --output runs/extrap-k1-w6
done
python src/efficient/bootstrap.py runs/paper-k5 runs/relax-k2-s3 runs/extrap-k1-w6
```

## Citation

```bibtex
@misc{hu2026efficientcorrective,
  author       = {Hu, Minghui},
  title        = {Efficient Corrective Sampling by Over-Relaxation},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://mhh0318.github.io/blog/efficient-corrective-sampling/}},
  note         = {Blog post}
}
```

## Acknowledgments

This work builds on the code, models and released checkpoints of Correction Training for Diffusion Models by Donghoon
Ahn, Runqian Wang, Alexei A. Efros, Taesung Park and Yash Belhe. We thank the authors for making them public.

## License

MIT, as upstream ([LICENSE](LICENSE)); adapted components keep their notices in [licenses/](licenses).
