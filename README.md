# Efficient sampling with a trained corrector

This repository is a fork of the official code of **Correction Training for Diffusion Models** (Ahn, Wang, Efros, Park
and Belhe, 2026; [arXiv:2610.14136](https://arxiv.org/abs/2610.14136),
[project page](https://sunovivid.github.io/correction-training/),
[code](https://github.com/sunovivid/correction-training) at `475e214`). The models, training code, released
checkpoints and the original correction sampler are theirs; see the upstream README for all of them.

We add only sampling code (`src/efficient/`) that uses the released corrector differently, without any training.

## What we changed

The released SiT-REPA pair combines the corrector proposal with a learned per-channel gate,
`v_k = v_{k-1} + γ ⊙ (C(x̂0(v_{k-1})) - v_{k-1})`, and the paper applies it K = 5 times per step inside the noise
interval t ∈ [0.5, 0.7]. The learned gate is small (γ ≈ 0.036–0.052 per latent channel), so each correction adopts only
about 4% of the corrector's proposal. We treat the gate as a guidance strength instead of a fixed step:

| Method | Update inside the correction interval | Corrector calls per active step |
| :--- | :--- | :---: |
| paper | K gated corrections at the learned gate γ | K |
| `extrap` | K gated corrections, total move scaled by w: `v + w (v_K - v)` | K |
| `relax` | K corrections with the gate scaled by s (over-relaxation): `v_k = v_{k-1} + s γ ⊙ (C(x̂0(v_{k-1})) - v_{k-1})` | K |

Both reduce bit-exactly to the paper's sampler at unit scale (`src/tests/test_efficient.py`). Outside the interval every
method is the plain denoiser.

## Results

SiT-REPA-XL/2 (`sit-repa-50k`), class-conditional ImageNet 256×256, no CFG, the paper's SiT-REPA protocol
(`repa_sde`, 123 steps, corrections for t ∈ [0.5, 0.7], FID against `VIRTUAL_imagenet256_labeled.npz`). NFE counts
denoiser and corrector forward passes per image.

**FID-50K.** Differences come from a paired bootstrap over images (30 resamples; all runs share each image's noise
seed and class).

| Method | NFE | FID-50K | Difference to paper K = 5 |
| :--- | :---: | :---: | :---: |
| paper, K = 5 | 248 | 1.291 | — |
| `relax`, K = 2, s = 3 | 173 | **1.286** | −0.005 ± 0.007 |
| `extrap`, K = 1, w = 6 | 148 | 1.312 | +0.021 ± 0.009 |
| `extrap`, K = 2, w = 3 | 173 | 1.324 | +0.029 ± 0.011 |

Over-relaxation with two corrector calls matches the paper's five (the difference is within one standard error) at 70%
of the NFE. Extrapolation with a single corrector call is within 0.02 FID of the paper at 60% of the NFE. The paper's
sampler is not beaten by a clear margin: every setting we tried has its optimum at a total push of about 0.25–0.33 of
‖v‖ per step.

**FID-10K** (images 0–9999; same protocol).

| Method | NFE | FID-10K |
| :--- | :---: | :---: |
| no correction (K = 0) | 123 | 9.02 |
| no correction, 250 steps | 250 | 8.11 |
| paper, K = 1 | 148 | 7.28 |
| paper, K = 5 | 248 | 3.89 |
| `extrap`, K = 1, w = 2 / 3 / 4 / 5 / 6 / 8 | 148 | 5.92 / 4.96 / 4.36 / 4.02 / 3.91 / 3.98 |
| `extrap`, K = 2, w = 2 / 3 | 173 | 4.17 / 3.89 |
| `relax`, K = 2, s = 3 / 4 / 6 | 173 | 3.88 / 3.90 / 4.02 |
| `relax`, K = 3, s = 2 / 3 | 198 | 3.90 / 4.10 |
| `relax`, K = 4, s = 1.5 | 223 | 3.88 |
| `relax`, K = 5, s = 1.5 / 2 | 248 | 4.03 / 4.26 |

Widening the correction interval hurt: the paper's K = 5 on t ∈ [0.3, 0.9] gives FID-10K 5.68.

## Reproducing

We ran on NVIDIA B300 GPUs with Python 3.12, PyTorch 2.14.0 (CUDA 13.0), NumPy 2.5.3, SciPy 1.18.1, diffusers 0.40.0
and pytorch-fid 0.3.0. The upstream `requirements.txt` pins PyTorch 2.5.1, which does not support Blackwell GPUs;
install a PyTorch build for your GPU first, then the remaining packages:

```bash
pip install numpy pillow einops safetensors tqdm pyyaml scipy diffusers huggingface_hub accelerate pytorch-fid==0.3.0
python src/tools/prepare_evaluation.py --model sit-repa      # FID reference statistics
python -m unittest src.tests.test_efficient                  # unit-scale equivalence, CPU
```

FID-50K, four shards per run (one GPU each; the checkpoint downloads on first use, `CORRECTION_CHECKPOINT_DIR` sets
where). The last shard to finish writes `result.json`:

```bash
for r in 0 1 2 3; do
  python src/efficient/evaluate.py --method paper  --k 5           --nshards 4 --shard $r --output runs/paper-k5
  python src/efficient/evaluate.py --method relax  --k 2 --scale 3 --nshards 4 --shard $r --output runs/relax-k2-s3
  python src/efficient/evaluate.py --method extrap --k 1 --scale 6 --nshards 4 --shard $r --output runs/extrap-k1-w6
  python src/efficient/evaluate.py --method extrap --k 2 --scale 3 --nshards 4 --shard $r --output runs/extrap-k2-w3
done
python src/efficient/bootstrap.py runs/paper-k5 runs/relax-k2-s3 runs/extrap-k1-w6 runs/extrap-k2-w3
```

FID-10K rows use `--samples 10000` with one shard (`--method base` for K = 0, plus `--steps 250` for the 250-step row;
`--correction-interval 0.3 0.9` for the wider interval). Every shard seeds the sampler's path noise with 220922, and
image i uses CPU noise seed i and class i mod 1000. With batch size 64 on B300 GPUs, re-running this code reproduced
the Inception features of our original runs bit for bit (checked on images 0–63 for paper K = 5, `relax` K = 2 s = 3
and `extrap` K = 1 w = 6). Other GPUs or batch sizes change the stochastic path noise and the numbers slightly.

## Acknowledgments

This work builds on the code, models and released checkpoints of
[Correction Training for Diffusion Models](https://arxiv.org/abs/2610.14136) by Donghoon Ahn, Runqian Wang,
Alexei A. Efros, Taesung Park and Yash Belhe ([code](https://github.com/sunovivid/correction-training)). We thank the
authors for making them public.

## License

MIT, as upstream ([LICENSE](LICENSE)); adapted components keep their notices in [licenses/](licenses).
