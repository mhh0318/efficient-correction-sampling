<h1 align="center">Correction Training for Diffusion Models</h1>

<p align="center">
  <a href="https://correction-training.github.io/donghoon/">Donghoon Ahn</a><sup>1</sup> &nbsp;
  <a href="https://raywang4.github.io/">Runqian Wang</a><sup>1</sup> &nbsp;
  <a href="https://people.eecs.berkeley.edu/~efros/">Alexei A. Efros</a><sup>1</sup> &nbsp;
  <a href="https://taesung.me/">Taesung Park</a><sup>2</sup> &nbsp;
  <a href="https://yashbelhe.github.io/">Yash Belhe</a><sup>2</sup>
  <br>
  <sup>1</sup> UC Berkeley &emsp; <sup>2</sup> Reve
</p>

<p align="center">
  <a href="https://correction-training.github.io/paper.pdf?v=arxiv-v2-be01855b"><img src="assets/badges/paper.svg" alt="Paper: PDF"></a>
  <a href="https://arxiv.org/abs/2610.14136"><img src="assets/badges/arxiv.svg" alt="arXiv: 2610.14136"></a>
  <a href="https://correction-training.github.io/"><img src="assets/badges/project.svg" alt="Project page"></a>
  <a href="https://github.com/sunovivid/correction-training/releases/tag/imagenet-checkpoints-v1"><img src="assets/badges/checkpoints.svg" alt="Checkpoints: Available"></a>
</p>

<p align="center"><b>No guidance needed.</b></p>

<p align="center">
  Correction achieves <b>2.07 FID with DiT, 1.92 with SiT, and 2.96 with JiT</b><br>
  on class-conditional ImageNet-256, <b>without classifier-free guidance</b>.
</p>

<p align="center">
  <img src="assets/samples.png" width="100%" alt="ImageNet samples generated with correction: class-conditional generation without CFG, and unconditional generation.">
</p>

**Correction training** adds a network that refines the denoiser's prediction using the same clean target. During sampling, we reuse the corrector within each denoising step.

## Method

[![Watch the training and sampling animation: an outer denoising loop and an inner correction loop.](assets/method.png)](https://correction-training.github.io/assets/motion/method-steps.mp4)

The corrector learns to recover the clean target from the denoiser's prediction, with stop-gradient through that prediction. At inference, its output can be corrected again before constructing the next noisy state.

## Installation

This implementation supports **JiT, SiT, and SiT-REPA**. DiT and text-to-image code are coming soon.

Sampling requires an NVIDIA GPU. On a cluster, run the sampling commands on a GPU node.

```bash
git clone https://github.com/sunovivid/correction-training.git
cd correction-training
conda create -n correction python=3.11 pip -y --override-channels -c conda-forge # Use these flags to avoid Anaconda's ToS prompt.
conda activate correction
python -m pip install -r requirements-latent.txt
```

`requirements-latent.txt` includes `requirements.txt` and installs the sampling and training dependencies for **all three models: JiT, SiT, and SiT-REPA**. For a JiT-only setup, `requirements.txt` is sufficient.

Activate `correction` before running sampling, training, or evaluation commands.

<details>
<summary>Using venv</summary>

With Python 3.11 installed, run these commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-latent.txt
```

</details>

## Sampling

Start with **16 golden retrievers** in a 4 × 4 grid:

```bash
# SiT-REPA-XL/2 (try this first!)
python src/sample.py --config configs/eval/qualitative/sit-repa.yml --output samples/sit-repa-100k
# SiT-XL/2
python src/sample.py --config configs/eval/qualitative/sit.yml --output samples/sit
# JiT-L/16
python src/sample.py --config configs/eval/qualitative/jit.yml --output samples/jit
```

This will automatically download checkpoints from the [public Release](https://github.com/sunovivid/correction-training/releases/tag/imagenet-checkpoints-v1) on first use.
Note: Download speeds sometimes decrease significantly during downloads (< 1 MB/s). If this happens, stopping the job with `Ctrl+C` and then rerunning it helps.

To store checkpoints elsewhere, set `export CORRECTION_CHECKPOINT_DIR=/path/to/checkpoints` before running commands. The default is `checkpoints/`. (Allow about 11 GB of disk space per SiT/SiT-REPA checkpoint for the download parts and assembled weights.)

The paper's baseline comparisons use approximately matched NFEs. Larger K can
produce better images, so we use larger K for default sampling configuration. For the
paper settings, see [Appendix B.1](https://correction-training.github.io/paper.pdf#page=23)
and the [evaluation guide](docs/evaluation.md). (Sampling configs: [SiT-REPA](configs/eval/qualitative/sit-repa.yml),
[SiT](configs/eval/qualitative/sit.yml), [JiT](configs/eval/qualitative/jit.yml))


<br>
You can override some options in the default configuration. Some of the main arguments you may want to modify are listed below:

<details>
<summary>Sampling arguments</summary>

| Argument | Role | Example | Default |
| :--- | :--- | :--- | :--- |
| `--checkpoint` | Select weights or a local bundle | `sit-repa-50k` | From config |
| `--classes` | [ImageNet class IDs](docs/imagenet-classes.md) | `207 250` | `207` (golden retriever) |
| `--samples` | Total images | `8` | `16` |
| `--batch-size` | Images per batch | `8` | `4` |
| `--seed` | Random seed | `42` | `0` |
| `--denoising_steps` | Outer denoising steps | `42` | From config |
| `--corrections_per_step` | **K in the paper**: corrections per active denoising step | `8` | From config |
| `--correction-interval` | Correction start t, end t (1 = noise, 0 = data) | `0.7 0.5` | From config |
| `--cfg` | CFG scale; values other than 1 require K=0 | `3` | `1` |
| `--output` | Parent directory for timestamped runs | `samples/dogs` | `samples/<checkpoint>/` |

See the [full sampling options table](docs/sampling.md#customize-samples) for all options.

</details>

<br>
Also check out some fun examples of sweeping K and comparing correction with CFG!

<details>
<summary>Sampling examples</summary>

CLI arguments override the YAML defaults. For example:

```bash
# Siberian huskies: 16 images, a fixed seed, 4 images per batch
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --classes 250 --samples 16 --seed 42 --batch-size 4 --output samples/huskies

# Use the 50K checkpoint with the same sampling recipe (the default is 100K)
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --checkpoint sit-repa-50k --classes 207 --seed 42 --output samples/dogs-50k

# Change denoising steps, correction count K, and interval (248 NFEs)
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --denoising_steps 136 --corrections_per_step 4 --correction-interval 0.7 0.5 --output samples/k4

# Unconditional generation
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --checkpoint sit-repa-uncond-100k --unconditional --samples 16 --seed 42 \
  --output samples/unconditional

```

**Baseline (no guidance, no correction).** Use the same checkpoint with
K=0 and CFG=1. These commands generate the same 16 golden retrievers with
seed 0 and batch size 4, at approximately 250 NFEs:

```bash
# SiT-REPA: 250 NFEs
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --denoising_steps 250 --corrections_per_step 0 --cfg 1 --output samples/sit-repa-baseline

# SiT: 250 NFEs
python src/sample.py --config configs/eval/qualitative/sit.yml \
  --denoising_steps 250 --corrections_per_step 0 --cfg 1 --output samples/sit-baseline

# JiT: 249 NFEs (Heun)
python src/sample.py --config configs/eval/qualitative/jit.yml \
  --denoising_steps 125 --corrections_per_step 0 --cfg 1 --output samples/jit-baseline
```

**Compare baseline, CFG, and correction.** Run the [example script](src/examples/compare_sampling.sh):

```bash
bash src/examples/compare_sampling.sh
```

The script uses the same four initial noises and Siberian husky class for all
three rows, at approximately 250 NFEs per image. It saves the grid to `samples/compare.png`.

**Sweep CFG scale.** Run the [example script](src/examples/sweep_cfg.sh) for w=1, 3, 5, 7:

```bash
bash src/examples/sweep_cfg.sh
```

**Sweep correction count K.** Run the [example script](src/examples/sweep_corrections.sh)
for K=0, 1, 2, 4, 6 at approximately 250 NFEs per image:

```bash
bash src/examples/sweep_corrections.sh
```

</details>

## Checkpoints

| Name | Model | Updates | Use |
| :--- | :--- | ---: | :--- |
| `sit-repa-50k` | SiT-REPA-XL/2 | 50K | Main-table evaluation |
| `sit-repa-100k` | SiT-REPA-XL/2 | 100K | Visual sampling |
| `sit-repa-uncond-100k` | SiT-REPA-XL/2 | 100K | Unconditional sampling |
| `sit` | SiT-XL/2 | 100K | Evaluation and sampling |
| `jit` | JiT-L/16 | 100K | Evaluation and sampling |

Note that these checkpoints will be automatically downloaded when the corresponding [sampling](#sampling) or [evaluation](#evaluation) code runs on first use (recommended). Alternatively, you can manually download it from [public Release](https://github.com/sunovivid/correction-training/releases/tag/imagenet-checkpoints-v1).

Bundles include both networks and their trained corrector configuration; no conversion is needed. See [checkpoint settings](docs/checkpoints.md) for parameterizations, gates, and downloads.

## Training and evaluation

> [!NOTE]
> Training, checkpointing, resuming, and evaluation have been tested on short runs.
> Full training-to-evaluation reproduction is still in progress.

First [prepare ImageNet](docs/data.md). For an installation check, follow the
[two-update training and resume example](docs/usage.md#first-training-run) on one GPU.
For multi-GPU startup hangs, see [distributed startup](docs/usage.md#distributed-startup).

To train correction from the official pretrained SiT denoiser:

```bash
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/train/sit.yml --init-base sit \
  --data /path/to/imagenet --output runs/sit
```

The official base weights download automatically and are converted on load.
For SiT-REPA, use `--config configs/train/sit-repa.yml --init-base sit-repa`.
Use `--init-pair sit` to fine-tune our released denoiser–corrector pair.

The joint-training recipes save a restart checkpoint every 5K updates and an EMA bundle
every 10K updates. Choose an output drive with about 100 GB free for a 100K run,
including temporary space when replacing a restart checkpoint.

- [Data preparation](docs/data.md): ImageNet download, preprocessing, and optional VAE caching.
- [Training](docs/usage.md): model configs, pretrained initialization, fine-tuning, and resume.
- [Evaluation](docs/evaluation.md): 50K generation, FID, precision, and recall.
- [Configs](configs/README.md): model-specific training, sampling, and evaluation recipes.

<a id="evaluation"></a>

Generate evaluation samples with the same command for all three models:

```bash
torchrun --standalone --nproc_per_node=8 src/evaluate.py --model sit --output runs/sit-50k
```

Use `--model jit` or `--model sit-repa` for the other checkpoints. The
[evaluation guide](docs/evaluation.md) provides evaluator installation, reference
downloads, and FID / precision / recall commands for each model.

See the [documentation index](docs/README.md) for all guides.

## ImageNet results

**High-quality generation without CFG.** Correction improves FID over the unguided baselines at approximately the same sampling budget as the CFG references.

| Model | Base FID ↓ | Correction, no CFG ↓ | CFG reference ↓ |
| :--- | ---: | ---: | ---: |
| DiT-XL/2 | 9.58 | **2.07** | 2.51 |
| SiT-XL/2 | 8.55 | **1.92** | 2.11 |
| SiT-REPA-XL/2 | 5.53 | **1.31** | 1.35 |
| JiT-L/16 | 14.86 | **2.96** | 2.35 |

*ImageNet-256, FID@50K, approximately 250 network evaluations per image, including denoiser, corrector, and CFG calls.*

<details>
<summary>Control refinement with K</summary>

![Final SiT-REPA samples across K, with DiT FID and precision–recall curves.](assets/imagenet-k-sweep.png)

Final generations share the same initial noise.

</details>

### More diversity, less oversaturation than strong CFG

![SiT-REPA husky samples: CFG on top, correction below, with the strongest settings highlighted.](assets/imagenet-diversity.png)

Same class and initial noises, at approximately matched NFE. Top: CFG. Bottom: correction without CFG. Correction retains varied poses, framing, backgrounds, and colors.

## Text-to-image: correction also works with CFG

Correction can reduce oversaturation and repetitive compositions while preserving variety across seeds.

![CFG plus correction produces more varied bench designs and viewpoints.](assets/cfg-plus-correction.png)

*Same prompt and eight seeds. Top: no correction; bottom: K=6. Left: no CFG; right: CFG (w=6).*

| Setting | GenEval ↑ | DPG-Bench ↑ |
| :--- | ---: | ---: |
| CFG | 68.08 | 83.38 |
| CFG + correction | **70.48** | **83.69** |

*NanoGen-256, w=4, approximately matched sampling budget: 250 vs. 246 network evaluations.*

[More results](docs/results.md) include unguided text-to-image, human preferences, and additional K sweeps. **Text-to-image code is coming soon.**

## Citation

```bibtex
@misc{ahn2026correction,
  title  = {Correction Training for Diffusion Models},
  author = {Ahn, Donghoon and Wang, Runqian and Efros, Alexei A. and Park, Taesung and Belhe, Yash},
  year   = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). Adapted components retain their original copyright and license notices in [licenses/](licenses).

## Acknowledgments

We build on [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA), [LightningDiT](https://github.com/hustvl/LightningDiT), and [JiT](https://github.com/LTH14/JiT). Their original license notices are preserved in [licenses/](licenses).
