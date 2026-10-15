# Training

Run commands from the repository root after [installation](../README.md#installation).
Training requires CUDA and ImageNet-256; see [data preparation](data.md).

## Model configurations

| Model | Training config | Learning rate | Additional updates |
| :--- | :--- | ---: | ---: |
| JiT-L/16 | [jit.yml](../configs/train/jit.yml) | 5e-5 | 100K |
| SiT-XL/2 | [sit.yml](../configs/train/sit.yml) | 1e-4 | 100K |
| SiT-REPA-XL/2 | [sit-repa.yml](../configs/train/sit-repa.yml) | 1e-4 | 100K |

These configs train the denoiser and corrector jointly, with one correction
per training input. REPA's 50K and 100K checkpoints use the same training config.

FP32 SiT training enables TF32, following the official SiT training code.

## First training run

First [install the dependencies](../README.md#installation) and
[prepare ImageNet](data.md). Replace `/path/to/imagenet` with the extracted
training directory or indexed tar directory. The SiT examples below use eight
GPUs; `--nproc_per_node` is the number of GPUs on your node.

For a short first run on one 80 GB GPU with at least 32 GB of host memory,
use the same SiT-REPA model with a smaller batch and two optimizer updates.
Choose an output drive with about 100 GB free for checkpoints and temporary saves:

```bash
torchrun --standalone --nproc_per_node=1 src/train.py \
  --config configs/train/sit-repa.yml --init-base sit-repa \
  --data /path/to/imagenet --output runs/repa-first \
  --global-batch-size 2 --steps 2 --save-every 2 --export-every 2
```

The base model and VAE download automatically on first use. This runs real
training and saves both a restart checkpoint and an EMA model:

```text
runs/repa-first/
├── training.json
├── performance.json
├── progress.json
├── checkpoint-last.pt
└── step-0000002/
    ├── config.json
    └── model.safetensors
```

Continue the same run for one more update:

```bash
torchrun --standalone --nproc_per_node=1 src/train.py \
  --config configs/train/sit-repa.yml --resume \
  --data /path/to/imagenet --output runs/repa-first \
  --global-batch-size 2 --steps 3 --save-every 2 --export-every 2
```

`--steps` is the total update count, so this resumes at update 2 and stops at 3.
Sample the exported EMA model by passing its directory:

```bash
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --checkpoint runs/repa-first/step-0000003 --output samples/repa-first
```

Two updates check the installation and training workflow; use the full training
commands below to train a corrector for sample quality.

## Start from a pretrained denoiser

SiT and SiT-REPA download the official pretrained denoiser when you specify
`--init-base sit` or `--init-base sit-repa`. The corrector is initialized from
that denoiser. Downloads show progress and are cached under
`CORRECTION_CHECKPOINT_DIR/pretrained/`, or `checkpoints/pretrained/` by default.
Weights are converted in memory when loaded; no separate conversion command is needed.

SiT uses a [pinned Hugging Face copy](https://huggingface.co/sairights/sit-xl-2-256x256-sde-cfg/tree/5a15dccbe5710eb5bc9a0b0f71e6fb3deb31e24e)
of the original weights, checked tensor by tensor against the official checkpoint.
SiT-REPA uses [REPA's official download](https://github.com/sihyun-yu/REPA/blob/main/utils.py).
Previously cached SiT `.pt` files continue to work.

For JiT, obtain the [official base weights](https://github.com/LTH14/JiT#evaluation)
and convert once as shown below.

### JiT-L/16

```bash
python src/tools/convert_checkpoint.py /path/to/jit-l-16.pth checkpoints/jit-init
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/train/jit.yml --init-base checkpoints/jit-init \
  --data /path/to/imagenet --output runs/jit
```

### SiT-XL/2

```bash
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/train/sit.yml --init-base sit \
  --data /path/to/imagenet --output runs/sit
```

### SiT-REPA-XL/2

```bash
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/train/sit-repa.yml --init-base sit-repa \
  --data /path/to/imagenet --output runs/repa
```

To use an already downloaded SiT/REPA file, replace the name with its path:
`--init-base /path/to/model.pt` (or `.safetensors`). Nested `ema` weights are
selected automatically. A converted local bundle also works:
`--init-base /path/to/base-bundle`.

Latent training can also use the [packed VAE cache](data.md#optional-packed-vae-cache)
in place of the image directory.

## Fine-tune a trained pair

Use `--init-pair` to keep its trained denoiser, corrector,
and gate while starting a new optimizer run:

```bash
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/train/sit.yml --init-pair sit \
  --data /path/to/imagenet --output runs/sit-finetune
```

Official names download automatically to `CORRECTION_CHECKPOINT_DIR` or
`checkpoints/`. A local bundle path also works with `--init-pair`.

## Resume training

Use the same model and optimizer settings, effective batch size, GPU count,
compile setting, and output directory:

```bash
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/train/sit.yml --resume \
  --data /path/to/imagenet --output runs/sit-finetune
```

`--resume` restores optimizer, EMAs, RNG, and data position. Existing output
directories are not resumed implicitly. `--init-base` and `--init-pair` start
new runs; omitting both starts from random weights.

Set `training.deterministic=true` for deterministic training and resume.

`--global-batch-size` overrides the effective batch size and must be divisible
by GPU count times accumulation. `performance.json` records training speed
and ETA, excluding checkpoint writes. Restart checkpoints use `training.save_every`; separate EMA exports
use `training.export_every`.

The joint-training recipes overwrite `checkpoint-last.pt` every 5K updates
and export an EMA bundle every 10K updates. They also save at the final update.
`SIGUSR1` or `SIGTERM` requests a restart checkpoint after the current update.
For 100K updates, allow about
76 GB for one restart checkpoint and ten EMA exports, plus temporary space
while replacing the restart checkpoint.

Use `--steps`, `--save-every`, and `--export-every` to override these values
without editing a config. These runtime controls can also change on resume.

## Distributed startup

If a multi-GPU run hangs before its first update, try disabling NCCL GPU
peer-to-peer communication, then repeat the same training command:

```bash
export NCCL_P2P_DISABLE=1
```

## Further guides

- [Sampling](sampling.md): generate images from released or newly trained bundles.
- [Evaluation](evaluation.md): compute FID, precision, and recall.
- [Distillation](distillation.md): train a single denoiser from a correction teacher.
- [Implementation](implementation.md): core modules and tests.
