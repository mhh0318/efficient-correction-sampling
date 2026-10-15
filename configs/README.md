# Configuration files

Training configs define the model, corrector, and optimizer. Evaluation configs
select the checkpoint and sampling recipe.

Edit the YAML (`.yml`) files to change defaults; explicit CLI arguments take
priority. Sampling uses `denoising_steps` and `corrections_per_step` (K),
matching the CLI. Correction and CFG intervals are **[start t, end t]** in
sampling order, from noise (t=1) toward data (t=0), for every model. Training `steps` counts optimizer updates.
Checkpoint metadata and run records retain their native settings in JSON.

| Model | Training | Quantitative evaluation | Qualitative sampling |
| :--- | :--- | :--- | :--- |
| JiT-L/16 | [train/jit.yml](train/jit.yml) | [eval/quantitative/jit.yml](eval/quantitative/jit.yml) | [eval/qualitative/jit.yml](eval/qualitative/jit.yml) |
| SiT-XL/2 | [train/sit.yml](train/sit.yml) | [eval/quantitative/sit.yml](eval/quantitative/sit.yml) | [eval/qualitative/sit.yml](eval/qualitative/sit.yml) |
| SiT-REPA-XL/2 | [train/sit-repa.yml](train/sit-repa.yml) | [eval/quantitative/sit-repa.yml](eval/quantitative/sit-repa.yml), 50K weights | [eval/qualitative/sit-repa.yml](eval/qualitative/sit-repa.yml), 100K weights |

The [unconditional sampling preset](eval/qualitative/sit-repa-uncond.yml) selects
`sit-repa-uncond-100k` with 42 denoising steps and K=5 (252 NFEs).

SiT-REPA uses one training config. The 50K and 100K checkpoints share its model
and corrector settings; evaluation uses 50K weights and visual sampling uses 100K.

## Training

From the official pretrained SiT denoiser (downloaded automatically):

```bash
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/train/sit.yml --init-base sit \
  --data /path/to/imagenet --output runs/sit
```

Use `--init-pair` to fine-tune a trained pair and `--resume` to restore a run.
See [training](../docs/usage.md) for initialization and [data preparation](../docs/data.md).

## Sampling and evaluation

```bash
python src/sample.py --config configs/eval/qualitative/sit-repa.yml --output samples/repa
```

The config downloads its checkpoint automatically. CLI options override sampling
settings. See [sampling](../docs/sampling.md) for qualitative defaults, NFE counts,
seeds, classes, and K.
Use `--checkpoint /path/to/bundle` to apply a sampling recipe to your own trained weights.

Quantitative configs generate 50K images through `src/evaluate.py`. See [evaluation](../docs/evaluation.md) for generation and metric commands.

## Distillation

[Distillation recipes](distill/) train a single denoiser from a correction
teacher. See the [distillation guide](../docs/distillation.md).
