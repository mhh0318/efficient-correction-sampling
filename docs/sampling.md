# Sampling

Run commands from the repository root. Checkpoints download automatically;
see [checkpoints](checkpoints.md) for model selection and local weights.
With an official `--checkpoint` name and no `--config`, sampling uses the
corresponding qualitative preset below. Local bundles use their saved recipe.

## Visual sampling

The default recipes below use correction without CFG. SiT-REPA uses its
100K checkpoint.

**K in the paper = `--corrections_per_step`: correction iterations within each
active denoising step.** `--denoising_steps` controls the outer sampling loop.

```bash
python src/sample.py --config configs/eval/qualitative/sit-repa.yml --output samples/repa
python src/sample.py --config configs/eval/qualitative/sit.yml --output samples/sit
python src/sample.py --config configs/eval/qualitative/jit.yml --output samples/jit
```

The paper's baseline comparisons use approximately matched NFEs. Larger K can
produce better images, so these sampling presets do not match NFEs. For the
paper settings, see [Appendix B.1](https://correction-training.github.io/paper.pdf#page=23)
and the [evaluation guide](evaluation.md).

Sampling configs: [SiT-REPA](../configs/eval/qualitative/sit-repa.yml),
[SiT](../configs/eval/qualitative/sit.yml), [JiT](../configs/eval/qualitative/jit.yml).

For unguided SiT and SiT-REPA with correction throughout the trajectory,
NFEs = denoising steps × (1 + K). JiT's Heun solver and correction interval
change this count. CFG also adds network calls for the unconditional branch.
The progress bar counts denoising steps. Overriding K or an interval does not
automatically adjust denoising steps to maintain the NFE budget.

Each command generates 16 golden retrievers (class 207), in batches of four,
and saves individual images, a 4 × 4 grid, and the actual
settings in `sampling.json`. `--output` selects a parent directory; every run
creates a new timestamped subdirectory and prints its path. Repeating the same
command preserves earlier runs. Without `--output`, the parent directory is
`samples/<checkpoint>/`.

```text
samples/repa/
├── 20261014-153012/
│   ├── grid.png
│   ├── 000-class207.png
│   ├── ...
│   └── sampling.json
└── 20261014-154205/
    └── ...
```
During sampling, a progress bar shows denoising steps and ETA for the current
sample or batch, for example `Samples 1-4/16: 21/42`.

## Customize samples

CLI arguments override the YAML defaults. The same options work with JiT,
SiT, and SiT-REPA; use the YAML preset for the corresponding model.

| Option | What it controls | Example |
| :--- | :--- | :--- |
| `--checkpoint` | Official checkpoint name or local bundle | `--checkpoint sit-repa-50k` |
| `--classes` | [ImageNet class IDs](imagenet-classes.md); repeated in order if more samples are requested | `--classes 207 388` |
| `--unconditional` | Generate with the null class embedding, without CFG | `--unconditional` |
| `--samples` | Total images; qualitative presets default to 16 | `--samples 16` |
| `--batch-size` | Images processed together; defaults to 4 | `--batch-size 8` |
| `--grid-columns` | Columns in the saved image grid; defaults to 4 | `--grid-columns 3` |
| `--seed` | Random seed; defaults to the YAML value, or 0 | `--seed 42` |
| `--denoising_steps` | Number of denoising steps | `--denoising_steps 42` |
| `--corrections_per_step` | Correction iterations K per active denoising step; 0 disables correction | `--corrections_per_step 5` |
| `--correction-interval` | Correction start t, end t (1 = noise, 0 = data) | `--correction-interval 0.7 0.5` |
| `--cfg` | CFG scale; values other than 1 require K=0 | `--cfg 3` |
| `--cfg-interval` | CFG start t, end t (1 = noise, 0 = data) | `--cfg-interval 0.8 0.1` |
| `--precision` | Transformer forward precision | `--precision bf16` |
| `--compile` | Compile transformer blocks | `--compile` |
| `--checkpoint-dir` | Checkpoint download/cache directory | `--checkpoint-dir /data/checkpoints` |
| `--output` | Parent directory for timestamped runs; defaults to `samples/<checkpoint>/` | `--output samples/dogs` |

Correction and CFG intervals are written as **[start t, end t]**, following
sampling from noise to data: t=1 is pure noise and t=0 is clean data.
This order applies to both YAML configs and CLI arguments for every model. For example, `--correction-interval 1.0 0.8`
applies correction near the start of sampling.
Batch size controls memory use independently of the total sample count.

### Classes, sample count, and seed

The presets generate 16 golden retrievers (class 207), four at a time.
Use class 250 for Siberian huskies:

```bash
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --classes 250 --samples 16 --batch-size 4 --seed 42 --output samples/huskies
```

With `--classes 207 388 --samples 8`, classes alternate: 207, 388, 207, 388, ...
To try another random draw, change `--seed`. Keep the seed, classes, batch size,
and precision fixed when comparing sampling settings.

### Compare 50K and 100K checkpoints

The qualitative SiT-REPA config selects 100K weights. Change only the checkpoint
to compare the two training snapshots with identical sampling settings:

```bash
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --checkpoint sit-repa-50k --classes 207 --seed 42 --output samples/repa-50k

python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --checkpoint sit-repa-100k --classes 207 --seed 42 --output samples/repa-100k
```

50K and 100K refer to training updates. `--samples` controls how many images to generate.

### Denoising steps and repeated correction

Use `--denoising_steps` for the outer denoising loop and
`--corrections_per_step` for K. Compare correction counts at approximately
250 NFEs with the same checkpoint, seed, classes, and batch size:

```bash
for k in 0 2 4 6; do
  python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
    --classes 207 --seed 42 --denoising_steps "$((250 / (1 + k)))" --corrections_per_step "$k" --output "samples/k${k}"
done
```

To apply four corrections only within a selected noise interval (248 NFEs):

```bash
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --classes 207 --seed 42 --denoising_steps 136 --corrections_per_step 4 \
  --correction-interval 0.7 0.5 \
  --output samples/correction-interval
```

### Baseline: no guidance, no correction

Set `--corrections_per_step 0 --cfg 1` to sample the denoiser in the same
checkpoint without correction or guidance. The commands below keep the
default 16 golden retrievers, seed 0, and batch size 4, at approximately
250 NFEs per image:

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

To disable correction while keeping the original denoising-step count,
omit `--denoising_steps`. That comparison uses fewer NFEs.

### Comparison grids

**Compare baseline, CFG, and correction.** Run the [example script](../src/examples/compare_sampling.sh):

```bash
bash src/examples/compare_sampling.sh
```

The script uses the same four initial noises and Siberian husky class for all
three rows, at approximately 250 NFEs per image. It saves the grid to `samples/compare.png`.

**Sweep CFG scale.** Run the [example script](../src/examples/sweep_cfg.sh) for w=1, 3, 5, 7:

```bash
bash src/examples/sweep_cfg.sh
```

Each setting generates a 3 × 3 husky grid from the same nine initial noises.
The grids are arranged side by side in `samples/cfg-scales.png`, with 125 denoising steps fixed.
The baseline uses 125 NFEs; guided rows use 250 NFEs.

**Sweep correction count K.** Run the [example script](../src/examples/sweep_corrections.sh)
for K=0, 1, 2, 4, 6 at approximately 250 NFEs per image:

```bash
bash src/examples/sweep_corrections.sh
```

Each setting generates a 3 × 3 husky grid from the same nine initial noises.
The grids are arranged side by side in `samples/correction-counts.png`.

`make_grid.py` stacks each input's `grid.png`, using the latest timestamped run
when given a parent directory. Pass exact run directories to select older results.
Use `--samples 16` in the sampling commands for larger comparisons.

### Conditional, unconditional, and CFG

The default is class-conditional sampling with CFG off (`--cfg 1`).
For unconditional generation, select the dedicated `sit-repa-uncond-100k`
checkpoint used for the paper's unconditional examples. `--unconditional` uses
the null class embedding and disables CFG:

```bash
python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
  --checkpoint sit-repa-uncond-100k --unconditional --samples 16 --seed 42 \
  --output samples/unconditional
```

You can also use `--config configs/eval/qualitative/sit-repa-uncond.yml`.
That preset selects the unconditional checkpoint and null conditioning automatically.

CFG and correction can be combined, but this ImageNet implementation does not
support their simultaneous use because our ImageNet experiments did not use it.
Use `--cfg 1` with correction, or `--corrections_per_step 0` with CFG.
Support for combining them is planned for text-to-image.

### YAML defaults and local weights

You can also set generation defaults in your sampling YAML:

```yaml
generation:
  runner: sample
  seed: 42
  classes: [207, 388]
  samples: 16
  batch_size: 4
  unconditional: false
```

Sampling YAML uses the same names as the CLI:

```yaml
sampling:
  method: repa_sde
  precision: fp32
  denoising_steps: 42
  corrections_per_step: 5
  correction_interval: [1.0, 0.0] # [correction start t, correction end t], t=0.0 is data, t=1.0 is noise
  cfg: 1.0
  cfg_interval: [1.0, 0.0]
```

`--classes` selects conditional generation even if the YAML sets
`unconditional: true`. `--no-unconditional` also restores conditional sampling.

Use `--checkpoint /path/to/bundle` for your own trained weights. The full CLI
is available with `python src/sample.py --help`.

## Quantitative sampling recipes

These configs use approximately matched network-evaluation budgets without CFG.
SiT-REPA uses the 50K checkpoint. Passing them to `src/sample.py` previews four
images; the [evaluation guide](evaluation.md) covers 50K generation and metrics.

```bash
python src/sample.py --config configs/eval/quantitative/jit.yml --output samples/jit-eval
python src/sample.py --config configs/eval/quantitative/sit.yml --output samples/sit-eval
python src/sample.py --config configs/eval/quantitative/sit-repa.yml --output samples/repa-eval
```

| Model | Sampler | Steps | K | Correction interval (start → end) | D + C calls |
| :--- | :--- | ---: | ---: | :--- | ---: |
| JiT-L/16 | Heun + final Euler | 45 | 9 | 1.0 → 0.8 | 242 |
| SiT-XL/2 | SiT SDE | 83 | 2 | Full trajectory | 249 |
| SiT-REPA-XL/2, 50K | REPA SDE | 123 | 5 | 0.7 → 0.5 | 248 |

Noise fraction 1 means pure noise; 0 means clean data. Sampler, intervals,
and precision are supplied by each config. JiT uses BF16 network forwards;
SiT/SiT-REPA use FP32 networks and VAE, with TF32 enabled on CUDA and FP64
integration for REPA. Checkpoint metadata and `sampling.json` store the native
sampler settings; YAML and CLI intervals use [start t, end t], from noise to data.
