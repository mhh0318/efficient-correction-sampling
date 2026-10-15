# Checkpoints

Each bundle includes the trained denoiser, corrector, and sampling configuration.
Models generate ImageNet-256 images. `sit-repa-uncond-100k` is the dedicated
unconditional model; the other checkpoints are class-conditional.

| Name | Model | Updates | Weights (GB) | Use |
| :--- | :--- | ---: | ---: | :--- |
| `jit` | JiT-L/16 | 100K | 3.67 | Evaluation and sampling |
| `sit` | SiT-XL/2 | 100K | 5.40 | Evaluation and sampling |
| `sit-repa-50k` | SiT-REPA-XL/2 | 50K | 5.40 | Main-table evaluation |
| `sit-repa-100k` | SiT-REPA-XL/2 | 100K | 5.40 | Visual sampling |
| `sit-repa-uncond-100k` | SiT-REPA-XL/2 | 100K | 5.40 | Unconditional sampling |

The two conditional SiT-REPA checkpoints share the same architecture and corrector
parameterization, with weights saved at different training updates.
Their evaluation and visual-sampling recipes are separate.

## Download

Sampling downloads the selected checkpoint automatically:

```bash
python src/sample.py --config configs/eval/qualitative/sit.yml --output samples/sit
```

To download ahead of time, including for offline use:

```bash
python src/utils/download.py sit  # Or: jit / sit-repa-50k / sit-repa-100k / sit-repa-uncond-100k
```

Weights are cached in `checkpoints/NAME/` and checked automatically before use.
Set `CORRECTION_CHECKPOINT_DIR` to use another location for sampling, evaluation,
and distillation teachers:

```bash
export CORRECTION_CHECKPOINT_DIR=/path/to/checkpoints
```

Use `--checkpoint-dir DIR` with sampling or evaluation, or `--output DIR` with
the downloader, to override this location for one command.
Interrupted downloads resume when rerun.
No GitHub login is required.

With an official checkpoint name and no `--config`, `src/sample.py` uses the
model's qualitative preset (approximately 250 NFEs). For example:

```bash
python src/sample.py --checkpoint sit-repa-100k --output samples/repa
```

A local bundle also works: `--checkpoint /path/to/bundle`, using its saved
sampling recipe unless a config or CLI override is supplied.
SiT and SiT-REPA download their VAE on first use.

## Corrector settings

These settings and the learned gate load automatically from the bundle.
`x` means clean-image prediction; `v` means velocity. SiT models operate in
VAE latents, while JiT operates in pixels.

| Checkpoint | Denoiser output | Corrector input → output | Combination | Gate |
| :--- | :--- | :--- | :--- | :--- |
| `jit` | x | x → x | `direct` | None |
| `sit` | v | x → v | `gated`, `blend` | Learned scalar |
| `sit-repa-50k`, `sit-repa-100k`, `sit-repa-uncond-100k` | v | x → v | `gated`, `difference` | Learned per-channel values |

For sampling recipes, see [sampling](sampling.md). To fine-tune a bundle,
use `--init-pair`; to resume an existing training run, use `--resume` with its
restart checkpoint. See [training](usage.md).
