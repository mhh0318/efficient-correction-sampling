# Documentation

Start with [installation and sampling](../README.md#installation).
All commands run from the repository root.

| Guide | Contents |
| :--- | :--- |
| [Sampling](sampling.md) | Seeds, classes, K, qualitative recipes, and quantitative recipes |
| [Checkpoints](checkpoints.md) | Available weights, parameterizations, trained gates, and downloads |
| [Data preparation](data.md) | ImageNet download, standard preprocessing, flips, and packed VAE caches |
| [Training](usage.md) | JiT/SiT/SiT-REPA configs, pretrained initialization, fine-tuning, and resume |
| [Evaluation](evaluation.md) | 50K generation and model-specific paper FID, precision, and recall |
| [Implementation](implementation.md) | Core modules, training/sampling flow, and CPU tests |
| [Distillation](distillation.md) | Train a single denoiser from a frozen denoiser–corrector teacher |
| [Additional results](results.md) | More K sweeps, text-to-image metrics, and human preferences |

## Repository map

- [src/](../src): models, datasets, samplers, training, and sampling entry points.
- [configs/](../configs/README.md): model training and sampling recipes.
- [assets/](../assets): illustrations used in the README and results guide.
- [licenses/](../licenses): original license notices for adapted code.

For the core method, read [denoiser.py](../src/models/denoiser.py),
[correction.py](../src/models/correction.py), and [heun.py](../src/samplers/heun.py).
Native latent samplers live in [sit.py](../src/samplers/sit.py) and
[repa.py](../src/samplers/repa.py).
