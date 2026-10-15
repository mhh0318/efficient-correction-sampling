# Implementation

The backbones expose `net(state, t, labels) -> prediction`. The shared wrapper
converts predictions between clean-image and velocity representations, computes
training losses, and runs the selected sampler.

| Module | Role |
| :--- | :--- |
| [denoiser.py](../src/models/denoiser.py) | Shared model wrapper and denoising/correction losses |
| [correction.py](../src/models/correction.py) | Corrector input/output spaces and combination rules |
| [jit.py](../src/models/jit.py), [sit.py](../src/models/sit.py) | Backbone definitions |
| [heun.py](../src/samplers/heun.py) | JiT sampling |
| [sit.py](../src/samplers/sit.py), [repa.py](../src/samplers/repa.py) | Native latent-model sampling |
| [codec.py](../src/datasets/codec.py) | Pixel inputs and frozen VAE encoding/decoding |

## Training and sampling

Both networks learn from the same clean target. Correction-loss gradients stop
at the denoiser's prediction. During sampling, the corrector refines that
prediction repeatedly before the sampler constructs the next noisy state.
[Checkpoint settings](checkpoints.md) describe the corrector used by each model.

The command-line entry points are [train.py](../src/train.py),
[sample.py](../src/sample.py), and the evaluation runners described in
[evaluation](evaluation.md).

## Tests

After installing `requirements-latent.txt`, run the CPU tests without ImageNet
or pretrained weights:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s src/tests -p 'test_*.py'
```
