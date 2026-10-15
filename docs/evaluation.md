# Evaluation

The commands below follow each model's paper evaluation protocol. Run them from
the repository root. All quantitative presets generate 50K samples without CFG.

| Model | FID | Precision / recall |
| :--- | :--- | :--- |
| JiT | JiT's torch-fidelity fork | ADM |
| SiT | ADM | ADM |
| SiT-REPA | pytorch-fid, using float decoded images | PRDC, fixed 10K feature subsets |

## 1. Install evaluators

In your existing PyTorch environment:

```bash
pip install -r requirements-eval.txt
```

ADM uses a separate environment so its TensorFlow CUDA dependencies do not
conflict with PyTorch:

```bash
python -m venv .venv-adm
.venv-adm/bin/python -m pip install -r requirements-adm.txt
```

## 2. Prepare references

The helper downloads the public reference files and the ADM evaluator. Run the
command for the model you want to evaluate; already downloaded files are reused.

```bash
python src/tools/prepare_evaluation.py --model jit
python src/tools/prepare_evaluation.py --model sit
python src/tools/prepare_evaluation.py --model sit-repa
```

For **SiT-REPA precision/recall**, also prepare real-image features from the
ImageNet training directory described in [data preparation](data.md):

```bash
python src/tools/prepare_evaluation.py --model sit-repa --data /path/to/imagenet
```

This selects 50 images per class, applies the paper's bicubic resize and center
crop, and saves `evaluation-assets/inception_train.npz`. It runs on one GPU and
only needs to be done once. FID uses the public
reference statistics and does not require ImageNet training images.

## 3. Generate samples

Checkpoints download automatically. Use eight GPUs for the paper presets:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# JiT-L/16
torchrun --standalone --nproc_per_node=8 src/evaluate.py --model jit --output runs/jit-50k
# SiT-XL/2
torchrun --standalone --nproc_per_node=8 src/evaluate.py --model sit --output runs/sit-50k
# SiT-REPA-XL/2, 50K-update checkpoint
torchrun --standalone --nproc_per_node=8 src/evaluate.py --model sit-repa --output runs/repa-50k
```

Images are saved in `runs/MODEL-50k/images/`. SiT-REPA also saves
`inception-features.npy` **before PNG quantization**; use these features for its
paper metrics. Each generation command starts from the beginning. Use a separate
output directory for each run. Metrics use the images or features you provide;
they do not require a particular sample count or a generation record.

`--config` selects a custom sampling recipe, and `--checkpoint` loads a local
bundle. `--samples` and `--batch-size` override the generation size.

## 4. Compute metrics

**JiT** uses the PyTorch environment for FID and the ADM environment for
precision/recall:

```bash
python src/metrics.py --model jit --samples runs/jit-50k --metric fid
.venv-adm/bin/python src/metrics.py --model jit --samples runs/jit-50k --metric pr
```

**SiT** computes all three metrics with ADM:

```bash
.venv-adm/bin/python src/metrics.py --model sit --samples runs/sit-50k
```

**SiT-REPA** computes all three from its saved float-image features:

```bash
python src/metrics.py --model sit-repa --samples runs/repa-50k
```

Use `--metric fid` for SiT-REPA if you have not prepared the real-feature bank.
Results are printed and saved as `metrics-fid.json`, `metrics-pr.json`, or
`metrics-all.json` in the sample directory. `--reference DIR` selects another
reference directory; use the same directory with the preparation helper's
`--output DIR`.
