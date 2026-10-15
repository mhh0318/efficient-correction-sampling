# Distill correction into a denoiser

A student denoiser learns the frozen teacher's corrected predictions using
velocity MSE. It starts from the teacher's denoiser and samples with K=0.

## SiT

The [SiT recipe](../configs/distill/sit.yml) uses teacher K=2, 100K training
updates, and an 83-step student sampler.

```bash
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/distill/sit.yml --teacher sit --global-batch-size 256 \
  --data /path/to/imagenet-moments --output runs/distill-sit
```

Teacher weights download automatically. Use `--teacher /path/to/bundle` for
local weights. ImageFolder or [packed VAE cache](data.md#optional-packed-vae-cache)
inputs work; the teacher and student must use the same codec.

To resume, repeat the command with `--resume`. To change the target, use
`--teacher-corrections K`.

The distillation recipes save a restart checkpoint every 5K updates and an
EMA bundle every 10K updates, and save both at the final update.

## Sample the student

```bash
python src/sample.py --checkpoint runs/distill-sit/step-0100000 \
  --classes 207 388 949 980 --output samples/distill-sit
```

The exported bundle sets K=0. Add `--denoising_steps 250` to use more denoising steps.

<details>
<summary>Additional SiT-REPA example recipe</summary>

The student uses 250 denoising steps.

```bash
torchrun --standalone --nproc_per_node=8 src/train.py \
  --config configs/distill/sit-repa.yml --teacher sit-repa-100k --global-batch-size 256 \
  --data /path/to/imagenet-moments --output runs/distill-repa

python src/sample.py --checkpoint runs/distill-repa/step-0005000 \
  --classes 207 388 949 980 --output samples/distill-repa
```

</details>
