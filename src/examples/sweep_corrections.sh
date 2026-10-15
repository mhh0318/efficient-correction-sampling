#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# Approximately 250 NFEs per image. Use --denoising_steps 42 for fixed steps instead.
# Generate all nine images in one batch so every setting starts from the same noises.
for k in 0 1 2 4 6; do
  python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
    --classes 250 --samples 9 --batch-size 9 --grid-columns 3 --seed 42 \
    --denoising_steps "$((250 / (1 + k)))" --corrections_per_step "$k" --cfg 1 \
    --output "samples/correction/k${k}"
done
python src/tools/make_grid.py \
  samples/correction/k0 samples/correction/k1 samples/correction/k2 \
  samples/correction/k4 samples/correction/k6 \
  --horizontal --labels "K=0 (baseline)" "K=1" "K=2" "K=4" "K=6" \
  --output samples/correction-counts.png
