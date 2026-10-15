#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# Use the same four initial noises for every row (Siberian husky, ~250 NFEs).
options=(
  --config configs/eval/qualitative/sit-repa.yml
  --classes 250 --samples 4 --batch-size 4 --seed 42
)

# Baseline: 250 NFEs.
python src/sample.py "${options[@]}" \
  --denoising_steps 250 --corrections_per_step 0 --cfg 1 --output samples/compare/base

# CFG: 250 NFEs, including both branches.
python src/sample.py "${options[@]}" \
  --denoising_steps 125 --corrections_per_step 0 --cfg 3 --output samples/compare/cfg

# Correction: 252 NFEs.
python src/sample.py "${options[@]}" \
  --denoising_steps 42 --corrections_per_step 5 --cfg 1 --output samples/compare/correction

python src/tools/make_grid.py samples/compare/base samples/compare/cfg samples/compare/correction \
  --labels "Baseline" "CFG w=3" "Correction K=5" --output samples/compare.png
