#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# w=1 uses 125 NFEs; guided rows use 250 NFEs.
# Generate all nine images in one batch so every setting starts from the same noises.
for w in 1 3 5 7; do
  python src/sample.py --config configs/eval/qualitative/sit-repa.yml \
    --classes 250 --samples 9 --batch-size 9 --grid-columns 3 --seed 42 \
    --denoising_steps 125 --corrections_per_step 0 --cfg "$w" --output "samples/cfg/w${w}"
done
python src/tools/make_grid.py samples/cfg/w1 samples/cfg/w3 samples/cfg/w5 samples/cfg/w7 \
  --horizontal --labels "CFG w=1 (baseline)" "CFG w=3" "CFG w=5" "CFG w=7" --output samples/cfg-scales.png
