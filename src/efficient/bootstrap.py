"""Paired bootstrap of FID differences between runs of src/efficient/evaluate.py.

Runs share image i's noise seed and class, so resampling image indices jointly across runs gives the standard error of
each run's FID and of its difference to the reference run. The trace term uses tr sqrtm(S1 S2) =
sum sqrt(eig(S1^1/2 S2 S1^1/2)), equal to evaluate.fid (checked to 4 decimals on the 50K runs) and much faster.

  python src/efficient/bootstrap.py runs/paper-k5 runs/relax-k2-s3 runs/extrap-k1-w6 --reps 30
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.efficient.evaluate import REFERENCE, fid


def symmetric_fid(reference=REFERENCE):
    with np.load(reference) as real:
        mu_r, s_r = real['mu'].astype(np.float64), real['sigma'].astype(np.float64)
    e, q = np.linalg.eigh(s_r)
    half = (q * np.sqrt(np.clip(e, 0, None))) @ q.T

    def score(features):
        f = features.astype(np.float64)
        mu, c = f.mean(0), np.cov(f, rowvar=False)
        lam = np.clip(np.linalg.eigvalsh(half @ c @ half), 0, None)
        return float(((mu - mu_r) ** 2).sum() + np.trace(s_r) + np.trace(c) - 2 * np.sqrt(lam).sum())
    return score


def load(run):
    """Features in image-index order (shard r holds images r, r + n, ...)."""
    n = json.loads((run / 'result.json').read_text())['shards']
    parts = [np.load(run / f'features-shard{r:03d}.npy') for r in range(n)]
    features = np.empty((sum(len(p) for p in parts), parts[0].shape[1]), np.float32)
    for r, part in enumerate(parts):
        features[r::n] = part
    return features


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', type=Path, nargs='+', help='the first run is the reference')
    p.add_argument('--reps', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    features = {run: load(run) for run in args.runs}
    sizes = {len(f) for f in features.values()}
    if len(sizes) != 1:
        raise ValueError('runs must have the same number of images')
    n = sizes.pop()
    score = symmetric_fid()
    rng = np.random.default_rng(args.seed)
    boots = {run: [] for run in args.runs}
    for _ in range(args.reps):
        idx = rng.integers(0, n, n)
        for run in args.runs:
            boots[run].append(score(features[run][idx]))
    reference = np.array(boots[args.runs[0]])
    point = {run: fid(features[run]) for run in args.runs}
    for run in args.runs:
        b = np.array(boots[run])
        # point difference of the full-sample FIDs, with the paired bootstrap standard error
        print(f'{run.name:24s} FID {point[run]:.3f}  s.e. {b.std():.3f}  '
              f'diff vs {args.runs[0].name} {point[run] - point[args.runs[0]]:+.3f} +- {np.std(b - reference):.3f}')


if __name__ == '__main__':
    main()
