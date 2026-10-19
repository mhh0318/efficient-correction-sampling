"""FID of a corrector method on class-conditional ImageNet-256 under the paper's SiT-REPA protocol, on one GPU or sharded.

The protocol of src/evaluate.py for SiT-REPA: repa_sde, 123 steps, no CFG, float pixels into pytorch-fid's Inception,
FID against VIRTUAL_imagenet256_labeled.npz; image i uses CPU noise seed i and class i % 1000. Shard r of n generates
images r, r + n, ...; every shard seeds the path noise with 220922 (the runs in the README). The last shard to finish
merges all shards into result.json.

  python src/efficient/evaluate.py --method relax --k 2 --scale 3 --output runs/relax-k2-s3 --shard 0 --nshards 4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image
from scipy import linalg

from src.datasets.codec import build_codec
from src.efficient.guidance import CorrectorGuidance, Method
from src.samplers.repa import sample as repa_sample
from src.utils.checkpoint import load_model
from src.utils.download import resolve_checkpoint

REFERENCE = Path(__file__).resolve().parents[2] / 'evaluation-assets/VIRTUAL_imagenet256_labeled.npz'


def frechet(mu1, s1, mu2, s2):
    """pytorch-fid 0.3.0 calculate_frechet_distance, without sqrtm's `disp` argument that SciPy >= 1.16 removed."""
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def fid(features, reference=REFERENCE):
    features = features.astype(np.float64)
    with np.load(reference) as real:
        return frechet(real['mu'], real['sigma'], features.mean(0), np.cov(features, rowvar=False))


def atomic(path, write):
    temporary = path.with_name(f'.tmp-{path.name}')
    write(temporary)
    os.replace(temporary, path)


def merge(output, record, nshards):
    """Shard metadata is written after its features, so a complete set of .json files means complete features."""
    done = sorted(output.glob('features-shard*.json'))
    if len(done) < nshards:
        return
    metas = [json.loads(p.read_text()) for p in done]
    features = np.concatenate([np.load(p.with_suffix('.npy')) for p in done])
    n = sum(m['samples'] for m in metas)
    record |= {'samples': n, 'fid': fid(features), 'shards': nshards,
                   'calls_per_image': {k: sum(m['calls'][k] for m in metas) / n for k in metas[0]['calls']}}
    (output / 'result.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', choices=['base', 'paper', 'extrap', 'relax'], required=True)
    p.add_argument('--k', type=int, default=1, help='corrections per active step')
    p.add_argument('--scale', type=float, default=1., help='w for extrap, s for relax')
    p.add_argument('--checkpoint', default='sit-repa-50k', help='official name or local bundle')
    p.add_argument('--samples', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--steps', type=int, default=123)
    p.add_argument('--correction-interval', type=float, nargs=2, default=(.5, .7), metavar=('LOW', 'HIGH'),
                   help='noise fraction where the corrector is active (1 = noise)')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    method = Method(args.method, args.k, args.scale)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    model, config = load_model(resolve_checkpoint(args.checkpoint), 'cuda')
    model.requires_grad_(False)
    codec = build_codec(config, 'cuda').eval()
    from pytorch_fid.inception import InceptionV3
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
    calls = {'base': 0, 'correction': 0}
    model.net.register_forward_hook(lambda m, i, o: calls.__setitem__('base', calls['base'] + len(i[0])))
    model.corr_net.register_forward_hook(lambda m, i, o: calls.__setitem__('correction', calls['correction'] + len(i[0])))
    guided = CorrectorGuidance(model, method)
    shape = (model.net.in_channels, model.image_size, model.image_size)
    ids = list(range(args.shard, args.samples, args.nshards))
    features, grid, start = [], [], time.time()
    torch.manual_seed(220922)
    for b in range(0, len(ids), args.batch_size):
        batch = ids[b:b + args.batch_size]
        noise = torch.stack([torch.randn(*shape, generator=torch.Generator().manual_seed(i)) for i in batch]).cuda()
        labels = torch.tensor([i % model.num_classes for i in batch], device='cuda')
        with torch.no_grad():
            z = repa_sample(guided, noise, labels, steps=args.steps, corrections=int(method.name != 'base'),
                            correction_interval=tuple(args.correction_interval), cfg=1., cfg_interval=(0., 1.))
            pixels = ((torch.cat([codec.decode(x.float()) for x in z.split(16)]) + 1) * .5).clamp(0, 1)
            features.append(torch.cat([inception(x)[0].flatten(1) for x in pixels.split(32)]).cpu().numpy())
        if args.shard == 0 and len(grid) < 16:
            grid += list((pixels[:16 - len(grid)] * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        print(f'shard {args.shard}: {b + len(batch)}/{len(ids)} {time.time() - start:.0f}s', flush=True)
    if args.shard == 0 and len(grid) == 16:
        Image.fromarray(np.concatenate([np.concatenate(grid[r*4:(r+1)*4], 1) for r in range(4)])).save(args.output / 'grid.png')
    part = args.output / f'features-shard{args.shard:03d}.npy'
    atomic(part, lambda tmp: np.save(tmp, np.concatenate(features).astype(np.float32)))
    meta = json.dumps({'samples': len(ids), 'calls': calls, 'seconds': time.time() - start})
    atomic(part.with_suffix('.json'), lambda tmp: tmp.write_text(meta))
    merge(args.output, {'method': method.__dict__, 'checkpoint': args.checkpoint, 'steps': args.steps,
                            'correction_interval': list(args.correction_interval), 'batch_size': args.batch_size}, args.nshards)


if __name__ == '__main__':
    main()
