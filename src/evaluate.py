"""Generate evaluation samples using the model's quantitative sampling recipe."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from src.datasets.codec import build_codec
from src.utils.checkpoint import load_model
from src.utils.config import read_sampling_config, sampling_recipe
from src.utils.download import resolve_checkpoint


def batches(family, rank, world, samples, batch_size):
    if family == 'repa':
        ids = list(range(rank, samples, world))
        return [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
    count = math.ceil(samples / (world * batch_size))
    if family == 'jit':
        return [[i for i in range((b*world+rank)*batch_size, (b*world+rank+1)*batch_size)
                 if i < samples] for b in range(count)]
    return [[b*world*batch_size + i*world + rank for i in range(batch_size)] for b in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['jit', 'sit', 'sit-repa'])
    parser.add_argument('--config', type=Path, help='Sampling config; defaults to the model quantitative preset')
    parser.add_argument('--checkpoint', help='Local checkpoint bundle; defaults to the preset checkpoint')
    parser.add_argument('--checkpoint-dir', type=Path,
                        help='Download/cache root (default: CORRECTION_CHECKPOINT_DIR or checkpoints)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    if args.model and args.config is None:
        args.config = Path(__file__).resolve().parents[1] / f'configs/eval/quantitative/{args.model}.yml'
    preset = read_sampling_config(args.config) if args.config else None
    generation = preset['generation'] if preset else {}
    if not args.checkpoint and not preset:
        parser.error('Specify --model, --config, or --checkpoint')
    if preset and generation['runner'] == 'sample':
        parser.error('Use a quantitative preset for evaluation')
    args.samples = args.samples if args.samples is not None else generation.get('samples', 50000)
    args.batch_size = args.batch_size if args.batch_size is not None else generation.get('batch_size', 125)
    if args.samples < 1 or args.batch_size < 1:
        parser.error('samples and batch-size must be positive')
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.cuda.set_device(local)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    dist.init_process_group('gloo')
    location = [None, None]
    if rank == 0:
        try:
            location[0] = str(resolve_checkpoint(args.checkpoint or preset['checkpoint'], args.checkpoint_dir).resolve())
        except (OSError, ValueError, RuntimeError) as error:
            location[1] = str(error)
    dist.broadcast_object_list(location, src=0)
    if location[1]:
        raise RuntimeError(location[1])
    model, config = load_model(location[0], f'cuda:{local}')
    family = generation.get('runner', 'jit' if not config['model']['model'].startswith('SiT-') else
                            'sit' if config['model']['data_time'] == 1 else 'repa')
    recipe = sampling_recipe(config, preset)
    precision = recipe.pop('precision')
    torch.backends.cuda.matmul.allow_tf32 = family != 'jit'
    torch.backends.cudnn.allow_tf32 = family != 'jit'
    if family == 'jit':
        torch.set_num_threads(6)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    codec = build_codec(config, next(model.parameters()).device).eval()
    inception = None
    if family == 'repa':
        from pytorch_fid.inception import InceptionV3
        inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(next(model.parameters()).device).eval()
    if args.compile:
        for net in (model.net, model.corr_net):
            if net is not None:
                for block in [*net.blocks, net.final_layer]:
                    block.forward = torch.compile(block.forward)
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output / 'images'
    images.mkdir(exist_ok=True)
    settings = dict(family=family, checkpoint=location[0], sampling=recipe, precision=precision,
                    samples=args.samples, batch_size=args.batch_size, world_size=world, compile=args.compile)
    if rank == 0:
        (args.output / 'generation.json').write_text(json.dumps(settings, indent=2) + '\n')
    torch.manual_seed((220922 if family == 'repa' else 0) + rank)
    groups = batches(family, rank, world, args.samples, args.batch_size)
    device = next(model.parameters()).device
    shape = (model.net.in_channels, model.image_size, model.image_size)
    rank_features = []
    with torch.inference_mode():
        for b, ids in enumerate(groups):
            if not ids:
                continue
            if family == 'jit':
                labels = torch.tensor([i*model.num_classes//args.samples for i in ids], device=device)
                noise = None
            elif family == 'repa':
                noise = torch.stack([torch.randn(*shape, generator=torch.Generator().manual_seed(i)) for i in ids]).to(device)
                labels = torch.tensor([i % model.num_classes for i in ids], device=device)
            else:
                noise = torch.randn(len(ids), *shape, device=device)
                labels = torch.randint(0, model.num_classes, (len(ids),), device=device)
            if config.get('conditioning') == 'unconditional':
                labels = torch.full_like(labels, model.num_classes)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                states = model.sample(labels, **recipe) if noise is None else model.sample(labels, noise=noise, **recipe)
            decoded = torch.cat([codec.decode(z.float()) for z in states.split(16)])
            if family == 'repa':
                pixels01 = ((decoded+1)*.5).clamp(0, 1)
                features = torch.cat([inception(x)[0].flatten(1) for x in pixels01.split(32)]).cpu().numpy()
                rank_features.append(features)
                pixels = (pixels01*255).round().to(torch.uint8)
            elif family == 'sit':
                pixels = (127.5*decoded+128).clamp(0, 255).to(torch.uint8)
            else:
                pixels = ((decoded+1)*127.5).clamp(0, 255).round().to(torch.uint8)
            for i, pixel in zip(ids, pixels.permute(0, 2, 3, 1).cpu().numpy()):
                if i < args.samples:
                    Image.fromarray(pixel).save(images / f'{i:06d}.png')
            print(f'Rank {rank}: batch {b+1}/{len(groups)}', flush=True)
    if family == 'repa':
        features = np.concatenate(rank_features) if rank_features else np.empty((0, 2048), dtype=np.float32)
        np.save(args.output / f'features-rank-{rank}.npy', features)
    dist.barrier()
    if rank == 0 and family == 'repa':
        features = np.concatenate([np.load(args.output / f'features-rank-{r}.npy') for r in range(world)])
        np.save(args.output / 'inception-features.npy', features)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
