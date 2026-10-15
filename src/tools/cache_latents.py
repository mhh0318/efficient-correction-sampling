"""Cache VAE posterior parameters for original images and horizontal flips."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.imagenet import ImageNet
from src.utils.config import read_config
from src.datasets.codec import build_codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', device_id=torch.device(f'cuda:{local}'))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    recipe = read_config(args.config)
    if recipe['codec']['name'] != 'vae':
        parser.error('Latent caching requires a VAE recipe')
    data = ImageNet(args.data, image_size=recipe['model']['image_size'], horizontal_flip=False)
    args.output.mkdir(parents=True, exist_ok=True)
    n = len(data)
    start, end = n * rank // world, n * (rank + 1) // world
    name = f'moments-{rank:03d}.npy'
    record_path = args.output / f'shard-{rank:03d}.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if (record['start'], record['end'], record['codec']) != (start, end, recipe['codec']):
            raise ValueError('Cache settings changed; use a new output directory')
    else:
        codec = build_codec(recipe).eval()
        size = recipe['model']['image_size'] // codec.config.downsample
        temporary = args.output / (name + '.partial')
        array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                         shape=(end-start, 2, 2*codec.config.channels, size, size))
        loader = DataLoader(Subset(data, range(start, end)), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True)
        cursor = 0
        with torch.inference_mode():
            for images, _ in loader:
                images = images.cuda(non_blocking=True)
                views = torch.stack([images, images.flip(-1)], dim=1).flatten(0, 1)
                moments = codec.vae.encode(views).latent_dist.parameters.float()
                count = len(images)
                array[cursor:cursor+count] = moments.cpu().numpy().reshape(count, 2, 2*codec.config.channels, size, size)
                cursor += count
                print(f'Rank {rank}: {cursor}/{end-start} images', flush=True)
        array.flush()
        del array
        temporary.replace(args.output / name)
        record = dict(file=name, start=start, end=end, codec=recipe['codec'])
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    dist.barrier()
    if rank == 0:
        shards = [json.loads((args.output / f'shard-{r:03d}.json').read_text()) for r in range(world)]
        np.save(args.output / 'labels.npy', np.asarray(data.labels, dtype=np.int64))
        manifest = dict(format='imagenet-vae-moments/1', complete=True, images=n,
                        image_size=recipe['model']['image_size'], codec=recipe['codec'], shards=shards)
        (args.output / 'latents.json').write_text(json.dumps(manifest, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
