"""SiT/ADM preprocessing from ImageFolder, an indexed official tar, or VAE moments."""
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from collections import Counter
import torch
from torch.utils.data import Dataset

IMAGENET_CLASSES, IMAGENET_IMAGES = 1000, 1281167


def center_crop(image, size=256):
    # ADM/JiT's antialiased resize, then center crop.
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.Resampling.BICUBIC)
    arr = np.array(image)
    y, x = (arr.shape[0] - size) // 2, (arr.shape[1] - size) // 2
    return arr[y:y+size, x:x+size]


class ImageNet(Dataset):
    latent = False
    def __init__(self, directory, seed=0, image_size=256, horizontal_flip=True):
        self.image_size = image_size
        self.root = Path(directory)
        self.samples = None
        if (self.root/'index.npy').is_file():
            self.index = np.load(self.root / 'index.npy', mmap_mode='r')
            self.manifest = json.loads((self.root / 'dataset.json').read_text())
            if self.manifest['classes'] != IMAGENET_CLASSES or len(self.index) != IMAGENET_IMAGES:
                raise ValueError('Expected the complete ImageNet training set')
            self.labels = self.index[:, 2]
        else:
            if (self.root/'train').is_dir():
                self.root = self.root/'train'
            classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
            if len(classes) != IMAGENET_CLASSES or any(len(c) != 9 or c[0] != 'n' or not c[1:].isdigit() for c in classes):
                raise ValueError('Expected train/n########/*.JPEG or an indexed official tar; see docs/data.md')
            self.samples = []
            for label, name in enumerate(classes):
                for directory, dirs, files in os.walk(self.root/name):
                    dirs.sort()
                    self.samples.extend((str(Path(directory)/f), label) for f in sorted(files)
                                        if Path(f).suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            if len(self.samples) != IMAGENET_IMAGES:
                raise ValueError('Expected 1,281,167 ImageNet training images')
            self.labels = np.asarray([label for _, label in self.samples], dtype=np.int64)
            counts = Counter(map(int, self.labels))
            if len(counts) != IMAGENET_CLASSES:
                raise ValueError('Every ImageNet class directory must contain images')
            self.manifest = dict(classes=IMAGENET_CLASSES, images=len(self.samples), layout='ImageFolder',
                                 class_to_idx={c: i for i, c in enumerate(classes)},
                                 counts={c: counts[i] for i, c in enumerate(classes)})
        self.horizontal_flip = horizontal_flip
        self.seed, self.epoch = seed, 0
        self.fd = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.samples is None:
            if self.fd is None:
                self.fd = os.open(self.root / 'ILSVRC2012_img_train.tar', os.O_RDONLY)
            offset, size, label = map(int, self.index[i])
            source = io.BytesIO(os.pread(self.fd, size, offset))
        else:
            source, label = self.samples[i]
        with Image.open(source) as image:
            arr = center_crop(image.convert('RGB'), self.image_size)
        # Match flips across worker counts and training restarts.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
        if self.horizontal_flip and rng.random() < .5:
            arr = arr[:, ::-1]
        x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255)
        return x.sub_(.5).div_(.5), label


class LatentMoments(Dataset):
    """Packed FP32 VAE posteriors for both pixel-space flips; sample latents on GPU."""
    latent = True

    def __init__(self, directory, seed=0, horizontal_flip=True):
        self.root = Path(directory)
        self.manifest = json.loads((self.root/'latents.json').read_text())
        if self.manifest['format'] != 'imagenet-vae-moments/1' or not self.manifest['complete']:
            raise ValueError('Expected a complete VAE posterior cache')
        self.labels = np.load(self.root/'labels.npy', mmap_mode='r')
        self.shards = self.manifest['shards']
        self.ends = np.array([s['end'] for s in self.shards])
        starts = [s['start'] for s in self.shards]
        if (not self.shards or starts != [0, *self.ends[:-1]] or
                np.any(self.ends <= starts) or self.ends[-1] != len(self.labels) or
                len(self.labels) != self.manifest['images']):
            raise ValueError('Latent cache index does not cover all images')
        self.arrays = {}
        self.horizontal_flip = horizontal_flip
        self.seed, self.epoch = seed, 0

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        shard = int(np.searchsorted(self.ends, i, side='right'))
        if shard not in self.arrays:
            self.arrays[shard] = np.load(self.root/self.shards[shard]['file'], mmap_mode='r')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
        flip = int(self.horizontal_flip and rng.random() < .5)
        params = self.arrays[shard][i-self.shards[shard]['start'], flip].copy()
        return torch.from_numpy(params), int(self.labels[i])
