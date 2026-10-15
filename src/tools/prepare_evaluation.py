"""Download public evaluation references and optionally prepare REPA real features."""
import argparse
import io
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ADM_COMMIT = '22e0df8183507e13a7813f8d38d51b072ca1e67c'
JIT_COMMIT = 'cbc743a2ada5e9762697da2c83f8c4f8379e8c17'
PUBLIC = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches'
ASSETS = {
    'VIRTUAL_imagenet256_labeled.npz': f'{PUBLIC}/imagenet/256/VIRTUAL_imagenet256_labeled.npz',
    'classify_image_graph_def.pb': f'{PUBLIC}/classify_image_graph_def.pb',
    'adm_evaluator.py': f'https://raw.githubusercontent.com/openai/guided-diffusion/{ADM_COMMIT}/evaluations/evaluator.py',
}
JIT_ASSETS = {
    'jit_in256_stats.npz': f'https://raw.githubusercontent.com/LTH14/JiT/{JIT_COMMIT}/fid_stats/jit_in256_stats.npz',
    'weights-inception-2015-12-05-6726825d.pth': 'https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth',
}


def download(url, path):
    if path.exists():
        return
    print(f'Downloading {path.name}', flush=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)


class ReferenceImages:
    """The paper's balanced ImageNet selection and single bicubic resize/crop."""
    def __init__(self, directory):
        import numpy as np
        from src.datasets.imagenet import ImageNet
        self.data = ImageNet(directory, horizontal_flip=False)
        self.fd = None if self.data.samples is not None else os.open(self.data.root / 'ILSVRC2012_img_train.tar', os.O_RDONLY)
        rng = np.random.RandomState(0)
        self.ids = []
        for label in np.unique(self.data.labels):
            rows = np.flatnonzero(self.data.labels == label)
            if self.fd is None:
                names = [Path(self.data.samples[i][0]).name for i in rows]
            else:
                names = [os.pread(self.fd, 100, int(self.data.index[i, 0])-512).split(b'\0', 1)[0] for i in rows]
            rows = rows[np.argsort(names)]
            selected = np.sort(rng.choice(len(rows), 50, replace=False))
            self.ids.extend(rows[selected])
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, j):
        from PIL import Image
        from torchvision.transforms import functional as TF
        i = self.ids[j]
        if self.data.samples is not None:
            source = self.data.samples[i][0]
        else:
            if self.fd is None:
                self.fd = os.open(self.data.root / 'ILSVRC2012_img_train.tar', os.O_RDONLY)
            offset, size, _ = map(int, self.data.index[i])
            source = io.BytesIO(os.pread(self.fd, size, offset))
        with Image.open(source) as image:
            image = image.convert('RGB')
            w, h = image.size
            if min(w, h) != 256:
                scale = 256 / min(w, h)
                image = TF.resize(image, [round(h*scale), round(w*scale)],
                                  interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            return TF.to_tensor(TF.center_crop(image, [256, 256]))


def real_features(data, output, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from pytorch_fid.inception import InceptionV3
    images = ReferenceImages(data)
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    loader = DataLoader(images, batch_size=64, num_workers=4, pin_memory=device.startswith('cuda'))
    features = []
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            features.append(net(batch.to(device))[0].flatten(1).cpu().numpy())
            print(f'Real-image features: batch {i+1}/{len(loader)}', flush=True)
    features = np.concatenate(features)
    # Preserve the paper's eight-rank reference ordering for the fixed PRDC subsets.
    order = np.concatenate([np.arange(rank, len(features), 8) for rank in range(8)])
    np.savez(output, feats=features[order].astype(np.float16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['jit', 'sit', 'sit-repa'], required=True)
    parser.add_argument('--output', type=Path, default=Path('evaluation-assets'))
    parser.add_argument('--data', type=Path, help='ImageNet training directory, needed for REPA precision/recall')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = ASSETS | (JIT_ASSETS if args.model == 'jit' else {})
    for name, url in assets.items():
        download(url, args.output / name)
    if args.model == 'sit-repa' and args.data and not (args.output / 'inception_train.npz').exists():
        real_features(args.data, args.output / 'inception_train.npz', args.device)
    print(f'References ready in {args.output}')


if __name__ == '__main__':
    main()
