"""Compute FID, precision and recall with each model's paper evaluator."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_paths(samples):
    paths = sorted((samples / 'images').glob('*.png'))
    if not paths:
        raise FileNotFoundError(f'No PNG images in {samples / "images"}')
    return paths


def precision_recall(real, fake, k=3):
    """PRDC precision/recall with the paper's inclusive k-NN balls."""
    real, fake = real.astype(np.float64), fake.astype(np.float64)
    def distances(x, y):
        squared = (x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*(x @ y.T)
        return np.sqrt(np.maximum(squared, 0))
    def radius(x):
        matrix = distances(x, x)
        np.fill_diagonal(matrix, np.inf)
        return np.partition(matrix, k-1, axis=1)[:, k-1]
    real_radius, fake_radius = radius(real), radius(fake)
    cross = distances(real, fake)
    return dict(precision=float((cross <= real_radius[:, None]).any(0).mean()),
                recall=float((cross <= fake_radius[None, :]).any(1).mean()))


def jit_fid(samples, reference, device='cuda'):
    import torch_fidelity
    image_paths(samples)
    # JiT's fork reads precomputed statistics through its FID+IS path.
    result = torch_fidelity.calculate_metrics(
        input1=str(samples / 'images'), input2=None,
        fid_statistics_file=str(reference / 'jit_in256_stats.npz'),
        feature_extractor_weights_path=str(reference / 'weights-inception-2015-12-05-6726825d.pth'),
        cuda=device != 'cpu', fid=True, isc=True, kid=False, prc=False, verbose=True)
    return dict(fid=float(result['frechet_inception_distance']))


def adm_metrics(samples, reference, metric):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    spec = importlib.util.spec_from_file_location('adm', reference / 'adm_evaluator.py')
    adm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adm)
    adm.INCEPTION_V3_PATH = str(reference / 'classify_image_graph_def.pb')
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4,
                           inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    paths = image_paths(samples)
    def batches():
        for i in range(0, len(paths), 64):
            images = []
            for path in paths[i:i+64]:
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert('RGB')))
            yield np.stack(images)
    with tf.Session(config=config) as session:
        evaluator = adm.Evaluator(session)
        fake, _ = evaluator.compute_activations(batches())
        result = {}
        reference_file = reference / 'VIRTUAL_imagenet256_labeled.npz'
        if metric in ('fid', 'all'):
            with np.load(reference_file) as stats:
                real_stats = adm.FIDStatistics(stats['mu'], stats['sigma'])
            result['fid'] = float(real_stats.frechet_distance(evaluator.compute_statistics(fake)))
        if metric in ('pr', 'all'):
            real, _ = evaluator.read_activations(str(reference_file))
            precision, recall = evaluator.compute_prec_recall(real, fake)
            result.update(precision=float(precision), recall=float(recall))
    return result


def repa_metrics(samples, reference, metric):
    fake = np.load(samples / 'inception-features.npy')
    result = {}
    if metric in ('fid', 'all'):
        from pytorch_fid.fid_score import calculate_frechet_distance
        fake64 = fake.astype(np.float64)
        with np.load(reference / 'VIRTUAL_imagenet256_labeled.npz') as real:
            result['fid'] = float(calculate_frechet_distance(
                real['mu'], real['sigma'], fake64.mean(0), np.cov(fake64, rowvar=False)))
    if metric in ('pr', 'all'):
        bank = reference / 'inception_train.npz'
        with np.load(bank) as data:
            real = data['feats'].astype(np.float32)
        count = min(10000, len(real), len(fake))
        ri = np.random.RandomState(0).choice(len(real), count, replace=False)
        fi = np.random.RandomState(1).choice(len(fake), count, replace=False)
        result.update(precision_recall(real[ri], fake[fi]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['jit', 'sit', 'sit-repa'], required=True)
    parser.add_argument('--samples', type=Path, required=True, help='Generation output directory')
    parser.add_argument('--reference', type=Path, default=Path('evaluation-assets'))
    parser.add_argument('--metric', choices=['fid', 'pr', 'all'], help='Defaults to fid for JiT, all otherwise')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='JiT FID device')
    args = parser.parse_args()
    metric = args.metric or ('fid' if args.model == 'jit' else 'all')
    if args.model == 'jit' and metric == 'all':
        parser.error('Run JiT --metric fid in the PyTorch environment and --metric pr in the ADM environment')
    if args.model == 'jit' and metric == 'fid':
        result = jit_fid(args.samples, args.reference, args.device)
    elif args.model == 'sit-repa':
        result = repa_metrics(args.samples, args.reference, metric)
    else:
        result = adm_metrics(args.samples, args.reference, metric)
    output = args.samples / f'metrics-{metric}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
