"""Paper metric semantics and public reference preparation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from src import metrics
from src.tools import prepare_evaluation


class MetricTests(unittest.TestCase):
    def test_prdc_includes_boundary_points_and_excludes_self_neighbors(self):
        real = np.arange(8, dtype=np.float32).reshape(-1, 1)
        self.assertEqual(metrics.precision_recall(real, real), dict(precision=1., recall=1.))
        self.assertEqual(metrics.precision_recall(real, real+100), dict(precision=0., recall=0.))

    def test_repa_uses_saved_float_features_and_fixed_subset_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples, reference = root / 'samples', root / 'reference'
            samples.mkdir(); reference.mkdir()
            rng = np.random.RandomState(7)
            fake = rng.randn(10020, 4).astype(np.float32)
            real = rng.randn(10030, 4).astype(np.float16)
            np.save(samples / 'inception-features.npy', fake)
            np.savez(reference / 'inception_train.npz', feats=real)
            with patch.object(metrics, 'precision_recall', return_value={'precision': .5, 'recall': .7}) as pr:
                result = metrics.repa_metrics(samples, reference, 'pr')
            self.assertEqual(result, dict(precision=.5, recall=.7))
            np.testing.assert_array_equal(pr.call_args.args[0],
                real[np.random.RandomState(0).choice(len(real), 10000, replace=False)].astype(np.float32))
            np.testing.assert_array_equal(pr.call_args.args[1],
                fake[np.random.RandomState(1).choice(len(fake), 10000, replace=False)])

    def test_reference_preparation_downloads_only_public_assets_and_reuses_files(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            def retrieve(url, path):
                self.assertTrue(url.startswith(('https://raw.githubusercontent.com/',
                    'https://openaipublic.blob.core.windows.net/', 'https://github.com/toshas/')))
                Path(path).write_bytes(b'test reference')
            argv = ['prepare_evaluation.py', '--model', 'jit', '--output', str(target)]
            with patch.object(sys, 'argv', argv), patch.object(prepare_evaluation.urllib.request,
                    'urlretrieve', side_effect=retrieve) as request:
                prepare_evaluation.main()
                self.assertEqual(request.call_count, 5)
                prepare_evaluation.main()
                self.assertEqual(request.call_count, 5)
            self.assertFalse(list(target.glob('*.partial')))

    def test_metric_cli_selects_backend_and_saves_readable_results(self):
        with tempfile.TemporaryDirectory() as directory:
            for model, option, backend in [('jit', 'fid', 'jit_fid'), ('jit', 'pr', 'adm_metrics'),
                                           ('sit', 'all', 'adm_metrics'), ('sit-repa', 'all', 'repa_metrics')]:
                with self.subTest(model=model, metric=option), \
                        patch.object(sys, 'argv', ['metrics.py', '--model', model, '--samples', directory,
                                                  '--metric', option]), \
                        patch.object(metrics, backend, return_value={'fid': 1.2}):
                    metrics.main()
                self.assertEqual(json.loads((Path(directory) / f'metrics-{option}.json').read_text()), {'fid': 1.2})


if __name__ == '__main__':
    unittest.main()
