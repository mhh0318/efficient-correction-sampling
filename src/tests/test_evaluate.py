"""Run the JiT evaluation CLI for plain models without allocating a GPU."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from src import evaluate
from src.models.jit import JiT, JiT_models
from src.utils.checkpoint import load_model, save_checkpoint
from src.utils.config import build_model, resolve


class EvaluateTests(unittest.TestCase):
    def test_paper_cohorts_keep_their_class_and_rank_allocation(self):
        import numpy as np
        for family, batch_size in [('jit', 125), ('sit', 32), ('repa', 64)]:
            groups = [evaluate.batches(family, rank, 8, 50000, batch_size) for rank in range(8)]
            ids = np.asarray([i for rank in groups for batch in rank for i in batch if i < 50000])
            np.testing.assert_array_equal(np.sort(ids), np.arange(50000))
            if family == 'jit':
                np.testing.assert_array_equal(groups[0][0], np.arange(125))
                np.testing.assert_array_equal(np.bincount(ids // 50), np.full(1000, 50))
            else:
                np.testing.assert_array_equal(groups[0][0], np.arange(batch_size)*8)
            if family == 'sit':
                self.assertEqual(sum(len(batch) for rank in groups for batch in rank), 50176)
            if family == 'repa':
                np.testing.assert_array_equal(ids, np.concatenate([np.arange(r, 50000, 8) for r in range(8)]))

    def test_compile_flag_preserves_base_and_student_samples(self):
        name = 'JiT-evaluation-test'
        factory = lambda **kw: JiT(hidden_size=32, depth=2, num_heads=4, patch_size=4,
                                  bottleneck_dim=8, in_context_len=2, in_context_start=1, **kw)
        tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            if str(kwargs.get('device', '')).startswith('cuda'):
                kwargs['device'] = 'cpu'
            return tensor(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.dict(JiT_models, {name: factory}):
            root = Path(directory)
            for mode in ('base', 'distill'):
                with self.subTest(mode=mode):
                    recipe = resolve(dict(
                        model=dict(model=name, image_size=16, num_classes=5, label_drop=0., time_sampling='uniform'),
                        training=dict(mode=mode, correction_loss_weight=0.),
                        sampling=dict(steps=3, corrections=0, precision='fp32')))
                    original = build_model(recipe)
                    checkpoint = root / mode
                    save_checkpoint(checkpoint, original.state_dict(), recipe, source={}, kind='base_init')
                    images = []
                    for compiled in (False, True):
                        output = root / f'{mode}-{compiled}'
                        argv = ['evaluate.py', '--checkpoint', str(checkpoint), '--output',
                                str(output), '--samples', '1', '--batch-size', '1']
                        if compiled:
                            argv.append('--compile')
                        with ExitStack() as stack:
                            stack.enter_context(patch.object(sys, 'argv', argv))
                            stack.enter_context(patch.dict('os.environ', RANK='0', LOCAL_RANK='0', WORLD_SIZE='1'))
                            stack.enter_context(patch.object(evaluate, 'load_model',
                                side_effect=lambda path, device: load_model(path, 'cpu')))
                            for method in ('init_process_group', 'barrier', 'destroy_process_group', 'broadcast_object_list'):
                                stack.enter_context(patch.object(evaluate.dist, method))
                            stack.enter_context(patch.object(torch.cuda, 'set_device'))
                            stack.enter_context(patch.object(torch, 'tensor', side_effect=cpu_tensor))
                            stack.enter_context(patch.object(torch, 'set_num_threads'))
                            stack.enter_context(patch.object(torch, 'use_deterministic_algorithms'))
                            for obj, attr in [(torch.backends.cuda.matmul, 'allow_tf32'),
                                              (torch.backends.cudnn, 'allow_tf32'),
                                              (torch.backends.cudnn, 'benchmark'),
                                              (torch.backends.cudnn, 'deterministic')]:
                                stack.callback(setattr, obj, attr, getattr(obj, attr))
                            compile_block = stack.enter_context(patch.object(torch, 'compile', side_effect=lambda f: f))
                            evaluate.main()
                            self.assertEqual(compile_block.call_count, 3 if compiled else 0)
                        self.assertEqual(json.loads((output / 'generation.json').read_text())['samples'], 1)
                        images.append((output / 'images/000000.png').read_bytes())
                    self.assertEqual(*images)


if __name__ == '__main__':
    unittest.main()
