"""Checkpoint intent, integrity and preservation checks on tiny CPU models."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.utils.checkpoint import initialize_model, load_model, save_checkpoint
from src.utils.config import build_model, resolve
from src.models.jit import JiT, JiT_models
from src.models.sit import SiT, SiT_models


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        SiT_models['SiT-checkpoint-test'] = lambda **kw: SiT(
            hidden_size=32, depth=1, num_heads=4, patch_size=2, **kw)
        self.recipe = resolve(dict(model=dict(model='SiT-checkpoint-test', image_size=32, num_classes=5),
                                   correction=dict(input='x', output='v', combine='gated')))
        self.trained = build_model(self.recipe)
        with torch.no_grad():
            # Distinct nonzero networks and gate expose accidental initialization.
            for p in self.trained.net.parameters(): p.uniform_(-.05, .05)
            for p in self.trained.corr_net.parameters(): p.uniform_(.1, .2)
            self.trained.output_gate.copy_(torch.tensor([.1, .2, .3, .4]))

    def save(self, kind='pair'):
        path = self.root/kind
        save_checkpoint(path, self.trained.state_dict(), self.recipe, source={'test': True}, kind=kind)
        return path

    def assert_state_equal(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)

    def test_pair_continuation_and_sampling_load_preserve_all_tensors(self):
        path = self.save()
        model = build_model(self.recipe)
        initialize_model(model, self.recipe, path, mode='pair')
        self.assert_state_equal(model.state_dict(), self.trained.state_dict())
        loaded, config = load_model(path, 'cpu')
        self.assertEqual(config['kind'], 'pair')
        self.assert_state_equal(loaded.state_dict(), self.trained.state_dict())
        z, t, y = torch.randn(2, 4, 4, 4), torch.full((2, 1, 1, 1), .4), torch.tensor([0, 3])
        torch.testing.assert_close(loaded.velocity(z, t, y, corrections=2),
                                   model.eval().velocity(z, t, y, corrections=2), rtol=0, atol=0)

    def test_base_initialization_creates_new_corrector(self):
        model = build_model(self.recipe)
        initialize_model(model, self.recipe, self.save('base_init'), mode='base')
        self.assert_state_equal(model.net.state_dict(), self.trained.net.state_dict())
        self.assert_state_equal(model.corr_net.state_dict(), model.net.state_dict())
        torch.testing.assert_close(model.output_gate, torch.zeros(4), rtol=0, atol=0)

    def test_jit_sampling_load_recreates_nonpersistent_rope_buffers(self):
        name = 'JiT-checkpoint-test'
        with patch.dict(JiT_models, {name: lambda **kw: JiT(
                hidden_size=32, depth=2, num_heads=4, patch_size=4,
                bottleneck_dim=8, in_context_len=2, in_context_start=1, **kw)}):
            recipe = resolve(dict(model=dict(model=name, image_size=16, num_classes=5)))
            original = build_model(recipe).eval()
            with torch.no_grad():
                for parameter in original.parameters():
                    if parameter.requires_grad:
                        parameter.uniform_(-.05, .05)
            path = self.root / 'jit'
            save_checkpoint(path, original.state_dict(), recipe, source={'test': True})
            loaded, _ = load_model(path, 'cpu')
            self.assert_state_equal(loaded.state_dict(), original.state_dict())
            buffers = dict(loaded.named_buffers())
            self.assertEqual(len(buffers), 8)
            for key, buffer in original.named_buffers():
                self.assertFalse(buffers[key].is_meta, key)
                torch.testing.assert_close(buffers[key], buffer, rtol=0, atol=0)
            z, t, y = torch.randn(2, 3, 16, 16), torch.full((2, 1, 1, 1), .4), torch.tensor([0, 3])
            torch.testing.assert_close(loaded.velocity(z, t, y, corrections=2),
                                       original.velocity(z, t, y, corrections=2), rtol=0, atol=0)

    def test_repa_base_converter_preserves_base_and_creates_fresh_pair(self):
        from src.tools.convert_sit import main as convert
        recipe = copy.deepcopy(self.recipe)
        recipe['model']['data_time'] = 0
        config = self.root/'recipe.json'
        config.write_text(json.dumps(recipe))
        original = dict(self.trained.net.state_dict())
        original['projectors.0.0.weight'] = torch.randn(8, 32)
        source = self.root/'original.pt'
        torch.save({'ema': original}, source)
        destination = self.root/'converted'
        with patch.object(sys, 'argv', ['convert_sit.py', str(source), str(destination),
                                       '--family', 'repa', '--key', 'ema', '--config', str(config)]):
            convert()
        model, metadata = load_model(destination, 'cpu')
        self.assertEqual(metadata['kind'], 'base_init')
        self.assertEqual(metadata['model']['data_time'], 0)
        self.assertEqual(metadata['source']['unused_projector_tensors'], ['projectors.0.0.weight'])
        self.assert_state_equal(model.net.state_dict(), self.trained.net.state_dict())
        self.assert_state_equal(model.net.state_dict(), model.corr_net.state_dict())
        torch.testing.assert_close(model.output_gate, torch.zeros(4), rtol=0, atol=0)

    def test_raw_base_files_select_ema_and_preserve_velocity_head_and_corrector_initialization(self):
        for family, direction in [('sit', 1), ('repa', 0)]:
            recipe = copy.deepcopy(self.recipe)
            recipe['model']['data_time'] = direction
            original = dict(self.trained.net.state_dict())
            if family == 'sit':
                for suffix in ('weight', 'bias'):
                    key = f'final_layer.linear.{suffix}'
                    value = original[key]
                    patches = value.reshape(4, 4, *value.shape[1:])
                    original[key] = torch.cat([patches, torch.full_like(patches, 123.)], dim=1).reshape(-1, *value.shape[1:])
            else:
                original['projectors.0.0.weight'] = torch.randn(8, 32)
            source = self.root / f'{family}.pt'
            torch.save({'ema': original, 'model': {}}, source)
            model = build_model(recipe)
            metadata = initialize_model(model, recipe, source, mode='base')
            self.assertEqual(metadata['source']['selected_weights'], 'ema')
            self.assert_state_equal(model.net.state_dict(), self.trained.net.state_dict())
            self.assert_state_equal(model.corr_net.state_dict(), self.trained.net.state_dict())
            torch.testing.assert_close(model.output_gate, torch.zeros(4), rtol=0, atol=0)
            base, correction = model.losses(torch.randn(2, 4, 4, 4), torch.tensor([0, 3]),
                                           t=torch.tensor([.3, .7]), noise=torch.randn(2, 4, 4, 4))
            (base + correction).backward()
            self.assertTrue(torch.isfinite(base + correction))
            self.assertGreater(model.net.final_layer.linear.weight.grad.abs().sum(), 0)
            self.assertIsNotNone(model.output_gate.grad)

    def test_official_jit_conversion_preserves_base_and_ema_without_a_file_allowlist(self):
        from src.tools.convert_checkpoint import main as convert
        import argparse
        name = 'JiT-conversion-test'
        factory = lambda **kw: JiT(hidden_size=32, depth=2, num_heads=4, patch_size=4,
                                  bottleneck_dim=8, in_context_len=2, in_context_start=1, **kw)
        with patch.dict(JiT_models, {name: factory}):
            recipe = resolve(dict(model=dict(model=name, image_size=16, num_classes=5)))
            original = build_model(recipe)
            raw = {'net.'+k: v for k, v in original.net.state_dict().items()}
            ema = {k: v.clone()+.01 for k, v in raw.items()}
            source = self.root / 'official.pth'
            torch.save(dict(model=raw, model_ema1=ema, args=argparse.Namespace(model=name)), source)
            config = self.root / 'jit.json'
            config.write_text(json.dumps(recipe))
            with patch.object(sys, 'argv', ['convert_checkpoint.py', str(source), str(self.root / 'jit'),
                                           '--config', str(config)]):
                convert()
            model, metadata = load_model(self.root / 'jit', 'cpu')
            self.assertEqual(metadata['kind'], 'base_init')
            self.assert_state_equal(model.net.state_dict(), original.net.state_dict())
            self.assert_state_equal(model.corr_net.state_dict(), original.net.state_dict())
            from safetensors.torch import load_file
            self.assert_state_equal(load_file(str(self.root / 'jit/base_ema1.safetensors')),
                                    {k.removeprefix('net.'): v for k, v in ema.items()})

    def test_wrong_start_mode_rejected_without_modifying_model(self):
        for kind, mode in [('pair', 'base'), ('base_init', 'pair')]:
            model = build_model(self.recipe)
            before = copy.deepcopy(model.state_dict())
            with self.assertRaisesRegex(ValueError, 'use --init-'):
                initialize_model(model, self.recipe, self.save(kind), mode=mode)
            self.assert_state_equal(model.state_dict(), before)

    def test_incompatible_recipe_rejected_without_modifying_model(self):
        path = self.save()
        for field, change in [('model', {'data_time': 0}), ('codec', {'scale': .5}),
                              ('correction', {'combine': 'direct'})]:
            recipe = copy.deepcopy(self.recipe)
            recipe[field].update(change)
            model = build_model(recipe)
            before = copy.deepcopy(model.state_dict())
            with self.assertRaises(ValueError):
                initialize_model(model, recipe, path, mode='pair')
            self.assert_state_equal(model.state_dict(), before)



    def test_missing_resume_and_base_bundle_are_rejected_before_cuda(self):
        for options, error in [(['--resume'], '--resume requires'),
                               (['--init-base', 'unused'], 'Initialization requires')]:
            result = subprocess.run([sys.executable, 'src/train.py', '--config', 'unused',
                                     '--data', 'unused', '--output', str(self.root/'run'), *options],
                                    cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(error, result.stderr)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
