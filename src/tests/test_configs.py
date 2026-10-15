"""Verify released recipes, checkpoint selection, and sampling config boundaries."""
import copy
from contextlib import chdir
from contextlib import ExitStack
import json
import yaml
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import read_config, read_sampling_config, resolve, sampling_recipe

ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.models = json.loads((ROOT / 'configs/checkpoints.json').read_text())['models']

    def test_training_recipes_preserve_released_model_and_optimizer_settings(self):
        for name, entry in self.models.items():
            if entry.get('conditioning') == 'unconditional':
                continue
            with self.subTest(name=name):
                recipe = read_config(ROOT / f'configs/train/{"sit-repa" if name.startswith("sit-repa") else name}.yml')
                expected = resolve({k: v for k, v in entry['config'].items()
                                    if k in ('model', 'codec', 'correction', 'training', 'sampling')})
                if name.startswith('sit-repa'):
                    expected['training']['steps'] = 100000
                    expected['sampling'] = sampling_recipe(self.models['sit-repa-50k']['config'])
                expected['training'].update(save_every=5000, export_every=10000)
                self.assertEqual(recipe, expected)
                self.assertEqual(recipe['training']['steps'], 100000)

    def test_quantitative_configs_preserve_validated_sampling_recipes(self):
        for path in (ROOT / 'configs/eval/quantitative').glob('*.yml'):
            with self.subTest(path=path.name):
                preset = read_sampling_config(path)
                entry = self.models[preset['checkpoint']]
                self.assertEqual(sampling_recipe(entry['config'], preset), sampling_recipe(entry['config']))
                self.assertEqual(preset['generation']['samples'], 50000)
                self.assertEqual(preset['generation']['runner'],
                                 {'jit': 'jit', 'sit': 'sit', 'sit-repa-50k': 'repa'}[preset['checkpoint']])

    def test_qualitative_presets_use_the_checkpoint_sampler_without_guidance(self):
        for path in (ROOT / 'configs/eval/qualitative').glob('*.yml'):
            with self.subTest(path=path.name):
                preset = read_sampling_config(path)
                name = preset['checkpoint']
                original = sampling_recipe(self.models[name]['config'])
                sampling = sampling_recipe(self.models[name]['config'], preset)
                self.assertEqual(sampling['method'], original['method'])
                self.assertEqual(sampling['precision'], original['precision'])
                self.assertEqual(sampling['cfg'], 1.)
                generation = (dict(runner='sample', seed=0, samples=4, unconditional=True)
                              if name == 'sit-repa-uncond-100k' else
                              dict(runner='sample', seed=0, classes=[207], samples=16, batch_size=4))
                self.assertEqual(preset['generation'], generation)

    def test_sampling_presets_can_be_used_with_finetuned_weights(self):
        preset = read_sampling_config(ROOT / 'configs/eval/quantitative/sit-repa.yml')
        result = sampling_recipe(self.models['sit-repa-100k']['config'], preset)
        self.assertEqual(result, sampling_recipe(self.models['sit-repa-50k']['config']))

    def test_public_yaml_names_and_noise_intervals_preserve_native_checkpoint_settings(self):
        for path in (ROOT / 'configs').rglob('*.yml'):
            with self.subTest(path=str(path)):
                raw = yaml.safe_load(path.read_text())
                if 'sampling' not in raw:
                    continue
                self.assertIn('denoising_steps', raw['sampling'])
                self.assertIn('corrections_per_step', raw['sampling'])
                self.assertNotIn('steps', raw['sampling'])
                self.assertNotIn('corrections', raw['sampling'])
                if 'checkpoint' in raw:
                    preset = read_sampling_config(path)
                    sampling_recipe(self.models[preset['checkpoint']]['config'], preset)
                else:
                    read_config(path)
        config = self.models['jit']['config']
        public = dict(sampling=dict(denoising_steps=10, corrections_per_step=0, cfg=3.,
                                    cfg_interval=[.9, .7]))
        native = sampling_recipe(config, public)
        self.assertEqual(native['cfg_interval'], [.1, .3])
        self.assertEqual(native['steps'], 10)
        self.assertEqual(native['corrections'], 0)
        for name in ('sit', 'sit-repa-100k'):
            self.assertEqual(sampling_recipe(self.models[name]['config'], public)['cfg_interval'], [.7, .9])

    def test_public_intervals_preserve_native_bounds_and_legacy_json(self):
        for name in ('jit', 'sit', 'sit-repa-100k'):
            config = self.models[name]['config']
            public = dict(sampling=dict(corrections=0, cfg=3.,
                          correction_interval=[.9, .6], cfg_interval=[.8, .2]))
            native = sampling_recipe(config, public)
            self.assertEqual(native['correction_interval'], [.6, .9])
            self.assertEqual(native['cfg_interval'], [.2, .8])
            for field in ('correction_interval', 'cfg_interval'):
                for bounds in ([0., 1.], [.5, .5], [1.1, 0.]):
                    with self.subTest(name=name, field=field, bounds=bounds), self.assertRaisesRegex(ValueError, 'start t, end t'):
                        sampling_recipe(config, dict(sampling={field: bounds}))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'legacy.json'
                path.write_text(json.dumps(dict(checkpoint=name, sampling=sampling_recipe(config),
                                               generation=dict(runner='sample',seed=0,classes=[207]))))
                self.assertEqual(sampling_recipe(config, read_sampling_config(path)),
                                 sampling_recipe(config))

    def test_imagenet_rejects_cfg_with_correction(self):
        import io
        from contextlib import redirect_stderr
        from src.sample import main as sample
        from src.models.denoiser import Denoiser
        from src.samplers.heun import sample as heun
        from src.samplers.sit import sample as sit
        from src.samplers.repa import sample as repa

        for name in ('jit', 'sit', 'sit-repa-100k'):
            config = self.models[name]['config']
            for cfg in (0.5, 3.):
                with self.subTest(name=name, cfg=cfg):
                    with self.assertRaisesRegex(ValueError, 'CFG with correction'):
                        sampling_recipe(config, dict(sampling=dict(corrections=2, cfg=cfg)))
            for corrections, cfg in ((2, 1.), (0, 3.)):
                result = sampling_recipe(config, dict(sampling=dict(corrections=corrections, cfg=cfg)))
                self.assertEqual((result['corrections'], result['cfg']), (corrections, cfg))

        # Reject before accessing the model or generating noise, including direct calls.
        calls = [lambda: heun(None, None, corrections=2, cfg=3.),
                 lambda: Denoiser.velocity(None, None, None, None, corrections=2, cfg=3.)]
        for solver in (sit, repa):
            calls.append(lambda solver=solver: solver(None, None, None, steps=3, corrections=2,
                         correction_interval=[0., 1.], cfg=3., cfg_interval=[0., 1.]))
        for call in calls:
            with self.assertRaisesRegex(ValueError, 'CFG with correction'):
                call()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(self.models['sit-repa-100k']['config']))
            stderr = io.StringIO()
            with patch.object(sys, 'argv', ['sample.py', '--checkpoint', 'sit-repa-100k', '--cfg', '3']), \
                    patch('src.sample.resolve_checkpoint', return_value=root), \
                    patch('src.utils.checkpoint.load_model') as load, redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as error:
                    sample()
            self.assertEqual(error.exception.code, 2)
            self.assertIn('CFG with correction', stderr.getvalue())
            self.assertIn('--corrections_per_step 0', stderr.getvalue())
            load.assert_not_called()

    def test_official_checkpoint_without_config_uses_qualitative_preset_and_cuda_tf32_policy(self):
        import torch
        from src.sample import main as sample
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(1, 1)
            model.num_classes = 1000
            for name in ('jit', 'sit', 'sit-repa-50k', 'sit-repa-100k', 'sit-repa-uncond-100k'):
                preset_name = ('sit-repa-uncond' if name == 'sit-repa-uncond-100k' else
                               'sit-repa' if name.startswith('sit-repa') else name)
                expected_preset = read_sampling_config(ROOT / f'configs/eval/qualitative/{preset_name}.yml')
                metadata = self.models[name]['config']
                metadata = metadata | resolve({k: v for k, v in metadata.items()
                                               if k in ('model', 'codec', 'correction', 'training', 'sampling')})
                (root / 'config.json').write_text(json.dumps(metadata))
                with self.subTest(name=name), \
                        patch.object(sys, 'argv', ['sample.py', '--checkpoint', name, '--classes', '207',
                                                  '--output', str(root / 'samples')]), \
                        patch('src.sample.resolve_checkpoint', return_value=root), \
                        patch('src.utils.checkpoint.load_model', return_value=(model, metadata)), \
                        patch('src.sample.sampling_recipe', wraps=sampling_recipe) as recipe, \
                        patch('src.datasets.codec.build_codec', side_effect=RuntimeError('stop before VAE allocation')), \
                        ExitStack() as backend_settings:
                    for backend in (torch.backends.cuda.matmul, torch.backends.cudnn):
                        backend_settings.callback(setattr, backend, 'allow_tf32', backend.allow_tf32)
                    with self.assertRaisesRegex(RuntimeError, 'stop before VAE allocation'):
                        sample()
                    self.assertEqual(recipe.call_args.args[1]['sampling'], expected_preset['sampling'])
                    self.assertEqual(torch.backends.cuda.matmul.allow_tf32, name != 'jit')
                    self.assertEqual(torch.backends.cudnn.allow_tf32, name != 'jit')

    def test_sampling_configs_reject_typos_incomplete_recipes_and_invalid_cohorts(self):
        original = read_sampling_config(ROOT / 'configs/eval/qualitative/jit.yml')
        cases = []
        value = copy.deepcopy(original)
        value['sampling']['correction'] = value['sampling'].pop('corrections')
        cases.append(value)
        value = copy.deepcopy(original)
        value['generation']['seed'] = -1
        cases.append(value)
        value = copy.deepcopy(original)
        value['generation']['classes'] = []
        cases.append(value)
        value = copy.deepcopy(original)
        value['training'] = {}
        cases.append(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.yml'
            for value in cases:
                path.write_text(yaml.safe_dump(value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    read_sampling_config(path)

    def test_sampling_config_cannot_switch_native_time_direction(self):
        preset = read_sampling_config(ROOT / 'configs/eval/quantitative/sit.yml')
        preset['sampling']['method'] = 'repa_sde'
        with self.assertRaisesRegex(ValueError, 'matching SiT velocity path'):
            sampling_recipe(self.models['sit']['config'], preset)

    def test_base_converter_defaults_load_matching_reorganized_train_recipe(self):
        import sys
        import torch
        from src.tools import convert_sit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'base.pt'
            torch.save({'x_embedder.proj.weight': torch.empty(4, 4, 2, 2),
                        'final_layer.linear.weight': torch.empty(16, 32)}, source)
            for family, name in [('sit', 'sit'), ('repa', 'sit-repa')]:
                with self.subTest(family=family), \
                        patch.object(sys, 'argv', ['convert_sit.py', str(source), str(root / family), '--family', family]), \
                        patch.object(convert_sit, 'build_model', side_effect=RuntimeError('stop before allocation')) as build, \
                        self.assertRaisesRegex(RuntimeError, 'stop before allocation'):
                    convert_sit.main()
                self.assertEqual(build.call_args.args[0], read_config(ROOT / f'configs/train/{name}.yml'))

    def test_sample_config_and_cli_overrides_generate_expected_png_on_cpu(self):
        import sys
        import numpy as np
        from PIL import Image
        import torch
        from src.utils.checkpoint import save_checkpoint
        from src.utils.config import build_model
        from src.models.jit import JiT, JiT_models
        from src.sample import main as sample
        name = 'JiT-config-test'
        with tempfile.TemporaryDirectory() as directory, patch.dict(JiT_models, {name: lambda **kw: JiT(
                hidden_size=32, depth=2, num_heads=4, patch_size=4,
                bottleneck_dim=8, in_context_len=2, in_context_start=1, **kw)}):
            root = Path(directory)
            recipe = resolve(dict(model=dict(model=name, image_size=16, num_classes=5),
                                  sampling=dict(precision='fp32', steps=3, corrections=1,
                                                correction_interval=[0., 1.])))
            model = build_model(recipe).eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.uniform_(-.03, .03)
            bundle = root / 'bundle'
            save_checkpoint(bundle, model.state_dict(), recipe, source={'test': True})
            preset = dict(checkpoint='local-only',
                          sampling=recipe['sampling'] | dict(steps=4, corrections=2,
                              correction_interval=[1., 0.], cfg_interval=[.9, 0.]),
                          generation=dict(runner='sample', seed=12, classes=[0, 3]))
            path = root / 'preset.yml'
            path.write_text('# Sampling defaults; CLI overrides take priority.\n' + yaml.safe_dump(preset))
            output = root / 'images'
            output.mkdir()
            (output / 'keep.txt').write_text('Existing results stay here.')
            with patch.object(sys, 'argv', ['sample.py', '--config', str(path), '--checkpoint', str(bundle),
                                           '--device', 'cpu', '--seed', '17', '--classes', '3',
                                           '--denoising_steps', '3', '--corrections_per_step', '1',
                                           '--correction-interval', '0.75', '0.25',
                                           '--cfg', '1', '--cfg-interval', '0.7', '0.2',
                                           '--precision', 'fp32', '--output', str(output)]), \
                    patch('src.sample.datetime') as clock:
                clock.now.return_value.strftime.return_value = '20300101-000000'
                sample()
                original_png = (output / '20300101-000000/000-class3.png').read_bytes()
                sample()
            self.assertEqual((output / 'keep.txt').read_text(), 'Existing results stay here.')
            self.assertEqual((output / '20300101-000000/000-class3.png').read_bytes(), original_png)
            self.assertEqual((output / '20300101-000000-1/000-class3.png').read_bytes(), original_png)
            output = output / '20300101-000000'
            record = json.loads((output / 'sampling.json').read_text())
            self.assertEqual(record['seed'], 17)
            self.assertEqual(record['classes'], [3])
            expected_recipe = preset['sampling'] | dict(steps=3, corrections=1, correction_interval=[.25, .75],
                                                       cfg=1., cfg_interval=[.3, .8])
            expected_recipe.pop('precision')
            self.assertEqual(record['sampling'], expected_recipe)
            torch.manual_seed(17)
            images = model.sample(torch.tensor([3]), **expected_recipe)
            pixels = ((images + 1) / 2).detach().numpy().transpose(0, 2, 3, 1)
            expected = np.round(np.clip(pixels[0] * 255, 0, 255)).astype(np.uint8)
            with Image.open(output / '000-class3.png') as actual:
                np.testing.assert_array_equal(np.asarray(actual), expected)

            preset['generation'].update(samples=5, batch_size=2, unconditional=True)
            path.write_text(yaml.safe_dump(preset))
            for unconditional in (False, True):
                count, batch_size, seed = (3, 1, 0) if unconditional else (5, 2, 42)
                output = root / ('unconditional' if unconditional else 'conditional')
                flags = (['--unconditional', '--samples', '3', '--batch-size', '1'] if unconditional else
                         ['--classes', '3', '1'])
                with self.subTest(unconditional=unconditional), patch.object(sys, 'argv', [
                        'sample.py', '--config', str(path), '--checkpoint', str(bundle), '--device', 'cpu',
                        '--seed', str(seed), '--steps', '3', '--corrections', '0',
                        '--cfg', '2', '--output', str(output), *flags]):
                    sample()
                output = next(output.glob('*/sampling.json')).parent
                record = json.loads((output / 'sampling.json').read_text())
                self.assertEqual(record['seed'], seed)
                self.assertEqual(record['samples'], count)
                self.assertEqual(record['batch_size'], batch_size)
                self.assertEqual(record['unconditional'], unconditional)
                self.assertEqual(record['classes'], None if unconditional else [3, 1, 3, 1, 3])
                self.assertEqual(record['sampling']['cfg'], 1. if unconditional else 2.)
                self.assertEqual(record['calls_per_image']['correction'], 0)
                self.assertEqual(record['calls_per_image']['base'], 5 if unconditional else 10)
                with Image.open(output / 'grid.png') as grid:
                    self.assertEqual(grid.size, (48, 16) if unconditional else (64, 32))
                torch.manual_seed(seed)
                for start in range(0, count, batch_size):
                    size = min(batch_size, count - start)
                    labels = torch.full((size,), 5) if unconditional else torch.tensor([3, 1][:size])
                    expected = model.sample(labels, **record['sampling'])
                    expected = np.round(np.clip(((expected+1)/2).numpy().transpose(0, 2, 3, 1)*255, 0, 255)).astype(np.uint8)
                    for i, pixels in enumerate(expected, start=start):
                        label = 'unconditional' if unconditional else f'class{record["classes"][i]}'
                        with Image.open(output / f'{i:03d}-{label}.png') as actual:
                            np.testing.assert_array_equal(np.asarray(actual), pixels)

            metadata = json.loads((bundle / 'config.json').read_text())
            metadata['conditioning'] = 'unconditional'
            (bundle / 'config.json').write_text(json.dumps(metadata))
            output = root / 'dedicated-unconditional'
            with patch.object(sys, 'argv', ['sample.py', '--checkpoint', str(bundle), '--device', 'cpu',
                                           '--samples', '2', '--batch-size', '1', '--seed', '42',
                                           '--corrections_per_step', '0', '--output', str(output)]):
                sample()
            output = next(output.glob('*/sampling.json')).parent
            record = json.loads((output / 'sampling.json').read_text())
            self.assertTrue(record['unconditional'])
            self.assertIsNone(record['classes'])
            self.assertEqual(record['sampling']['cfg'], 1.)
            torch.manual_seed(42)
            for i in range(2):
                expected = model.sample(torch.tensor([5]), **record['sampling'])
                pixels = np.round(np.clip(((expected+1)/2).numpy().transpose(0, 2, 3, 1)*255, 0, 255)).astype(np.uint8)
                with Image.open(output / f'{i:03d}-unconditional.png') as actual:
                    np.testing.assert_array_equal(np.asarray(actual), pixels[0])

            with chdir(root), patch.object(sys, 'argv', ['sample.py', '--checkpoint', str(bundle),
                    '--device', 'cpu', '--classes', '3', '--samples', '1', '--corrections_per_step', '0']):
                sample()
            default_output = next((root / 'samples/bundle').glob('*/sampling.json')).parent
            self.assertTrue((default_output / 'grid.png').is_file())


if __name__ == '__main__':
    unittest.main()
