"""Base training, frozen-base correction, and base-only checkpoint interoperability."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.utils.checkpoint import initialize_model, load_model, save_checkpoint
from src.utils.config import build_model, read_config, resolve
from src.models.sit import SiT, SiT_models


class TrainingModeTests(unittest.TestCase):
    def setUp(self):
        self.name = 'SiT-mode-test'
        self.mapping = patch.dict(SiT_models, {self.name: lambda **kw: SiT(
            hidden_size=32, depth=1, num_heads=4, patch_size=2, **kw)})
        self.mapping.start()
        self.addCleanup(self.mapping.stop)
        self.recipe = resolve(dict(model=dict(model=self.name, image_size=32, num_classes=5,
                                              label_drop=0., prediction='v'),
                                   training=dict(mode='base', correction_loss_weight=0.),
                                   sampling=dict(corrections=0, steps=3, precision='fp32')))
        self.x, self.noise = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
        self.t, self.labels = torch.tensor([.2, .7]), torch.tensor([0, 3])

    def test_base_has_one_network_and_trains_both_prediction_heads(self):
        for prediction in ('x', 'v'):
            recipe = copy.deepcopy(self.recipe)
            recipe['model']['prediction'] = prediction
            model = build_model(recipe)
            self.assertIsNone(model.corr_net)
            self.assertFalse(any(k.startswith('corr_net.') for k in model.state_dict()))
            before = model.net.final_layer.linear.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss, correction = model.losses(self.x, self.labels, t=self.t, noise=self.noise)
            self.assertEqual(correction.item(), 0.)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.net.final_layer.linear.weight))
            with self.assertRaisesRegex(ValueError, 'no corrector'):
                model.eval().velocity(self.noise, self.t[:, None, None, None], self.labels, corrections=1)

    def test_base_export_samples_and_initializes_frozen_corrector(self):
        original = build_model(self.recipe).eval()
        with torch.no_grad():
            for p in original.net.parameters():
                if p.requires_grad:
                    p.uniform_(-.05, .05)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory+'/base', original.state_dict(), self.recipe,
                            source={'test': True}, kind='base_init')
            loaded, config = load_model(directory+'/base', 'cpu')
            self.assertEqual(config['kind'], 'base_init')
            self.assertIsNone(loaded.corr_net)
            torch.testing.assert_close(
                loaded.sample(self.labels, noise=self.noise, corrections=0, steps=3),
                original.sample(self.labels, noise=self.noise, corrections=0, steps=3), rtol=0, atol=0)
            recipe = copy.deepcopy(self.recipe)
            recipe['training'].update(mode='corrector', base_loss_weight=0., correction_loss_weight=1.)
            recipe['correction'].update(input='x', output='v', combine='direct')
            model = build_model(recipe)
            initialize_model(model, recipe, directory+'/base', mode='base')
            self.assertTrue(all(not p.requires_grad for p in model.net.parameters()))
            self.assertTrue(any(p.requires_grad for p in model.corr_net.parameters()))
            before = copy.deepcopy(model.net.state_dict())
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            _, correction = model.losses(self.x, self.labels, t=self.t, noise=self.noise)
            correction.backward()
            self.assertTrue(all(p.grad is None for p in model.net.parameters()))
            self.assertGreater(model.corr_net.final_layer.linear.weight.grad.abs().sum(), 0.)
            optimizer.step()
            for name, parameter in model.net.state_dict().items():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_b_recipes_use_same_experiment_except_prediction(self):
        root = Path(__file__).resolve().parents[2]
        x, v = [read_config(root/f'configs/train/sit-b-{p}.yml') for p in ('x', 'v')]
        self.assertEqual(x['training']['mode'], 'base')
        self.assertTrue(x['training']['horizontal_flip'])
        self.assertEqual(x['training']['steps'], 400000)
        self.assertEqual(x['training']['betas'], [.9, .999])
        self.assertEqual(x['model']['time_sampling'], 'logit_normal')
        self.assertEqual(x['model']['p_mean'], -.8)
        self.assertEqual(x['model']['p_std'], .8)
        self.assertEqual(x['model']['t_eps'], 0.)
        for recipe in (x, v):
            recipe['model']['prediction'] = 'v'
            recipe['correction'].update(input='v', output='v')
        self.assertEqual(x, v)

    def test_invalid_modes_and_weights_fail_before_allocation(self):
        for training in (dict(mode='typo'), dict(mode='base'),
                         dict(mode='corrector'), dict(seed=-1), dict(horizontal_flip='false')):
            with self.subTest(training=training), self.assertRaises(ValueError):
                resolve(dict(training=training))
        with self.assertRaisesRegex(ValueError, 'corrections=0'):
            resolve(dict(training=dict(mode='base', correction_loss_weight=0.)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
