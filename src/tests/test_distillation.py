"""Frozen targets, plain students and checkpoint/sampling interoperability."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.utils.checkpoint import load_model, save_checkpoint
from src.utils.config import build_model, read_config, resolve, sampling_recipe
from src.models.distillation import initialize_student, teacher_targets
from src.models.sit import SiT, SiT_models


class DistillationTests(unittest.TestCase):
    def setUp(self):
        self.name = 'SiT-distillation-test'
        mapping = patch.dict(SiT_models, {self.name: lambda **kw: SiT(
            hidden_size=32, depth=1, num_heads=4, patch_size=2, **kw)})
        mapping.start(); self.addCleanup(mapping.stop)
        self.recipe = resolve(dict(model=dict(model=self.name, image_size=32, label_drop=0.),
            correction=dict(input='x', output='v', combine='gated', gate_shape='scalar', gate_mix='blend'),
            sampling=dict(steps=3, corrections=2, precision='fp32', correction_interval=[0., 1.])))
        self.x, self.noise = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
        self.t, self.labels = torch.tensor([.2, .7]), torch.tensor([0, 3])

    def pair(self, direction=1, mix='blend'):
        recipe = copy.deepcopy(self.recipe)
        recipe['model']['data_time'] = direction
        recipe['correction']['gate_mix'] = mix
        teacher = build_model(recipe).eval().requires_grad_(False)
        with torch.no_grad():
            for parameter in teacher.parameters():
                parameter.uniform_(-.04, .04)
            teacher.output_gate.fill_(.3)
        student_recipe = resolve(recipe | dict(
            training=dict(mode='distill', correction_loss_weight=0., warmup_epochs=0., warmup_steps=100),
            distillation=dict(teacher_corrections=5),
            sampling=recipe['sampling'] | dict(corrections=0)))
        student = build_model(student_recipe)
        initialize_student(student, teacher, student_recipe, recipe)
        return teacher, student, student_recipe

    def test_targets_match_manual_gated_iteration_in_both_time_directions(self):
        for direction in (0, 1):
            for mix in ('blend', 'difference'):
                teacher, _, _ = self.pair(direction, mix)
                z, t, base, target = teacher_targets(teacher, self.x, self.labels, 5, t=self.t, noise=self.noise)
                reference = base
                for _ in range(5):
                    clean = teacher.space.from_velocity(z, t, reference, 'x')
                    candidate = teacher.corr_net(clean, t.flatten(), self.labels)
                    reference = (.7*reference + .3*candidate if mix == 'blend' else
                                 reference + .3*(candidate-reference))
                torch.testing.assert_close(target, reference, rtol=0, atol=0)
                self.assertFalse(target.requires_grad)

    def test_initial_output_matches_d_and_only_student_receives_gradients(self):
        teacher, student, _ = self.pair()
        before = copy.deepcopy(teacher.state_dict())
        z, t, base, target = teacher_targets(teacher, self.x, self.labels, 5, t=self.t, noise=self.noise)
        torch.testing.assert_close(student.velocity(z, t, self.labels, corrections=0), base, rtol=0, atol=0)
        self.assertIsNone(student.corr_net)
        self.assertFalse(hasattr(student, 'output_gate'))
        target = target.clone().requires_grad_()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        loss, correction = student(z, self.labels, t=t.flatten(), target_velocity=target, return_losses=True)
        torch.testing.assert_close(loss, (base-target).square().mean(), rtol=0, atol=0)
        self.assertEqual(correction.item(), 0.)
        loss.backward(); optimizer.step()
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for key, tensor in teacher.state_dict().items():
            torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(student.net.final_layer.linear.weight, teacher.net.final_layer.linear.weight))

    def test_export_keeps_distillation_metadata_and_samples_without_corrector(self):
        teacher, student, recipe = self.pair()
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory+'/student', student.state_dict(), recipe,
                            source={'distillation': {'teacher_corrections': 5}}, kind='base_init')
            loaded, metadata = load_model(directory+'/student', 'cpu')
            self.assertEqual(metadata['distillation']['teacher_corrections'], 5)
            self.assertTrue(all(key.startswith('net.') for key in loaded.state_dict()))
            self.assertEqual(sampling_recipe(metadata)['corrections'], 0)
            torch.manual_seed(42)
            expected = student.sample(self.labels, noise=self.noise, steps=3, corrections=0)
            torch.manual_seed(42)
            actual = loaded.sample(self.labels, noise=self.noise, steps=3, corrections=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual).all())

    def test_canonical_conversion_preserves_teacher_depth_and_student_weights(self):
        from src.tools.convert_checkpoint import main as convert
        for depth in (2, 5):
            with self.subTest(teacher_corrections=depth), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, student, recipe = self.pair()
                recipe['distillation']['teacher_corrections'] = depth
                save_checkpoint(root / 'student', student.state_dict(), recipe,
                                source={'distillation': {'teacher_corrections': depth}}, kind='base_init')
                with patch.object(sys, 'argv', ['convert_checkpoint.py', str(root / 'student'),
                                               str(root / 'converted')]):
                    convert()
                loaded, metadata = load_model(root / 'converted', 'cpu')
                for field in ('model', 'codec', 'correction', 'training', 'sampling', 'distillation'):
                    self.assertEqual(metadata[field], recipe[field])
                self.assertEqual(metadata['kind'], 'base_init')
                self.assertEqual(metadata['source']['original']['distillation']['teacher_corrections'], depth)
                self.assertIsNone(loaded.corr_net)
                self.assertEqual(loaded.state_dict().keys(), student.state_dict().keys())
                for key, tensor in student.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)

    def test_sample_cli_supports_a_plain_student(self):
        from src.tests.test_common import tiny_vae
        from src import sample
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_vae().save_pretrained(root/'vae')
            self.recipe['codec']['pretrained'] = str(root/'vae')
            self.recipe['codec']['revision'] = None
            _, student, recipe = self.pair()
            save_checkpoint(root/'student', student.state_dict(), recipe, source={}, kind='base_init')
            with patch.object(sys, 'argv', ['sample.py', '--checkpoint', str(root/'student'),
                    '--device', 'cpu', '--classes', '0', '--steps', '2', '--output', str(root/'samples')]):
                sample.main()
            output = next((root/'samples').glob('*/sampling.json')).parent
            result = json.loads((output/'sampling.json').read_text())
            self.assertEqual(result['calls_per_image']['correction'], 0)
            self.assertEqual(result['sampling']['corrections'], 0)
            self.assertTrue((output/'grid.png').is_file())

    def test_bad_teacher_and_missing_target_are_rejected(self):
        teacher, student, recipe = self.pair()
        bad = copy.deepcopy(recipe); bad['model']['data_time'] = 0
        with self.assertRaisesRegex(ValueError, 'differs'):
            initialize_student(student, teacher, bad, self.recipe)
        with self.assertRaisesRegex(ValueError, 'trained pair'):
            initialize_student(student, teacher, recipe, self.recipe | dict(kind='base_init'))
        with self.assertRaisesRegex(ValueError, 'target_velocity'):
            student(self.x, self.labels)

    def test_recipes_and_validation(self):
        root = Path(__file__).resolve().parents[2]
        for family, k, steps, direction in [('sit', 2, 100000, 1), ('sit-repa', 5, 5000, 0)]:
            recipe = read_config(root/f'configs/distill/{family}.yml')
            self.assertEqual(recipe['distillation']['teacher_corrections'], k)
            self.assertEqual(recipe['training']['mode'], 'distill')
            self.assertEqual(recipe['training']['steps'], steps)
            self.assertEqual(recipe['training']['warmup_steps'], 100)
            self.assertEqual(recipe['training']['batch_size'] * recipe['training']['accumulation'] * 8, 256)
            self.assertEqual(recipe['model']['data_time'], direction)
            self.assertEqual(recipe['sampling']['corrections'], 0)
        _, _, good = self.pair()
        for key, value in [('teacher_corrections', 0), ('teacher_corrections', True), ('typo', 2)]:
            bad = copy.deepcopy(good); bad['distillation'] = {key: value}
            with self.assertRaises((ValueError, TypeError)):
                resolve(bad)
        for update in [dict(warmup_steps=-1), dict(warmup_epochs=5.), dict(grad_clip=-1)]:
            bad = copy.deepcopy(good); bad['training'].update(update)
            with self.assertRaises(ValueError): resolve(bad)
        bad = copy.deepcopy(good); bad['model']['label_drop'] = .1
        with self.assertRaisesRegex(ValueError, 'label_drop'): resolve(bad)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
