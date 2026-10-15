"""Space conversion, repeated correction and gradient boundaries (no GPU required)."""
import sys
import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from src.models.correction import Correction, refine
from src.utils.config import Training
from src.samplers.path import LinearPath
from src.models.denoiser import Denoiser
from src.utils.checkpoint import load_model, save_checkpoint
from src.utils.config import resolve, build_model
from safetensors.torch import save_file


class Scale(nn.Module):
    in_channels = 3

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))

    def forward(self, x, t, cond):
        return self.weight * x


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.x, self.eps = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
        self.t = torch.tensor([.2, .7]).view(2, 1, 1, 1)
        self.y = torch.tensor([1, 2])

    def test_oracle_all_input_output_spaces(self):
        for space, z, v in [
            (LinearPath(t_eps=.05), self.t*self.x+(1-self.t)*self.eps, self.x-self.eps),
            (LinearPath(data_time=0), (1-self.t)*self.x+self.t*self.eps, self.eps-self.x),
        ]:
            truth = dict(x=self.x, eps=self.eps, v=v)
            for input_space in truth:
                for output_space in truth:
                    seen = []
                    def oracle(value, t, y):
                        seen.append(value)
                        return truth[output_space]
                    actual = refine(oracle, z, self.t, self.y, v, space=space,
                                    spec=Correction(input_space, output_space), steps=3)
                    torch.testing.assert_close(actual, v)
                    for value in seen:
                        torch.testing.assert_close(value, truth[input_space])

    def test_original_noisy_state_is_fixed_during_iteration(self):
        space = LinearPath(data_time=0)
        z = (1-self.t)*self.x+self.t*self.eps
        start = torch.zeros_like(z)
        def contract(value, t, y):
            return .5*value + .5*self.x
        actual = refine(contract, z, self.t, self.y, start, space=space, steps=4)
        expected_x = self.x + (z-self.x) * .5**4
        torch.testing.assert_close(actual, (z-expected_x)/self.t)

    def test_gate_identity_direct_and_gradient(self):
        space = LinearPath(data_time=0)
        z = (1-self.t)*self.x+self.t*self.eps
        previous = (self.eps-self.x).requires_grad_()
        net = Scale()
        direct = refine(net, z, self.t, self.y, previous, space=space)
        for init in (0., 1.):
            gate = nn.Parameter(torch.full((3,), init))
            net.zero_grad()
            actual = refine(net, z, self.t, self.y, previous, space=space,
                            spec=Correction(combine='gated'), gate=gate, detach_input=True, detach_skip=True)
            torch.testing.assert_close(actual, previous if init == 0 else direct)
            actual.square().mean().backward()
            self.assertIsNone(previous.grad)
            self.assertTrue(torch.isfinite(gate.grad).all())
            if init == 0:
                self.assertEqual(net.weight.grad.item(), 0.)
            else:
                self.assertNotEqual(net.weight.grad.item(), 0.)

    def test_direct_model_has_no_gate_and_correction_loss_is_detached(self):
        for mode in ('direct', 'gated'):
            model = Denoiser(net=Scale(), num_classes=3, label_drop=0.,
                             correction=Correction(combine=mode, gate_init=.4))
            self.assertEqual('output_gate' in model.state_dict(), mode == 'gated')
            _, corr = model.losses(self.x, self.y, t=self.t, noise=self.eps)
            corr.backward()
            self.assertIsNone(model.net.weight.grad)
            self.assertIsNotNone(model.corr_net.weight.grad)

    def test_endpoints_are_explicit(self):
        with self.assertRaises(ValueError):
            LinearPath(data_time=0).to_velocity(self.x, torch.zeros_like(self.t), self.x, 'x')
        with self.assertRaises(ValueError):
            LinearPath(data_time=0).to_velocity(self.x, torch.ones_like(self.t), self.x, 'eps')
        with self.assertRaises(ValueError):
            LinearPath(t_eps=.05).to_velocity(self.x, torch.zeros_like(self.t), self.x, 'eps')

    def test_independent_gradient_paths_and_repeated_detach(self):
        # C(x)=w*x gives a closed-form Jacobian wrt the original velocity.
        z = (1-self.t)*self.x+self.t*self.eps
        for combine in ('direct', 'gated', 'residual'):
            for detach_input in (False, True):
                for detach_skip in (False, True):
                    for steps in (1, 3):
                        v = (self.eps-self.x).clone().requires_grad_()
                        net = Scale()
                        g = .4
                        actual = refine(net, z, self.t, self.y, v, space=LinearPath(data_time=0),
                                        spec=Correction(combine=combine), steps=steps,
                                        gate=torch.full((3,), g) if combine == 'gated' else None,
                                        detach_input=detach_input, detach_skip=detach_skip)
                        grad = torch.autograd.grad(actual.sum(), v, allow_unused=True)[0]
                        jac = .7 * (not detach_input)
                        if combine == 'gated': jac = g*jac + (1-g)*(not detach_skip)
                        if combine == 'residual': jac += (not detach_skip)
                        if grad is None: grad = torch.zeros_like(v)
                        torch.testing.assert_close(grad, torch.full_like(v, jac**steps))

    def test_residual_is_a_displacement_not_absolute_prediction(self):
        for space in (LinearPath(t_eps=.05), LinearPath(data_time=0)):
            z = self.x
            v = self.eps
            delta = torch.full_like(z, .25)
            actual = refine(lambda *_: delta, z, self.t, self.y, v, space=space,
                            spec=Correction(output='x', combine='residual'))
            expected = v + (delta/(1-self.t) if space.data_time == 1 else -delta/self.t)
            torch.testing.assert_close(actual, expected)

    def test_zero_steps_and_explicit_proposal(self):
        v = self.eps.requires_grad_()
        proposal = self.x.clone().requires_grad_()
        net = Scale()
        self.assertIs(refine(net, self.x, self.t, self.y, v, space=LinearPath(data_time=0), steps=0), v)
        result = refine(net, self.x, self.t, self.y, v, space=LinearPath(data_time=0), proposal=proposal,
                        detach_input=True, detach_skip=False)
        result.sum().backward()
        self.assertIsNone(proposal.grad)
        self.assertIsNone(v.grad)
        self.assertIsNotNone(net.weight.grad)

    def test_training_flags_loss_weights_and_base_loss(self):
        for di in (False, True):
            for ds in (False, True):
                recipe = Training(detach_input=di, detach_skip=ds, base_loss_weight=.3,
                                  correction_loss_weight=2., correction_steps=2)
                model = Denoiser(net=Scale(), label_drop=0., training=recipe,
                                 correction=Correction(combine='gated', gate_init=.4))
                base, corr = model.losses(self.x, self.y, t=self.t, noise=self.eps)
                total = model(self.x, self.y, t=self.t, noise=self.eps)
                torch.testing.assert_close(total, .3*base+2*corr)
                corr.backward()
                self.assertEqual(model.net.weight.grad is None, di and ds)
                model.zero_grad()
                base, _ = model.losses(self.x, self.y, t=self.t, noise=self.eps)
                base.backward()
                self.assertNotEqual(model.net.weight.grad.item(), 0.)

    def test_config_and_canonical_checkpoint(self):
        for bad in ({'unknown': 1}, {'training': {'detach_input': 'false'}},
                    {'training': {'correction_steps': 0}}, {'training': {'loss_space': 'x'}},
                    {'correction': {'combine': 'typo'}}, {'model': {'corrector_input': 'x'}}):
            with self.assertRaises((ValueError, TypeError)):
                resolve(bad)
        recipe = resolve({'correction': {'combine': 'gated', 'gate_init': .4}})
        original = build_model(recipe, net=Scale())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'export'
            save_checkpoint(path, original.state_dict(), recipe, source={'test': True})
            before = (path/'model.safetensors').read_bytes()
            save_checkpoint(path, original.state_dict(), recipe, source={'test': True})
            self.assertEqual(before, (path/'model.safetensors').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Conflicting checkpoint'):
                save_checkpoint(path, original.state_dict(), recipe, source={'test': False})
            with patch('src.utils.checkpoint.build_model', side_effect=lambda cfg: build_model(cfg, net=Scale())):
                loaded, config = load_model(path, 'cpu')
                self.assertEqual(config['training'], recipe['training'])
                for k, value in loaded.state_dict().items():
                    torch.testing.assert_close(value, original.state_dict()[k], rtol=0, atol=0)
                bad = dict(original.state_dict()); del bad['output_gate']
                save_file(bad, str(path/'model.safetensors'))
                with self.assertRaises(RuntimeError): load_model(path, 'cpu')
            (path/'model.safetensors').write_bytes(before)
            (path/'config.json').write_text(json.dumps({'format_version': 1}))
            with self.assertRaisesRegex(ValueError, 'checkpoint bundle'):
                load_model(path, 'cpu')




if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
