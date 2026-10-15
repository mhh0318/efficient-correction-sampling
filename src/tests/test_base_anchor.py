"""Complementary-component anchors remain fixed across corrections."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.correction import Correction, refine
from src.samplers.path import LinearPath


class BaseAnchorTests(unittest.TestCase):
    def test_heads_and_repeated_mapping_match_explicit_reference(self):
        torch.manual_seed(5)
        z, initial = torch.randn(2, 4, 3, 3), torch.randn(2, 4, 3, 3)
        t, labels = torch.tensor([.2, .8])[:, None, None, None], torch.tensor([0, 1])
        for direction in (0, 1):
            path = LinearPath(direction)
            clean = path.from_velocity(z, t, initial, 'x')
            noise = path.from_velocity(z, t, initial, 'eps')
            for input_space in ('x', 'v', 'eps'):
                for output_space in ('x', 'v', 'eps'):
                    spec = Correction(input=input_space, output=output_space, combine='base_anchor')
                    for k in (1, 3):
                        expected = initial
                        for _ in range(k):
                            q = .7*path.from_velocity(z, t, expected, input_space)+.1
                            if output_space == 'v':
                                expected = q
                            else:
                                field = q-noise if output_space == 'x' else clean-q
                                expected = field if direction == 1 else -field
                        actual = refine(lambda value, *_: .7*value+.1, z, t, labels, initial,
                                        space=path, spec=spec, steps=k)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_anchor_is_not_fixed_state_conversion_or_residual_addition(self):
        z = torch.ones(1, 1, 2, 2)
        v = torch.full_like(z, 2.)
        t = torch.full((1, 1, 1, 1), .25)
        labels = torch.tensor([0])
        path = LinearPath(1)
        net = lambda value, *_: torch.full_like(value, 3.)
        def calculate(combine):
            return refine(net, z, t, labels, v, space=path,
                          spec=Correction(input='v', output='x', combine=combine))
        # epsilon_base = 1 - .25*2 = .5, so the anchored field is 3 - .5.
        torch.testing.assert_close(calculate('base_anchor'), torch.full_like(v, 2.5))
        self.assertFalse(torch.equal(calculate('base_anchor'), calculate('direct')))
        self.assertFalse(torch.equal(calculate('base_anchor'), calculate('residual')))

    def test_detached_anchor_and_input_do_not_backpropagate_into_d(self):
        z = torch.randn(2, 1, 2, 2)
        velocity = torch.randn_like(z, requires_grad=True)
        t = torch.full((2, 1, 1, 1), .4)
        weight = torch.nn.Parameter(torch.tensor(.9))
        result = refine(lambda value, *_: weight*value, z, t, torch.tensor([0, 1]), velocity,
                        space=LinearPath(1), spec=Correction(input='v', output='x', combine='base_anchor'),
                        steps=3, detach_input=True, detach_skip=True)
        result.square().mean().backward()
        self.assertIsNone(velocity.grad)
        self.assertTrue(torch.isfinite(weight.grad))
        self.assertNotEqual(weight.grad.item(), 0.)


if __name__ == '__main__':
    unittest.main()
