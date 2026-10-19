"""Sampling-time corrector methods reduce to the paper's sampler at unit scale (no GPU required)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.efficient.guidance import CorrectorGuidance, Method
from src.models.correction import Correction
from src.models.denoiser import Denoiser
from src.models.sit import SiT


def gated_pair():
    torch.manual_seed(0)
    model = Denoiser(net=SiT(input_size=4, hidden_size=32, depth=2, num_heads=4, num_classes=5),
                     image_size=4, in_channels=4, num_classes=5, prediction='v', t_eps=0., time_sampling='uniform',
                     data_time=0, label_drop=0., correction=Correction(input='x', output='v', combine='gated'))
    model.initialize_corrector()
    with torch.no_grad():
        for p in model.corr_net.parameters():
            p.add_(.05 * torch.randn_like(p))
        model.output_gate.copy_(torch.tensor([.04, .05, .03, .06]))
    return model.eval()


class EfficientTests(unittest.TestCase):
    def setUp(self):
        self.model = gated_pair()
        self.z = torch.randn(3, 4, 4, 4)
        self.t = torch.tensor([.55, .6, .68]).view(3, 1, 1, 1)
        self.y = torch.tensor([0, 1, 2])

    def velocity(self, method, corrections=1):
        with torch.no_grad():
            return CorrectorGuidance(self.model, method).velocity(self.z, self.t, self.y, corrections=corrections)

    def official(self, k):
        with torch.no_grad():
            return self.model.velocity(self.z, self.t, self.y, corrections=k)

    def test_unit_scale_is_the_paper(self):
        for k in (1, 2, 5):
            torch.testing.assert_close(self.velocity(Method('paper', k)), self.official(k), rtol=0, atol=0)
            torch.testing.assert_close(self.velocity(Method('relax', k, 1.)), self.official(k), rtol=0, atol=0)
            torch.testing.assert_close(self.velocity(Method('extrap', k, 1.)), self.official(k), rtol=0, atol=1e-7)

    def test_outside_the_interval_is_the_denoiser(self):
        for method in (Method('paper', 5), Method('extrap', 1, 6.), Method('relax', 2, 3.)):
            torch.testing.assert_close(self.velocity(method, corrections=0), self.official(0), rtol=0, atol=0)

    def test_scales(self):
        v, v1 = self.official(0), self.official(1)
        torch.testing.assert_close(self.velocity(Method('extrap', 1, 6.)), v + 6 * (v1 - v))
        relaxed, paper = self.velocity(Method('relax', 2, 3.)), self.official(2)
        self.assertGreater(float((relaxed - v).norm()), float((paper - v).norm()))
        # one relaxed step equals one paper step with the gate scaled by s
        gate = self.model.output_gate.detach().clone()
        with torch.no_grad():
            self.model.output_gate.mul_(3.)
            scaled = self.official(1)
            self.model.output_gate.copy_(gate)
        torch.testing.assert_close(self.velocity(Method('relax', 1, 3.)), scaled, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
