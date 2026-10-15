"""Exercise shared paths, SiT and actual VAE boundaries on small CPU tensors."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.datasets.codec import Codec, CodecConfig, sample_posterior
from src.utils.config import resolve
from src.models.correction import Correction
from src.models.denoiser import Denoiser
from src.models.sit import SiT


def tiny_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                         down_block_types=('DownEncoderBlock2D',)*4,
                         up_block_types=('UpDecoderBlock2D',)*4,
                         block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                         layers_per_block=1, sample_size=32)


class CommonTests(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        for name, channels, size, prediction in [('JiT-L/16', 3, 256, 'x'), ('SiT-XL/2', 4, 32, 'v')]:
            recipe = resolve(dict(model=name))
            self.assertEqual(resolve(recipe), recipe)
            self.assertEqual(recipe['model']['prediction'], prediction)
            codec = CodecConfig(**recipe['codec'])
            self.assertEqual(codec.channels, channels)
            self.assertEqual(recipe['model']['image_size']//codec.downsample, size)

    def test_shared_sit_loss_and_sampling(self):
        for data_time in (0, 1):
            for output in ('x', 'v'):
                model = Denoiser(net=SiT(input_size=4, hidden_size=32, depth=2, num_heads=4, num_classes=5),
                                 image_size=4, in_channels=4, num_classes=5, prediction='v', t_eps=0.,
                                 time_sampling='uniform', data_time=data_time, label_drop=0.,
                                 correction=Correction(output=output))
                model.initialize_corrector()
                x, noise, t = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4), torch.tensor([.3, .6])
                labels = torch.tensor([0, 3])
                base, corr = model.losses(x, labels, t=t, noise=noise)
                torch.testing.assert_close(base, (x-noise).square().mean())
                corr.backward()
                self.assertTrue(all(p.grad is None for p in model.net.parameters()))
                self.assertGreater(model.corr_net.final_layer.linear.weight.grad.abs().sum(), 0)
                model.eval()
                sample = model.sample(labels, noise=noise, steps=4, corrections=2)
                self.assertEqual(sample.shape, x.shape)
                self.assertTrue(torch.isfinite(sample).all())
                baseline = model.sample(labels, noise=noise, steps=4, corrections=0)
                torch.testing.assert_close(baseline, noise, rtol=0, atol=0)

    def test_time_reversal_solver(self):
        # A constant oracle must reach the same data under either convention.
        for data_time in (0, 1):
            model = Denoiser(net=SiT(input_size=4, hidden_size=32, depth=1, num_heads=4, num_classes=5),
                             image_size=4, prediction='v', t_eps=0., data_time=data_time)
            noise, x = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
            model.velocity = lambda *args, **kwargs: x-noise if data_time else noise-x
            result = model.sample(torch.tensor([0, 1]), noise=noise, steps=4)
            torch.testing.assert_close(result, x, rtol=1e-5, atol=1e-6)

    def test_real_vae_normalization_and_grad_boundary(self):
        vae = tiny_vae()
        codec = Codec(CodecConfig(name='vae', scale=.3, bias=.2), vae=vae)
        images = torch.randn(2, 3, 32, 32, requires_grad=True)
        torch.manual_seed(12)
        raw = vae.encode(images).latent_dist.sample()
        torch.manual_seed(12)
        z = codec.encode(images)
        torch.testing.assert_close(z, raw*.3+.2)
        self.assertEqual(z.shape, (2, 4, 4, 4))
        self.assertFalse(z.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in vae.parameters()))
        torch.testing.assert_close(codec.decode(z), vae.decode(raw).sample, rtol=1e-4, atol=1e-5)
        self.assertEqual(codec.decode(z).shape, images.shape)
        identity = Codec()
        self.assertIs(identity.encode(images), images)
        self.assertIs(identity.decode(images), images)

    def test_cached_posterior_preserves_sampling(self):
        codec = Codec(CodecConfig(name='vae', scale=.3, bias=.2), vae=tiny_vae())
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            posterior = codec.vae.encode(images).latent_dist
        torch.manual_seed(17)
        expected = posterior.sample()*.3+.2
        torch.manual_seed(17)
        actual = sample_posterior(posterior.parameters, codec.config)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        another = sample_posterior(posterior.parameters, codec.config)
        self.assertFalse(torch.equal(actual, another))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
