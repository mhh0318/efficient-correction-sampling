"""Pixel/latent I/O. The frozen VAE stays outside DDP, EMA and the denoiser."""
from dataclasses import dataclass
import warnings
import torch
from torch import nn


def sample_posterior(moments, config):
    """Fresh posterior draw from cached FP32 mean/log-variance, matching AutoencoderKL."""
    mean, logvar = moments.float().chunk(2, dim=1)
    std = torch.exp(.5*logvar.clamp(-30., 20.))
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype)
    return (mean + std*noise)*config.scale + config.bias


@dataclass(frozen=True)
class CodecConfig:
    name: str = 'identity'
    pretrained: str = 'stabilityai/sd-vae-ft-ema'
    revision: str | None = None
    scale: float = .18215
    bias: float = 0.

    @property
    def channels(self): return 3 if self.name == 'identity' else 4

    @property
    def downsample(self): return 1 if self.name == 'identity' else 8

    def __post_init__(self):
        if self.name not in ('identity', 'vae') or self.scale <= 0:
            raise ValueError('codec.name must be identity/vae and scale must be positive')


class Codec(nn.Module):
    def __init__(self, config=CodecConfig(), *, vae=None):
        super().__init__()
        self.config = config
        self.vae = None
        if config.name == 'vae':
            if vae is None:
                from diffusers import AutoencoderKL
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', message=r'The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`\.',
                        category=UserWarning, module=r'huggingface_hub\.utils\._validators')
                    vae = AutoencoderKL.from_pretrained(config.pretrained, revision=config.revision)
            if vae.config.latent_channels != config.channels or 2**(len(vae.config.block_out_channels)-1) != config.downsample:
                raise ValueError('Expected a four-channel VAE with 8x downsampling')
            self.vae = vae.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, images):
        if self.vae is None: return images
        # Keep VAE arithmetic in FP32 even under the transformer's BF16 context.
        with torch.autocast(images.device.type, enabled=False):
            z = self.vae.encode(images.float()).latent_dist.sample()
            return z*self.config.scale + self.config.bias

    @torch.no_grad()
    def decode(self, z):
        if self.vae is None: return z
        with torch.autocast(z.device.type, enabled=False):
            return self.vae.decode((z.float()-self.config.bias)/self.config.scale).sample


def build_codec(recipe, device='cuda'):
    return Codec(CodecConfig(**recipe['codec'])).to(device)
