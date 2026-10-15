"""One training and velocity interface for pixel and latent backbones."""
import copy

import torch
from torch import nn

from .jit import JiT_models
from .sit import SiT_models
from .correction import Correction, refine
from src.utils.config import Training, validate_guidance
from src.samplers.path import LinearPath
from src.samplers import sample


class Denoiser(nn.Module):
    def __init__(self, model="JiT-L/16", image_size=256, num_classes=1000,
                 t_eps=0.05, noise_scale=1.0, p_mean=-0.8, p_std=0.8,
                 label_drop=0.1, prediction="x", data_time=1, time_sampling="logit_normal",
                 in_channels=3, net=None, correction=Correction(), training=Training()):
        super().__init__()
        self.net = net if net is not None else (JiT_models | SiT_models)[model](
            input_size=image_size, num_classes=num_classes, in_channels=in_channels)
        self.corr_net = None if training.mode in ('base', 'distill') else copy.deepcopy(self.net)
        if training.mode == 'corrector':
            self.net.requires_grad_(False)
        self.image_size, self.num_classes = image_size, num_classes
        self.t_eps, self.noise_scale = t_eps, noise_scale
        self.p_mean, self.p_std, self.label_drop = p_mean, p_std, label_drop
        self.correction, self.recipe = correction, training
        self.prediction, self.time_sampling = prediction, time_sampling
        self.space = LinearPath(data_time, t_eps)
        if correction.combine == "gated" and self.corr_net is not None:
            ref = next(self.net.parameters())
            channels = 1 if correction.gate_shape == "scalar" else self.net.in_channels
            self.output_gate = nn.Parameter(ref.new_full((channels,), float(correction.gate_init)))

    def correct(self, z, t, labels, velocity, *, steps=1, detach_input=False,
                detach_skip=False, proposal=None):
        if self.corr_net is None:
            if steps:
                raise ValueError('Base-only checkpoint has no corrector; use corrections=0')
            return velocity
        return refine(self.corr_net, z, t, labels, velocity, space=self.space,
                      spec=self.correction, steps=steps, gate=getattr(self, "output_gate", None),
                      detach_input=detach_input, detach_skip=detach_skip, proposal=proposal)

    def initialize_corrector(self):
        """Call after base initialization, never after loading a trained pair."""
        if self.corr_net is None:
            return
        self.corr_net.load_state_dict(self.net.state_dict(), strict=True)
        if self.correction.combine == "residual" or self.correction.output != self.prediction:
            nn.init.zeros_(self.corr_net.final_layer.linear.weight)
            nn.init.zeros_(self.corr_net.final_layer.linear.bias)

    def losses(self, x, labels, *, t=None, noise=None):
        if self.training:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop
            labels = torch.where(drop, self.num_classes, labels)
        if t is None:
            if self.time_sampling == "uniform":
                t = torch.rand(x.size(0), device=x.device).clamp_min(torch.finfo(torch.float32).eps)
            else:
                t = (torch.randn(x.size(0), device=x.device) * self.p_std + self.p_mean).sigmoid()
        t = t.reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x) * self.noise_scale if noise is None else noise
        z = self.space.interpolate(x, noise, t)
        target = self.space.target(x, noise, z, t)
        prediction = self.net(z, t.flatten(), labels)
        velocity = self.space.to_velocity(z, t, prediction, self.prediction)
        base_loss = ((target - velocity) ** 2).mean((1, 2, 3)).mean()
        if self.corr_net is None:
            return base_loss, base_loss.new_zeros(())
        corrected = self.correct(z, t, labels, velocity, steps=self.recipe.correction_steps,
                                 detach_input=self.recipe.detach_input, detach_skip=self.recipe.detach_skip,
                                 proposal=prediction if self.correction.input == self.prediction else None)
        corr_loss = ((target - corrected) ** 2).mean((1, 2, 3)).mean()
        return base_loss, corr_loss

    def forward(self, x, labels, *, t=None, noise=None, target_velocity=None, return_losses=False):
        """Train on clean x, or on noisy x with an explicit distillation target."""
        if self.recipe.mode == 'distill':
            if target_velocity is None or t is None or noise is not None:
                raise ValueError('Distillation requires noisy inputs, t and target_velocity')
            t = t.reshape(-1, 1, 1, 1)
            prediction = self.net(x, t.flatten(), labels)
            velocity = self.space.to_velocity(x, t, prediction, self.prediction)
            base = (velocity.float() - target_velocity.detach().float()).square().mean()
            correction = base.new_zeros(())
        else:
            if target_velocity is not None:
                raise ValueError('target_velocity requires training.mode=distill')
            base, correction = self.losses(x, labels, t=t, noise=noise)
        if return_losses:
            return base, correction
        return self.recipe.base_loss_weight * base + self.recipe.correction_loss_weight * correction

    def velocity(self, z, t, labels, *, corrections=9,
                 cfg=1.0, cfg_interval=(0.1, 1.0)):
        """Convert native prediction to velocity, then apply CFG or correction."""
        validate_guidance(corrections, cfg)
        pred = self.net(z, t.flatten(), labels)
        v = self.space.to_velocity(z, t, pred, self.prediction)
        low, high = cfg_interval
        if cfg != 1.0:
            null = torch.full_like(labels, self.num_classes)
            v_null = self.space.to_velocity(z, t, self.net(z, t.flatten(), null), self.prediction)
            mask = (t < high) & ((low == 0) | (t > low))
            scale = torch.where(mask, cfg, 1.0)
            v = v_null + scale * (v - v_null)
        if corrections:
            v = self.correct(z, t, labels, v, steps=corrections)
        return v

    @torch.no_grad()
    def sample(self, labels, **kwargs):
        return sample(self, labels, **kwargs)
