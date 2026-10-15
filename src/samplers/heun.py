"""Shared Heun sampler with a final Euler step; state may be pixels or latents."""
from fractions import Fraction
import torch
from tqdm import tqdm
from src.utils.config import validate_guidance


@torch.no_grad()
def sample(model, labels, *, noise=None, steps=45, corrections=9,
           correction_interval=(.8, 1.), cfg=1., cfg_interval=(.1, 1.), method='heun', progress=None):
    validate_guidance(corrections, cfg)
    if steps < 1 or corrections < 0:
        raise ValueError('steps must be positive and corrections nonnegative')
    for low, high in (correction_interval, cfg_interval):
        if not 0 <= low < high <= 1:
            raise ValueError('intervals must satisfy 0 <= low < high <= 1')
    if method not in ('heun', 'sit_sde', 'repa_sde'):
        raise ValueError('Unknown sampler method')
    if method != 'heun' and (steps < 2 or cfg < 1 or model.prediction != 'v'
                             or model.space.t_eps != 0
                             or model.space.data_time != (1 if method == 'sit_sde' else 0)):
        raise ValueError('Native SDE requires matching velocity path, steps >= 2 and cfg >= 1')
    # eps -> v cannot be evaluated at pure noise. Avoid a silent division by zero.
    if model.prediction == 'eps' or (corrections and model.correction.output == 'eps' and correction_interval[1] == 1):
        raise ValueError('eps prediction at pure noise needs an interior-time sampler')
    z = (model.noise_scale * torch.randn(len(labels), model.net.in_channels, model.image_size, model.image_size,
                                        device=labels.device) if noise is None else noise.clone())
    if method != 'heun':
        from .sit import sample as sit_sde
        from .repa import sample as repa_sde
        solver = sit_sde if method == 'sit_sde' else repa_sde
        return solver(model, z, labels, steps=steps, corrections=corrections,
                      correction_interval=correction_interval, cfg=cfg, cfg_interval=cfg_interval,
                      progress=progress)
    data_time = model.space.data_time
    times = torch.linspace(1-data_time, data_time, steps+1, device=z.device).view(-1, 1, 1, 1, 1)
    times = times.expand(-1, len(labels), -1, -1, -1)
    # Correction interval is always noise fraction, independent of native time.
    low, high = map(lambda x: Fraction(str(x)), correction_interval)
    active = [low < Fraction(steps-i, steps) <= high for i in range(steps+1)]

    def velocity(state, i):
        return model.velocity(state, times[i], labels, corrections=corrections if active[i] else 0,
                              cfg=cfg, cfg_interval=cfg_interval)

    for i in tqdm(range(steps), desc=progress, unit='step', dynamic_ncols=True, disable=progress is None):
        dt = times[i+1] - times[i]
        v = velocity(z, i)
        euler = z + dt*v
        z = z + dt*(.5*(v+velocity(euler, i+1))) if i < steps-1 else euler
    return z
