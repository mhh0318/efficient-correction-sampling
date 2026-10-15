"""REPA linear Euler-Maruyama with terminal mean update (see licenses/REPA.txt).

Adapted from REPA samplers.py; FP64 integration, native network input dtype,
and GPU path-noise draws follow the original sampler.
"""
import torch
from tqdm import tqdm
from src.utils.config import validate_guidance


@torch.no_grad()
def sample(model, noise, labels, *, steps, corrections, correction_interval, cfg, cfg_interval, progress=None):
    validate_guidance(corrections, cfg)
    times = torch.linspace(1., .04, steps, dtype=torch.float64)
    times = torch.cat([times, torch.tensor([0.], dtype=torch.float64)])
    z = noise.to(torch.float64)

    def drift(x, time):
        guided = cfg > 1 and cfg_interval[0] <= time <= cfg_interval[1]
        state = torch.cat([x, x]) if guided else x
        y = torch.cat([labels, torch.full_like(labels, model.num_classes)]) if guided else labels
        t = (torch.ones(len(state)).to(device=x.device, dtype=torch.float64)*time).view(-1, 1, 1, 1)
        native_t = t.to(noise.dtype)
        tau = float(native_t[0, 0, 0, 0])
        k = corrections if correction_interval[0]-1e-9 <= tau <= correction_interval[1]+1e-9 else 0
        v = model.velocity(state.to(noise.dtype), native_t, y, corrections=k, cfg=1.).to(torch.float64)
        # Keep the source operation order, including the unsimplified score.
        alpha, d_alpha = 1-t, torch.ones_like(state)*-1
        sigma, d_sigma = t, torch.ones_like(state)
        ratio = alpha/d_alpha
        var = sigma**2 - ratio*d_sigma*sigma
        score = (ratio*v-state)/var
        d = v - .5*(2*time)*score
        if guided:
            cond, null = d.chunk(2)
            d = null + cfg*(cond-null)
        return d

    with tqdm(total=steps, desc=progress, unit='step', dynamic_ncols=True, disable=progress is None) as bar:
        for t, next_t in zip(times[:-2], times[1:-1]):
            dt = next_t-t
            eps = torch.randn_like(z).to(z.device)
            deps = eps*torch.sqrt(torch.abs(dt))
            z = z + drift(z, t)*dt + torch.sqrt(2*t)*deps
            bar.update()
        z = z + (times[-1]-times[-2])*drift(z, times[-2])
        bar.update()
    return z
