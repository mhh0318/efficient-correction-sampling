"""SiT linear-path Euler SDE, sigma diffusion and terminal Mean (licenses/SiT.txt).

Uses SiT's CPU FP32 grid/noise draws and three-channel CFG convention.
"""
import torch
from tqdm import tqdm
from src.utils.config import validate_guidance


@torch.no_grad()
def sample(model, noise, labels, *, steps, corrections, correction_interval, cfg, cfg_interval, progress=None):
    validate_guidance(corrections, cfg)
    times = torch.linspace(0., .96, steps, dtype=torch.float32)
    dt = times[1] - times[0]
    guided = cfg > 1
    z = torch.cat([noise, noise]) if guided else noise
    y = torch.cat([labels, torch.full_like(labels, model.num_classes)]) if guided else labels

    def drift(x, time):
        t = (torch.ones(len(x)).to(x) * time).view(-1, 1, 1, 1)
        # Intervals are noise fraction. SiT's full band includes the endpoints.
        r = 1-float(t[0, 0, 0, 0])
        k = corrections if correction_interval[0]-1e-9 <= r <= correction_interval[1]+1e-9 else 0
        state = torch.cat([x[:len(labels)], x[:len(labels)]]) if guided else x
        v = model.velocity(state, t, y, corrections=k, cfg=1.)
        if guided:
            cond, null = v[:, :3].chunk(2)
            scale = cfg if cfg_interval[0]-1e-9 <= r <= cfg_interval[1]+1e-9 else 1.
            first = null + scale*(cond-null)
            v = torch.cat([torch.cat([first, first]), v[:, 3:]], dim=1)
        score = (t*v-x) / ((1-t)**2 + t*(1-t))
        return v + (1-t)*score, 1-t

    with tqdm(total=steps, desc=progress, unit='step', dynamic_ncols=True, disable=progress is None) as bar:
        for t in times[:-1]:
            # Do not replace with randn_like: original SiT draws on the CPU.
            dw = torch.randn(z.size()).to(z) * torch.sqrt(dt)
            d, diffusion = drift(z, t)
            z = z + d*dt + torch.sqrt(2*diffusion)*dw
            bar.update()
        d, _ = drift(z, times[-1])
        z = z + d*.04
        bar.update()
    return z[:len(labels)]
