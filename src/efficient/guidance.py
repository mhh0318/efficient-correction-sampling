"""Training-free ways to spend a trained corrector at sampling time.

With gate gamma, base velocity v and corrector proposal C(x0(v)) (all native to the released gated pairs):
  paper   K corrections at the trained step      v_k = v_{k-1} + gamma (C(x0(v_{k-1})) - v_{k-1})
  extrap  K corrections, total move scaled by w   v' = v + w (v_K - v)                 (K = 1: one corrector call)
  relax   K corrections, gate scaled by s         v_k = v_{k-1} + s gamma (C(x0(v_{k-1})) - v_{k-1})
relax with s = 1 and extrap with w = 1 are the paper's sampler (src/tests/test_efficient.py).
"""
from dataclasses import dataclass

from src.models.correction import refine


@dataclass(frozen=True)
class Method:
    name: str = 'paper'
    k: int = 5
    scale: float = 1.

    def __post_init__(self):
        if self.name not in ('base', 'paper', 'extrap', 'relax'):
            raise ValueError('method must be base, paper, extrap or relax')
        if self.name != 'base' and self.k < 1:
            raise ValueError('k must be positive')


class CorrectorGuidance:
    """The model interface the samplers call (velocity, num_classes). The samplers pass corrections > 0 only inside
    the correction interval; outside it the plain denoiser velocity is returned for every method."""

    def __init__(self, model, method):
        if method.name == 'relax' and getattr(model, 'output_gate', None) is None:
            raise ValueError('relax scales the learned gate; this checkpoint has none')
        self.model, self.method, self.num_classes = model, method, model.num_classes

    def velocity(self, z, t, labels, *, corrections=0, cfg=1., cfg_interval=(0., 1.)):
        v = self.model.velocity(z, t, labels, corrections=0, cfg=cfg, cfg_interval=cfg_interval)
        m = self.method
        if not corrections or m.name == 'base':
            return v
        if m.name == 'paper':
            return self.model.correct(z, t, labels, v, steps=m.k)
        if m.name == 'extrap':
            return v + m.scale * (self.model.correct(z, t, labels, v, steps=m.k) - v)
        return self.relaxed(z, t, labels, v)

    def relaxed(self, z, t, labels, v):
        model = self.model
        return refine(model.corr_net, z, t, labels, v, space=model.space, spec=model.correction, steps=self.method.k,
                      gate=model.output_gate * self.method.scale)
