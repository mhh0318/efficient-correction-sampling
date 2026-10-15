"""Correction spaces and optional mixing, independent of the network architecture."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Correction:
    input: str = "x"
    output: str = "x"
    combine: str = "direct"
    gate_init: float = 0.0
    gate_shape: str = "channel"
    gate_mix: str = "difference"

    def __post_init__(self):
        if self.input not in ("x", "v", "eps") or self.output not in ("x", "v", "eps"):
            raise ValueError("Correction spaces must be x, v or eps")
        if self.combine not in ("direct", "gated", "residual", "base_anchor"):
            raise ValueError("Correction combine must be direct, gated, residual or base_anchor")
        if self.gate_shape not in ("channel", "scalar") or self.gate_mix not in ("difference", "blend"):
            raise ValueError("gate_shape must be channel/scalar; gate_mix must be difference/blend")


def refine(net, z, t, cond, velocity, *, space, spec=Correction(), steps=1,
           gate=None, detach_input=False, detach_skip=False, proposal=None):
    """The network returns a candidate in spec.output; z/t stay fixed for all K.

    `proposal` preserves native training proposals without an extra conversion.
    Input and skip have independent gradient boundaries at every iteration.
    """
    if steps < 0: raise ValueError("steps must be nonnegative")
    if spec.combine == "gated" and gate is None: raise ValueError("gated correction requires a gate")
    if spec.combine != "gated" and gate is not None: raise ValueError("Only gated correction has a gate")
    if steps == 0: return velocity
    if spec.combine == 'base_anchor' and spec.output != 'v':
        # Keep the denoiser's complementary component fixed across corrections.
        other = 'eps' if spec.output == 'x' else 'x'
        anchor = space.from_velocity(z, t, velocity, other)
        anchor = anchor.detach() if detach_skip else anchor
    value = space.from_velocity(z, t, velocity, spec.input) if proposal is None else proposal
    for i in range(steps):
        output = net(value.detach() if detach_input else value, t.flatten(), cond)
        if spec.combine == 'base_anchor':
            if spec.output != 'v':
                clean_minus_noise = output-anchor if spec.output == 'x' else anchor-output
                velocity = clean_minus_noise if space.data_time == 1 else -clean_minus_noise
            else:
                velocity = output
            if i + 1 < steps:
                value = space.from_velocity(z, t, velocity, spec.input)
            continue
        if spec.combine != "direct":
            # Residual heads emit displacements: combine before affine conversion.
            skip = value if spec.input == spec.output else space.from_velocity(z, t, velocity, spec.output)
            skip = skip.detach() if detach_skip else skip
            if spec.combine == "residual":
                output = skip + output
            else:
                g = gate.to(output).view(1, -1, *([1] * (output.ndim-2)))
                output = ((1-g)*skip + g*output if spec.gate_mix == "blend"
                          else skip + g * (output-skip))
        if spec.input == spec.output:
            # Preserve native proposals and same-space iteration without round trips.
            value = output
        else:
            velocity = space.to_velocity(z, t, output, spec.output)
            if i + 1 < steps:
                value = space.from_velocity(z, t, velocity, spec.input)
    if spec.combine == 'base_anchor':
        return velocity
    return space.to_velocity(z, t, value, spec.output) if spec.input == spec.output else velocity
