"""Linear interpolation and x/v/eps conversions. Time direction is explicit."""


class LinearPath:
    def __init__(self, data_time=1, t_eps=0.):
        self.data_time, self.t_eps = data_time, t_eps

    def interpolate(self, x, noise, t):
        return t*x + (1-t)*noise if self.data_time == 1 else (1-t)*x + t*noise

    def target(self, x, noise, z, t):
        # JiT weights the endpoint region through its clamped x-to-v conversion.
        if self.t_eps:
            return self.to_velocity(z, t, x, 'x')
        return x-noise if self.data_time == 1 else noise-x

    def from_velocity(self, z, t, v, space):
        if space == 'v': return v
        if self.data_time == 1:
            if space == 'x': return z + (1-t).clamp_min(self.t_eps)*v
            if space == 'eps': return z - t*v
        else:
            if space == 'x': return z - t.clamp_min(self.t_eps)*v
            if space == 'eps': return z + (1-t)*v
        raise ValueError(space)

    def to_velocity(self, z, t, value, space):
        if space == 'v': return value
        if self.data_time == 1:
            delta, denominator = (value-z, (1-t).clamp_min(self.t_eps)) if space == 'x' else (z-value, t)
        else:
            delta, denominator = (z-value, t.clamp_min(self.t_eps)) if space == 'x' else (value-z, 1-t)
        if space not in ('x', 'eps'): raise ValueError(space)
        if (space == 'eps' or not self.t_eps) and not bool((denominator > 0).all()):
            raise ValueError(f'{space} conversion is undefined at this path endpoint')
        return delta / denominator
