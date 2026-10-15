"""One resolved recipe shared by training, inference and exported checkpoints."""
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import yaml

from src.models.correction import Correction

@dataclass(frozen=True)
class Distillation:
    teacher_corrections: int = 2

    def __post_init__(self):
        if type(self.teacher_corrections) is not int or self.teacher_corrections < 1:
            raise ValueError('teacher_corrections must be a positive integer')


@dataclass(frozen=True)
class Training:
    mode: str = "joint"
    seed: int = 0
    horizontal_flip: bool = True
    loss_space: str = "v"
    detach_input: bool = True
    detach_skip: bool = True
    deterministic: bool = False
    correction_steps: int = 1
    base_loss_weight: float = 1.0
    correction_loss_weight: float = 1.0
    steps: int = 100000
    batch_size: int = 32
    accumulation: int = 4
    lr: float = 5e-5
    betas: tuple[float, float] = (.9, .95)
    ema: tuple[float, ...] = (.9999, .9996)
    precision: str = 'bf16'
    gradient_precision: str = 'fp32'
    gradient_bucket_mb: int = 25
    grad_clip: float = 0.0
    warmup_epochs: float = 5.0
    warmup_steps: int = 0
    save_every: int = 1000
    export_every: int = 1000

    def __post_init__(self):
        if self.mode not in ('joint', 'base', 'corrector', 'distill'):
            raise ValueError('Training mode must be joint/base/corrector/distill')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Training seed must be a nonnegative integer')
        if self.loss_space != "v":
            raise ValueError("Only velocity MSE is implemented")
        if any(type(getattr(self, k)) is not bool for k in ('detach_input', 'detach_skip', 'deterministic', 'horizontal_flip')):
            raise ValueError("detach_input, detach_skip, deterministic and horizontal_flip must be booleans")
        for name in ('correction_steps', 'steps', 'batch_size', 'accumulation', 'save_every', 'export_every', 'gradient_bucket_mb'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.base_loss_weight < 0 or self.correction_loss_weight < 0:
            raise ValueError('Loss weights must be nonnegative')
        if self.mode in ('base', 'distill'):
            if self.base_loss_weight <= 0 or self.correction_loss_weight != 0:
                raise ValueError('Base training requires base > 0 and correction = 0')
        elif self.correction_loss_weight <= 0 or (self.mode == 'corrector' and self.base_loss_weight != 0):
            raise ValueError('Correction training requires correction > 0; frozen-base training requires base = 0')
        if self.lr <= 0 or self.warmup_epochs < 0:
            raise ValueError("lr must be positive and warmup_epochs nonnegative")
        if self.grad_clip < 0:
            raise ValueError('grad_clip must be nonnegative')
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError('warmup_steps must be a nonnegative integer')
        if self.warmup_steps and self.warmup_epochs:
            raise ValueError('Choose warmup_steps or warmup_epochs, not both')
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError('betas must contain two values in [0, 1)')
        if not self.ema or any(not 0 <= d < 1 for d in self.ema):
            raise ValueError('ema must contain at least one decay in [0, 1)')
        if self.precision not in ('bf16', 'fp32'):
            raise ValueError('precision must be bf16/fp32')
        if self.gradient_precision not in ('bf16', 'fp32'):
            raise ValueError('gradient_precision must be bf16/fp32')


MODEL = dict(model="JiT-L/16", image_size=256, num_classes=1000,
             t_eps=0.05, noise_scale=1.0, p_mean=-0.8, p_std=0.8, label_drop=0.1,
             prediction="x", data_time=1, time_sampling="logit_normal")
SAMPLING = dict(method='heun', precision='bf16', steps=45, corrections=9,
                correction_interval=[0.8, 1.0], cfg=1.0, cfg_interval=[0.1, 1.0])


def validate_guidance(corrections, cfg):
    if corrections > 0 and cfg != 1.0:
        raise ValueError(
            'CFG with correction is not supported by this ImageNet implementation. '
            'The combination is possible, but was not used in our ImageNet experiments; '
            'support is planned for text-to-image. '
            'Use cfg=1 (--cfg 1) or corrections=0 (--corrections_per_step 0).'
        )


def resolve(value):
    """Fill defaults and reject unsupported fields before model allocation."""
    unknown = value.keys() - {'model', 'codec', 'correction', 'training', 'sampling', 'distillation'}
    if unknown:
        raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
    model = value.get('model', {})
    if isinstance(model, str):
        model = dict(model=model)
    defaults = MODEL
    latent = model.get('model', MODEL['model']).startswith('SiT-')
    if latent:
        defaults = MODEL | dict(prediction='v', t_eps=0., time_sampling='uniform')
    from src.datasets.codec import CodecConfig
    codec = CodecConfig(**({'name': 'vae'} if latent else {}) | value.get('codec', {}))
    sampling = value.get('sampling', {})
    for name, supplied, allowed in [('model', model, defaults), ('sampling', sampling, SAMPLING)]:
        if supplied.keys() - allowed.keys():
            raise ValueError(f"Unknown {name} fields: {sorted(supplied.keys() - allowed.keys())}")
    result = dict(model=defaults | model, codec=asdict(codec),
                  correction=asdict(Correction(**value.get('correction', {}))),
                  training=asdict(Training(**value.get('training', {}))),
                  sampling=SAMPLING | sampling)
    if result['training']['mode'] == 'distill':
        result['distillation'] = asdict(Distillation(**value.get('distillation', {})))
        if result['model']['time_sampling'] != 'uniform' or result['model']['label_drop'] != 0:
            raise ValueError('Distillation uses uniform times and label_drop=0')
    elif 'distillation' in value:
        raise ValueError('Distillation settings require training.mode=distill')
    for name in ('betas', 'ema'):
        result['training'][name] = list(result['training'][name])
    m = result['model']
    if m['prediction'] not in ('x', 'v', 'eps') or m['data_time'] not in (0, 1):
        raise ValueError('prediction must be x/v/eps and data_time must be 0/1')
    if m['time_sampling'] not in ('uniform', 'logit_normal') or not 0 <= m['t_eps'] < 1:
        raise ValueError('Invalid time sampling or t_eps')
    if m['image_size'] < 1 or m['image_size'] % codec.downsample:
        raise ValueError('Image size must be divisible by codec.downsample')
    s = result['sampling']
    validate_guidance(s['corrections'], s['cfg'])
    if result['training']['mode'] in ('base', 'distill') and s['corrections'] != 0:
        raise ValueError('Base-only recipes require sampling.corrections=0')
    if s['method'] not in ('heun', 'sit_sde', 'repa_sde') or s['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Unknown sampling method or precision')
    if s['method'] != 'heun':
        direction = 1 if s['method'] == 'sit_sde' else 0
        if not latent or m['prediction'] != 'v' or m['data_time'] != direction or m['t_eps'] != 0:
            raise ValueError('Native latent SDE requires matching SiT velocity path and t_eps=0')
        if s['steps'] < 2 or s['cfg'] < 1:
            raise ValueError('Native latent SDE requires steps >= 2 and cfg >= 1')
    if type(s['steps']) is not int or s['steps'] < 1 or type(s['corrections']) is not int or s['corrections'] < 0:
        raise ValueError('Sampling steps must be positive and corrections nonnegative integers')
    for name in ('correction_interval', 'cfg_interval'):
        low, high = s[name]
        if not 0 <= low < high <= 1:
            raise ValueError(f'{name} must satisfy 0 <= low < high <= 1')
    return result


def _read_config_file(path):
    path = Path(path)
    text = path.read_text()
    return json.loads(text) if path.suffix == '.json' else yaml.safe_load(text)


def _sampling_fields(value):
    value = dict(value)
    for public, internal in [('denoising_steps', 'steps'), ('corrections_per_step', 'corrections')]:
        if public in value:
            if internal in value:
                raise ValueError(f'Use only {public}, not both {public} and {internal}')
            value[internal] = value.pop(public)
    return value


def _convert_intervals(value, method, data_time):
    """Convert between public [start t, end t] and native bounds (self-inverse)."""
    value = dict(value)
    for name in ('correction_interval', 'cfg_interval'):
        if name not in value:
            continue
        bounds = value[name]
        if name == 'cfg_interval' and method == 'heun' and data_time == 1:
            value[name] = [float(1-Fraction(str(t))) for t in bounds]
        else:
            value[name] = list(reversed(bounds))
    return value


def _native_sampling(value, method, data_time):
    """Convert public intervals in sampling order: noise (1) toward data (0)."""
    for name in ('correction_interval', 'cfg_interval'):
        if name in value:
            start, end = value[name]
            if not 0 <= end < start <= 1:
                raise ValueError(f'{name} must satisfy 0 <= end t < start t <= 1; use [start t, end t]')
    return _convert_intervals(value, method, data_time)


def read_config(path):
    value = _read_config_file(path)
    if Path(path).suffix != '.json' and 'sampling' in value:
        sampling = _sampling_fields(value['sampling'])
        model = value.get('model', {})
        data_time = model.get('data_time', 1) if isinstance(model, dict) else 1
        value['sampling'] = _native_sampling(sampling, sampling.get('method', 'heun'), data_time)
    return resolve(value)


def read_sampling_config(path):
    """Read an explicit checkpoint, sampling recipe, and generation cohort."""
    value = _read_config_file(path)
    if not isinstance(value, dict) or set(value) != {'checkpoint', 'sampling', 'generation'}:
        raise ValueError('Sampling config requires checkpoint, sampling, and generation')
    if not isinstance(value['checkpoint'], str) or not value['checkpoint']:
        raise ValueError('Sampling config requires a checkpoint name')
    if isinstance(value['sampling'], dict):
        value['sampling'] = _sampling_fields(value['sampling'])
        if Path(path).suffix == '.json':
            sampling = value['sampling']
            value['sampling'] = _convert_intervals(sampling, sampling.get('method', 'heun'), 1)
    if not isinstance(value['sampling'], dict) or set(value['sampling']) != set(SAMPLING):
        raise ValueError(f'Sampling config must explicitly set {sorted(SAMPLING)}')
    generation = value['generation']
    allowed = {'runner', 'samples', 'batch_size', 'seed', 'classes', 'unconditional'}
    if not isinstance(generation, dict) or generation.keys() - allowed:
        raise ValueError('Unknown generation config fields')
    if generation.get('runner') not in ('sample', 'jit', 'sit', 'repa'):
        raise ValueError('Generation runner must be sample/jit/sit/repa')
    required = ({'runner', 'seed'} | (set() if generation.get('unconditional') else {'classes'})
                if generation['runner'] == 'sample' else
                {'runner', 'samples', 'batch_size'})
    optional = {'samples', 'batch_size', 'unconditional', 'classes'} if generation['runner'] == 'sample' else set()
    if not required <= generation.keys() or generation.keys() - required - optional:
        raise ValueError(f'Generation config for {generation["runner"]} requires {sorted(required)}')
    if 'unconditional' in generation and type(generation['unconditional']) is not bool:
        raise ValueError('generation.unconditional must be true or false')
    for field in ('samples', 'batch_size'):
        if field in generation and (type(generation[field]) is not int or generation[field] < 1):
            raise ValueError(f'generation.{field} must be a positive integer')
    if 'seed' in generation and (type(generation['seed']) is not int or generation['seed'] < 0):
        raise ValueError('generation.seed must be a nonnegative integer')
    if 'classes' in generation and (not isinstance(generation['classes'], list) or not generation['classes']
                                   or any(type(y) is not int or y < 0 for y in generation['classes'])):
        raise ValueError('generation.classes must be a nonempty list of nonnegative integers')
    return value


def sampling_recipe(config, preset=None):
    """Apply sampling settings without changing the trained model."""
    recipe = dict(config['sampling'])
    if preset is not None:
        supplied = _sampling_fields(preset['sampling'])
        recipe.update(_native_sampling(supplied, supplied.get('method', recipe.get('method', 'heun')),
                                       config['model'].get('data_time', 1)))
    resolved = resolve({k: config[k] for k in ('model', 'codec', 'correction', 'training', 'distillation') if k in config}
                       | dict(sampling=recipe))
    return resolved['sampling']


def build_model(config, **kwargs):
    from src.models.denoiser import Denoiser
    model = dict(config['model'])
    from src.datasets.codec import CodecConfig
    codec = CodecConfig(**config['codec'])
    model['image_size'] //= codec.downsample
    return Denoiser(**model, in_channels=codec.channels,
                    correction=Correction(**config['correction']),
                    training=Training(**config['training']), **kwargs)
