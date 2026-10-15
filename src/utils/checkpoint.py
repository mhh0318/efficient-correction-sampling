"""Save and load denoisers and denoiser-corrector pairs with their configurations."""
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .config import build_model, resolve

FORMAT = 'correction-training/2'


def read_checkpoint(directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    if config.get('format') != FORMAT:
        raise ValueError('Expected a correction-training checkpoint bundle')
    recipe = resolve({k: config[k] for k in ('model', 'codec', 'correction', 'training', 'sampling', 'distillation') if k in config})
    config.update(recipe)
    config.setdefault('kind', 'pair')
    if config['kind'] not in ('base_init', 'pair'):
        raise ValueError('Checkpoint kind must be base_init or pair')
    weights = directory / 'model.safetensors'
    return load_file(str(weights)), config


def save_checkpoint(directory, state, recipe, *, source, kind='pair'):
    directory = Path(directory)
    recipe = resolve(recipe)
    if kind not in ('base_init', 'pair'):
        raise ValueError('Checkpoint kind must be base_init or pair')
    state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    # An interrupted run can replay exports newer than its last restart checkpoint.
    if directory.exists():
        previous, config = read_checkpoint(directory)
        if (any(config[k] != v for k, v in recipe.items()) or config['source'] != source or config['kind'] != kind
                or previous.keys() != state.keys()
                or any(not torch.equal(previous[k], v) for k, v in state.items())):
            raise ValueError(f'Conflicting checkpoint already exists: {directory}')
        return config
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix='.export-') as temporary:
        staging = Path(temporary) / 'checkpoint'
        staging.mkdir()
        weights = staging / 'model.safetensors'
        save_file(state, str(weights))
        config = dict(format=FORMAT, kind=kind, **recipe, source=source)
        (staging / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        staging.rename(directory)
    return config


def load_model(directory, device='cuda'):
    state, config = read_checkpoint(directory)
    with torch.device('meta'):
        model = build_model(config)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device).eval(), config


def initialize_model(model, recipe, directory, *, mode):
    """Start a new optimizer run from a base initialization or a trained pair."""
    if mode not in ('base', 'pair'):
        raise ValueError('Initialization mode must be base or pair')
    directory = Path(directory)
    if mode == 'base' and directory.is_file():
        from .pretrained import load_pretrained_state
        m = recipe['model']
        if not m['model'].startswith('SiT-') or m['prediction'] != 'v' or m['t_eps'] != 0:
            raise ValueError('Official SiT/REPA base files require a SiT velocity-prediction config with t_eps=0')
        state, source = load_pretrained_state(directory, 'sit' if m['data_time'] == 1 else 'repa')
        state = {'net.'+k: v for k, v in state.items()}
        config = dict(kind='base_init', **recipe, source=source)
    else:
        state, config = read_checkpoint(directory)
    fields = ('model', 'image_size', 'num_classes', 'prediction', 'data_time', 't_eps', 'noise_scale')
    if (any(config['model'][k] != recipe['model'][k] for k in fields)
            or config['codec'] != recipe['codec']):
        raise ValueError('Initialization architecture, prediction, path or codec differs from training')
    if mode == 'base':
        if config['kind'] != 'base_init':
            raise ValueError('Trained pair supplied to --init-base; use --init-pair to preserve its corrector')
        model.net.load_state_dict({k.removeprefix('net.'): v for k, v in state.items()
                                   if k.startswith('net.')}, strict=True)
        model.initialize_corrector()
        if hasattr(model, 'output_gate'):
            with torch.no_grad():
                model.output_gate.fill_(recipe['correction']['gate_init'])
    else:
        if config['kind'] != 'pair':
            raise ValueError('Base initialization supplied to --init-pair; use --init-base')
        if config['correction'] != recipe['correction']:
            raise ValueError('Pair correction recipe differs from training')
        model.load_state_dict(state, strict=True)
    return config
