"""Official SiT/REPA base weights, cached locally and converted on load."""
import os
from pathlib import Path

import torch
from safetensors.torch import load_file


PRETRAINED = {
    'sit': dict(repo_id='sairights/sit-xl-2-256x256-sde-cfg',
                filename='pretrained_models/SiT-XL-2-256x256.safetensors',
                revision='5a15dccbe5710eb5bc9a0b0f71e6fb3deb31e24e'),
    'sit-repa': dict(url='https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&dl=0'),
}


def resolve_pretrained(value, recipe):
    name = str(value)
    if name not in PRETRAINED:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'Pretrained weights not found: {path}')
        return path
    model = recipe['model']
    direction = 1 if name == 'sit' else 0
    if (model['model'] != 'SiT-XL/2' or model['image_size'] != 256
            or model['prediction'] != 'v' or model['data_time'] != direction or model['t_eps'] != 0):
        raise ValueError(f'--init-base {name} requires its SiT-XL/2 velocity-prediction config at resolution 256')
    root = Path(os.environ.get('CORRECTION_CHECKPOINT_DIR', 'checkpoints')).expanduser()
    directory = root / 'pretrained'
    legacy = directory / f'{name}.pt'
    if legacy.is_file():
        return legacy
    source = PRETRAINED[name]
    path = directory / name / source['filename'] if 'repo_id' in source else legacy
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {name} base weights to {path}', flush=True)
        if 'repo_id' in source:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(**source, local_dir=directory / name))
        else:
            torch.hub.download_url_to_file(source['url'], str(path), progress=True)
    return path


def prediction_head(state, channels=4):
    """Discard the unused variance channels within each patch pixel."""
    state = dict(state)
    p = state['x_embedder.proj.weight'].shape[-1]
    rows = state['final_layer.linear.weight'].shape[0]
    if rows == p*p*channels:
        return state
    if rows != p*p*channels*2:
        raise ValueError('Unexpected SiT output head shape')
    for name in ('weight', 'bias'):
        key = f'final_layer.linear.{name}'
        value = state[key]
        state[key] = value.reshape(p, p, 2*channels, *value.shape[1:])[:, :, :channels].reshape(p*p*channels, *value.shape[1:]).contiguous()
    return state


def load_pretrained_state(path, family, key=None):
    path = Path(path)
    state = (load_file(str(path)) if path.suffix == '.safetensors' else
             torch.load(path, map_location='cpu', weights_only=True, mmap=True))
    key = key or ('ema' if 'ema' in state else None)
    if key:
        state = state[key]
    removed = []
    if family == 'repa':
        removed = sorted(k for k in state if k.startswith('projectors.'))
        state = {k: v for k, v in state.items() if k not in removed}
    else:
        state = prediction_head(state)
    return state, dict(selected_weights=key, family=family, unused_projector_tensors=removed)
