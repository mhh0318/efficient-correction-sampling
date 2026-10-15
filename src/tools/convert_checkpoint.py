"""Convert official JiT weights, or copy an existing checkpoint bundle."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from safetensors.torch import load_file, save_file
from src.utils.checkpoint import read_checkpoint, save_checkpoint
from src.utils.config import read_config, resolve, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, help='Training recipe matching the official JiT model')
    parser.add_argument('--key', default='model', help='State dict to use from an official JiT checkpoint')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    averages = {}
    if args.checkpoint.is_dir():
        state, config = read_checkpoint(args.checkpoint)
        recipe = resolve({k: config[k] for k in
                          ('model', 'codec', 'correction', 'training', 'sampling', 'distillation') if k in config})
        kind = config['kind']
        source = dict(original=config.get('source', {}))
        averages = {i: load_file(str(args.checkpoint / f'base_ema{i}.safetensors'))
                    for i in (1, 2) if (args.checkpoint / f'base_ema{i}.safetensors').exists()}
    else:
        recipe = read_config(args.config or Path(__file__).resolve().parents[2] / 'configs/train/jit.yml')
        # Official JiT checkpoints may include their argparse configuration.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
        base = {k.removeprefix('net.'): v for k, v in checkpoint[args.key].items()}
        model = build_model(recipe)
        model.net.load_state_dict(base, strict=True)
        model.initialize_corrector()
        state, kind = model.state_dict(), 'base_init'
        source = dict(selected_weights=args.key, purpose='official JiT base initialization')
        averages = {i: {k.removeprefix('net.'): v for k, v in checkpoint[f'model_ema{i}'].items()}
                    for i in (1, 2) if f'model_ema{i}' in checkpoint}
    save_checkpoint(args.output, state, recipe, source=source, kind=kind)
    for i, weights in averages.items():
        save_file({k: v.contiguous() for k, v in weights.items()}, str(args.output / f'base_ema{i}.safetensors'))
    print(args.output)


if __name__ == '__main__':
    main()
