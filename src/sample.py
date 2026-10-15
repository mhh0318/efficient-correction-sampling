"""Generate images from a canonical pixel or latent checkpoint on one GPU."""
import argparse
import contextlib
from datetime import datetime
from itertools import count
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.download import resolve_checkpoint
from src.utils.config import read_sampling_config, sampling_recipe


def new_run_directory(root):
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    for index in count():
        path = root / (timestamp if index == 0 else f'{timestamp}-{index}')
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="Sampling config (default: official checkpoint's qualitative preset)")
    p.add_argument("--checkpoint", help="Official checkpoint name or local bundle directory")
    p.add_argument("--checkpoint-dir", type=Path,
                   help="Download/cache root (default: CORRECTION_CHECKPOINT_DIR or checkpoints)")
    conditioning = p.add_mutually_exclusive_group()
    conditioning.add_argument("--classes", nargs="+", type=int, help="Class IDs; cycle through these when generating more samples")
    conditioning.add_argument("--unconditional", action=argparse.BooleanOptionalAction, default=None,
                              help="Use the null class embedding without CFG")
    p.add_argument("--samples", type=int, help="Total images (default: sampling config, otherwise one per selected class)")
    p.add_argument("--batch-size", type=int, help="Images per GPU batch (default: 4)")
    p.add_argument("--grid-columns", type=int, default=4, help="Columns in the saved grid (default: 4)")
    p.add_argument("--seed", type=int, help="Random seed (default: sampling config, or 0)")
    p.add_argument("--denoising_steps", "--steps", dest="steps", type=int,
                   help="Denoising steps (default: sampling config)")
    p.add_argument("--corrections_per_step", "--corrections", dest="corrections", type=int,
                   help="K in the paper: correction iterations per active denoising step")
    p.add_argument("--correction-interval", nargs=2, type=float, metavar=("START_T", "END_T"),
                   help="Correction interval: START_T END_T (1=noise, 0=data; e.g. 1.0 0.8)")
    p.add_argument("--cfg", type=float, help="CFG scale; requires --corrections_per_step 0 when not 1")
    p.add_argument("--cfg-interval", nargs=2, type=float, metavar=("START_T", "END_T"),
                   help="CFG interval: START_T END_T (1=noise, 0=data; e.g. 0.8 0.1)")
    p.add_argument("--output", type=Path, help="Parent directory for timestamped runs (default: samples/<checkpoint>)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["fp32", "bf16"], help="Default: checkpoint sampling precision")
    p.add_argument("--compile", action="store_true", help="Compile transformer blocks (optional)")
    args = p.parse_args()
    try:
        if args.config is None and args.checkpoint in ('jit', 'sit', 'sit-repa-50k', 'sit-repa-100k', 'sit-repa-uncond-100k'):
            name = ('sit-repa-uncond' if args.checkpoint == 'sit-repa-uncond-100k' else
                    'sit-repa' if args.checkpoint.startswith('sit-repa') else args.checkpoint)
            args.config = Path(__file__).resolve().parents[1] / f'configs/eval/qualitative/{name}.yml'
        preset = read_sampling_config(args.config) if args.config else None
        generation = preset['generation'] if preset and preset['generation']['runner'] == 'sample' else {}
        args.checkpoint = args.checkpoint or (preset['checkpoint'] if preset else None)
        if not args.checkpoint:
            p.error('Specify --config or --checkpoint')
        if args.classes is not None:
            args.unconditional = False
        elif args.unconditional is None:
            args.unconditional = generation.get('unconditional')
        args.classes = args.classes if args.classes is not None else generation.get('classes', [207, 388, 949, 980])
        args.samples = args.samples if args.samples is not None else generation.get('samples', len(args.classes))
        args.batch_size = args.batch_size if args.batch_size is not None else generation.get('batch_size', 4)
        args.seed = args.seed if args.seed is not None else generation.get('seed', 0)
        if args.samples < 1 or args.batch_size < 1 or args.grid_columns < 1:
            p.error('samples, batch-size and grid-columns must be positive')
        print('Loading checkpoint...', flush=True)
        checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
        metadata = json.loads((checkpoint / 'config.json').read_text())
        if args.unconditional is None:
            args.unconditional = metadata.get('conditioning') == 'unconditional'
        overrides = dict(preset['sampling']) if preset else {}
        for name in ('steps', 'corrections', 'correction_interval', 'cfg', 'cfg_interval', 'precision'):
            if getattr(args, name) is not None:
                overrides[name] = getattr(args, name)
        if args.unconditional:
            overrides['cfg'] = 1.
        print('Loading model...', flush=True)
        recipe = sampling_recipe(metadata, dict(sampling=overrides))
    except (OSError, ValueError, RuntimeError) as error:
        p.error(str(error))
    import numpy as np
    from PIL import Image
    import torch
    from src.utils.checkpoint import load_model
    from src.datasets.codec import build_codec

    model, config = load_model(checkpoint, args.device)
    if torch.device(args.device).type == 'cuda':
        latent = config['model']['model'].startswith('SiT-')
        torch.backends.cuda.matmul.allow_tf32 = latent
        torch.backends.cudnn.allow_tf32 = latent
    if not args.unconditional and any(y < 0 or y >= model.num_classes for y in args.classes):
        p.error(f"classes must be between 0 and {model.num_classes - 1}")
    if args.compile:
        for net in (model.net, model.corr_net):
            if net is None:
                continue
            for block in [*net.blocks, net.final_layer]:
                block.forward = torch.compile(block.forward)
    if config['codec']['name'] == 'vae':
        print('Loading VAE...', flush=True)
    codec = build_codec(config, next(model.parameters()).device).eval()
    precision = recipe.pop('precision')
    if config['kind'] == 'base_init' and recipe['corrections']:
        p.error('This is a base initialization, not a trained pair; use --corrections_per_step 0')
    torch.manual_seed(args.seed)
    class_ids = None if args.unconditional else [args.classes[i % len(args.classes)] for i in range(args.samples)]
    counts = {"base": 0, "correction": 0}
    def counter(name):
        def hook(_module, _inputs, _output):
            counts[name] += len(_inputs[0])
        return hook
    handles = [model.net.register_forward_hook(counter("base"))]
    if model.corr_net is not None:
        handles.append(model.corr_net.register_forward_hook(counter("correction")))
    ctx = (torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16)
           if precision == "bf16" else contextlib.nullcontext())
    images = []
    for start in range(0, args.samples, args.batch_size):
        size = min(args.batch_size, args.samples - start)
        labels = (torch.full((size,), model.num_classes, device=args.device) if args.unconditional else
                  torch.tensor(class_ids[start:start+size], device=args.device))
        progress = (f'Sample {start+1}/{args.samples}' if size == 1 else
                    f'Samples {start+1}-{start+size}/{args.samples}')
        with ctx:
            samples = model.sample(labels, progress=progress, **recipe)
        pixels = codec.decode(samples.float())
        pixels = ((pixels + 1) / 2).detach().cpu().numpy().transpose(0, 2, 3, 1)
        images.extend(np.round(np.clip(pixels * 255, 0, 255)).astype(np.uint8))
    for handle in handles:
        handle.remove()
    args.output = new_run_directory(args.output or Path('samples') / checkpoint.resolve().name)
    columns = min(args.grid_columns, len(images))
    height, width = images[0].shape[:2]
    grid = np.zeros(((len(images)+columns-1)//columns*height, columns*width, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        label = 'unconditional' if args.unconditional else f'class{class_ids[i]}'
        Image.fromarray(image).save(args.output / f"{i:03d}-{label}.png")
        row, column = divmod(i, columns)
        grid[row*height:(row+1)*height, column*width:(column+1)*width] = image
    Image.fromarray(grid).save(args.output / "grid.png")
    record = dict(seed=args.seed, classes=class_ids, unconditional=args.unconditional,
                  samples=args.samples, batch_size=args.batch_size, precision=precision,
                  tf32=torch.backends.cuda.matmul.allow_tf32,
                  compiled=args.compile, sampling=recipe,
                  calls_per_image={name: count // args.samples for name, count in counts.items()},
                  sampling_config=preset,
                  torch=torch.__version__, checkpoint=config.get("source", {}),
                  model=config['model'], correction=config['correction'], codec=config['codec'])
    (args.output / "sampling.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()
