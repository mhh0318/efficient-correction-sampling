"""Base, joint, or frozen-base corrector training with shared I/O and EMA."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import signal
import time
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.imagenet import ImageNet, LatentMoments
from src.datasets.codec import build_codec, CodecConfig, sample_posterior
from src.utils.config import read_config, build_model, resolve
from src.utils.checkpoint import initialize_model, load_model, save_checkpoint
from src.utils.pretrained import PRETRAINED, resolve_pretrained
from src.models.distillation import initialize_student, teacher_targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    start = p.add_mutually_exclusive_group()
    start.add_argument('--init-base', help='Official pretrained base (sit, sit-repa), local SiT/REPA file, or base bundle')
    start.add_argument('--init-pair', help='Official checkpoint name or local trained pair; start a new optimizer')
    start.add_argument('--resume', action='store_true', help='Restore checkpoint-last.pt in --output, including optimizer, EMA and RNG')
    p.add_argument('--teacher', help='Distillation teacher: official checkpoint name or local pair bundle')
    p.add_argument('--teacher-corrections', type=int, help='Override the teacher K in the distillation recipe')
    p.add_argument('--data', type=Path, required=True, help='ImageNet ImageFolder, indexed tar, or posterior cache')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--global-batch-size', type=int, help='Override the effective batch across all GPUs and accumulation')
    p.add_argument('--steps', type=int, help='Override the number of training updates')
    p.add_argument('--save-every', type=int, help='Override the restart checkpoint interval')
    p.add_argument('--export-every', type=int, help='Override the EMA export interval')
    p.add_argument('--compile', action='store_true')
    a = p.parse_args()
    last = a.output / 'checkpoint-last.pt'
    manifest = a.output / 'training.json'
    previous = None
    if a.resume:
        if not last.is_file() or not manifest.is_file():
            p.error('--resume requires checkpoint-last.pt and training.json in --output')
        previous = json.loads(manifest.read_text())
    elif a.output.exists() and any(a.output.iterdir()):
        p.error('Output is not empty; use --resume for an existing run or choose a new output directory')
    initial_dir = a.init_base or a.init_pair
    if a.init_base and a.init_base not in PRETRAINED:
        path = Path(a.init_base).expanduser()
        if not (path.is_file() or all((path/name).is_file() for name in ('config.json', 'model.safetensors'))):
            p.error('Initialization requires sit, sit-repa, a local weight file, or a base bundle')
    initialization_mode = previous['initialization_mode'] if previous else ('base' if a.init_base else 'pair' if a.init_pair else 'scratch')
    recipe = read_config(a.config)
    for name in ('steps', 'save_every', 'export_every'):
        if getattr(a, name) is not None:
            recipe['training'][name] = getattr(a, name)
    recipe = resolve(recipe)
    h = recipe['training']
    distilling = h['mode'] == 'distill'
    if a.teacher_corrections is not None:
        if not distilling:
            p.error('--teacher-corrections requires a distillation recipe')
        recipe['distillation']['teacher_corrections'] = a.teacher_corrections
        recipe = resolve(recipe)
        h = recipe['training']
    if distilling:
        if a.init_base or a.init_pair or (not a.resume and not a.teacher):
            p.error('Distillation initializes from --teacher; use --resume to continue')
        initialization_mode = 'teacher'
    elif a.teacher:
        p.error('--teacher requires a distillation recipe')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if a.global_batch_size is not None:
        divisor = world * h['accumulation']
        if a.global_batch_size < divisor or a.global_batch_size % divisor:
            p.error('--global-batch-size must be divisible by world size times accumulation')
        h['batch_size'] = a.global_batch_size // divisor
    if h['mode'] == 'corrector' and not (a.init_base or a.init_pair or a.resume):
        p.error('Frozen-base corrector training requires --init-base, --init-pair, or --resume')
    local = int(os.environ['LOCAL_RANK'])
    if h['deterministic']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', device_id=torch.device(f'cuda:{local}'))
    torch.set_num_threads(1)
    torch.manual_seed(h['seed'] * world + rank)
    np.random.seed(h['seed'] * world + rank)
    torch.use_deterministic_algorithms(h['deterministic'])
    torch.backends.cudnn.deterministic = h['deterministic']
    torch.backends.cudnn.benchmark = not h['deterministic']
    if recipe['model']['model'].startswith('SiT-'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if distilling or a.init_pair or a.init_base:
        # Keep long downloads on a CPU group without changing the training group.
        download_group = dist.new_group(backend='gloo', timeout=timedelta(hours=24))
        from src.utils.download import resolve_checkpoint
        location = [None, None]
        if rank == 0:
            try:
                if a.init_base:
                    location[0] = str(resolve_pretrained(a.init_base, recipe).resolve())
                else:
                    requested = (a.teacher or previous['teacher']['directory']) if distilling else a.init_pair
                    location[0] = str(resolve_checkpoint(requested).resolve())
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                location[1] = str(error)
        dist.broadcast_object_list(location, src=0, group=download_group)
        dist.destroy_process_group(download_group)
        if location[1]:
            raise ValueError(location[1])
        initial_dir = Path(location[0])
    if rank == 0:
        print(f'Loading training data from {a.data}', flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    cached = (a.data/'latents.json').exists()
    data_args = dict(seed=h['seed'], horizontal_flip=h['horizontal_flip'])
    data = (LatentMoments(a.data, **data_args) if cached else
            ImageNet(a.data, image_size=recipe['model']['image_size'], **data_args))
    n = len(data)
    microbatches = (n + world - 1) // world // h['batch_size']
    if microbatches < h['accumulation']:
        raise ValueError('Dataset is too small for one accumulated update')
    teacher_info = dict(directory=str(initial_dir)) if distilling else None
    if distilling and previous and previous['teacher']['directory'] != str(initial_dir):
        p.error('Use the same teacher directory when resuming distillation')
    settings = dict(recipe=recipe, world_size=world,
                    global_batch_size=world*h['batch_size']*h['accumulation'],
                    initialization_mode=initialization_mode, dataset=data.manifest,
                    compile=a.compile)
    if distilling:
        settings['teacher'] = teacher_info
    if rank == 0:
        print(f"Initializing {recipe['model']['model']} ({h['mode']}); "
              f"{n:,} images, global batch {settings['global_batch_size']}", flush=True)
    if recipe['model']['model'].startswith('SiT-') and (initial_dir or a.resume):
        with torch.device('meta'):
            model = build_model(recipe)
        model.to_empty(device='cuda')
    else:
        model = build_model(recipe).cuda()
    model.train()
    teacher = None
    if distilling:
        teacher, teacher_config = load_model(initial_dir)
        teacher.requires_grad_(False)
        initialize_student(model, teacher, recipe, teacher_config)
    codec_config = CodecConfig(**recipe['codec'])
    if cached:
        if codec_config.name != 'vae' or data.manifest['image_size'] != recipe['model']['image_size']:
            raise ValueError('Latent cache image size/codec mismatch')
        if data.manifest['codec'] != recipe['codec']:
            raise ValueError('Latent cache VAE/normalization differs from this recipe')
    if rank == 0 and not cached and codec_config.name == 'vae':
        print(f'Loading VAE {codec_config.pretrained}', flush=True)
    codec = None if cached else build_codec(recipe).eval()
    if a.resume:
        if rank == 0:
            print(f'Restoring training checkpoint from {last}', flush=True)
        with last.open('rb', buffering=0) as checkpoint_file:
            ck = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'], strict=True)
    else:
        if initial_dir and not distilling:
            initialize_model(model, recipe, initial_dir, mode=initialization_mode)
        elif not distilling:
            model.initialize_corrector()
        ck = None
    if rank == 0:
        if manifest.exists():
            expected = json.loads(manifest.read_text())
            for name in ('steps', 'save_every', 'export_every'):
                expected['recipe']['training'][name] = h[name]
            assert expected == settings, 'Resume model or optimizer settings changed'
        manifest.write_text(json.dumps(settings, indent=2)+'\n')
    dist.barrier()
    wrapped = DDP(model, device_ids=[local], gradient_as_bucket_view=True,
                  bucket_cap_mb=h['gradient_bucket_mb'])
    averages = [copy.deepcopy(model).requires_grad_(False).eval() for _ in h['ema']]
    for i, average in enumerate(averages, 1):
        if ck is not None:
            average.load_state_dict(ck[f'ema{i}'], strict=True)
        elif a.init_base and initial_dir.is_dir():
            # Initialize base EMA from pretrained EMA, and the corrector from raw base weights.
            ema_path = initial_dir / f'base_ema{i}.safetensors'
            if ema_path.exists():
                average.net.load_state_dict(load_file(str(ema_path)), strict=True)
    if a.compile:
        for net in (model.net, model.corr_net):
            if net is None:
                continue
            for block in [*net.blocks, net.final_layer]:
                block.forward = torch.compile(block.forward)
    if h['gradient_precision'] == 'bf16':
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
        wrapped.register_comm_hook(None, bf16_compress_hook)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=h['lr'],
                                 betas=tuple(h['betas']), eps=1e-8, weight_decay=0., fused=True)
    names = [name for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    source_parameters = [dict(model.named_parameters())[name] for name in names]
    average_parameters = [[dict(average.named_parameters())[name] for name in names] for average in averages]
    step, epoch, start_batch = 0, 0, 0
    if ck is not None:
        optimizer.load_state_dict(ck['optimizer'])
        step, epoch, start_batch = ck['step'], ck['epoch'], ck['next_batch']
        state = ck['rng'][rank]
        torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state(state['cuda'])
        del ck
    if rank == 0:
        print(f"Training: {step}/{h['steps']} updates completed", flush=True)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    window_started, window_step = time.monotonic(), step
    loss_totals = torch.zeros(3 if distilling else 2, device='cuda')
    while step < h['steps']:
        data.epoch = epoch
        sampler = DistributedSampler(data, world, rank, seed=h['seed'])
        sampler.set_epoch(epoch)
        indices = list(iter(sampler))[start_batch * h['batch_size']:]
        loader = DataLoader(data, batch_size=h['batch_size'], sampler=indices, num_workers=a.workers,
                            pin_memory=True, drop_last=True,
                            generator=torch.Generator().manual_seed(epoch * world + rank))
        optimizer.zero_grad(set_to_none=True)
        loss_totals.zero_()
        for batch, (images, labels) in enumerate(loader, start=start_batch):
            progress = epoch + batch / microbatches
            lr = h['lr'] * (min(progress / h['warmup_epochs'], 1.) if h['warmup_epochs'] else 1.)
            if h['warmup_steps']:
                lr = h['lr'] * min((step + 1) / h['warmup_steps'], 1.)
            for group in optimizer.param_groups:
                group['lr'] = lr
            synchronize = (batch + 1) % h['accumulation'] == 0
            with nullcontext() if synchronize else wrapped.no_sync():
                inputs = images.cuda(non_blocking=True)
                state = sample_posterior(inputs, codec_config) if cached else codec.encode(inputs)
                labels = labels.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=h['precision'] == 'bf16'):
                    if distilling:
                        z, t, base, target = teacher_targets(
                            teacher, state, labels, recipe['distillation']['teacher_corrections'])
                        base_loss, correction_loss = wrapped(z, labels, t=t, target_velocity=target,
                                                             return_losses=True)
                        correction_size = (target.float() - base.float()).square().mean()
                    else:
                        base_loss, correction_loss = wrapped(state, labels, return_losses=True)
                    loss = h['base_loss_weight']*base_loss + h['correction_loss_weight']*correction_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at step {step}')
                loss_totals.add_(torch.stack([base_loss.detach(), correction_loss.detach()] +
                                            ([correction_size.detach()] if distilling else [])))
                (loss / h['accumulation']).backward()
            if (batch + 1) % h['accumulation']:
                continue
            if h['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), h['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for targets, decay in zip(average_parameters, h['ema']):
                    torch._foreach_mul_(targets, decay)
                    torch._foreach_add_(targets, source_parameters, alpha=1-decay)
            step += 1
            elapsed = time.monotonic()-started
            halt = torch.tensor(int(stop), device='cuda')
            dist.all_reduce(halt, op=dist.ReduceOp.MAX)
            halt = bool(halt.item())
            if step % 20 == 0 or step == 1 or step == h['steps'] or halt:
                values = loss_totals / h['accumulation']
                dist.all_reduce(values)
                values /= world
                torch.cuda.synchronize()
                now = time.monotonic()
                rate = (step-window_step)/(now-window_started)
                if rank == 0:
                    base, correction = values[:2].tolist()
                    status = dict(step=step,epoch=epoch,batch=batch,
                                  loss=h['base_loss_weight']*base+h['correction_loss_weight']*correction,
                                  base_loss=base,correction_loss=correction,lr=lr,seconds=elapsed,
                                  steps_per_second=rate,seconds_per_step=1/rate,
                                  eta_seconds=(h['steps']-step)/rate,
                                  tf32=torch.backends.cuda.matmul.allow_tf32,
                                  world_size=world,global_batch_size=settings['global_batch_size'])
                    if distilling:
                        size = values[2].item()
                        status.update(distill_mse=base, correction_mse=size,
                                      residual_frac=base/max(size, 1e-12))
                    print(json.dumps(status),flush=True)
                    (a.output/'performance.json').write_text(json.dumps(status)+'\n')
                window_started, window_step = now, step
            loss_totals.zero_()
            save_restart = step % h['save_every'] == 0 or step == h['steps'] or halt
            export_ema = step % h['export_every'] == 0 or step == h['steps']
            if save_restart or export_ema:
                saving_started = time.monotonic()
                if rank == 0:
                    print(f'Saving step {step}...', flush=True)
                if save_restart:
                    rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
                    all_rng = [None] * world
                    dist.all_gather_object(all_rng, rng)
                if rank == 0:
                    if save_restart:
                        temporary = last.with_suffix('.tmp')
                        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                        **{f'ema{i}': average.state_dict() for i, average in enumerate(averages, 1)},
                                        step=step, epoch=epoch, next_batch=batch+1, rng=all_rng), temporary)
                        temporary.replace(last)
                        (a.output/'progress.json').write_text(json.dumps(dict(step=step,epoch=epoch,seconds=elapsed,complete=step==h['steps']))+'\n')
                    if export_ema:
                        out = a.output / f'step-{step:07d}'
                        source = dict(step=step,
                                      ema=f'model_ema{len(averages)}',
                                      dataset='imagenet')
                        if distilling:
                            source['distillation'] = dict(
                                teacher=str(initial_dir),
                                **recipe['distillation'])
                        save_checkpoint(out, averages[-1].state_dict(), recipe,
                                        source=source, kind='base_init' if model.corr_net is None else 'pair')
                dist.barrier()
                saving_seconds = time.monotonic() - saving_started
                window_started += saving_seconds
                if rank == 0:
                    print(f'Saved step {step} in {saving_seconds:.1f}s', flush=True)
            if step == h['steps'] or halt:
                dist.destroy_process_group()
                return
        optimizer.zero_grad(set_to_none=True)  # Discard incomplete accumulation groups at the end of an epoch.
        epoch += 1
        start_batch = 0
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
