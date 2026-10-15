"""Match a frozen denoiser + corrector with one ordinary denoiser evaluation."""
import torch


def initialize_student(student, teacher, recipe, teacher_config):
    if teacher_config.get('kind', 'pair') != 'pair':
        raise ValueError('Distillation teacher must be a trained pair bundle')
    fields = ('model', 'image_size', 'num_classes', 'prediction', 'data_time', 't_eps', 'noise_scale')
    if (any(recipe['model'][key] != teacher_config['model'][key] for key in fields)
            or recipe['codec'] != teacher_config['codec']):
        raise ValueError('Teacher architecture, prediction, path or codec differs from student')
    if student.corr_net is not None or teacher.corr_net is None:
        raise ValueError('Distillation requires a plain student and a teacher with a corrector')
    student.net.load_state_dict(teacher.net.state_dict(), strict=True)


@torch.no_grad()
def teacher_targets(teacher, x, labels, corrections, *, t=None, noise=None):
    """Uniform-time real-data interpolants; no CFG or sampling-interval mask."""
    if t is None:
        t = torch.rand(x.size(0), device=x.device).clamp_min(torch.finfo(torch.float32).eps)
    t = t.reshape(-1, 1, 1, 1)
    noise = torch.randn_like(x) * teacher.noise_scale if noise is None else noise
    z = teacher.space.interpolate(x, noise, t)
    proposal = teacher.net(z, t.flatten(), labels)
    base = teacher.space.to_velocity(z, t, proposal, teacher.prediction)
    target = teacher.correct(z, t, labels, base, steps=corrections,
                             proposal=proposal if teacher.correction.input == teacher.prediction else None)
    return z, t, base.detach(), target.detach()
