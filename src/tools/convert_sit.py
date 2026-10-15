"""Convert official SiT or REPA-SiT base weights to a shared initialization bundle."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import read_config, build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.pretrained import load_pretrained_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--family', choices=['sit', 'repa'], default='sit')
    p.add_argument('--config', type=Path)
    p.add_argument('--key', help='State-dict key (default: ema if present, otherwise a plain state dict)')
    a = p.parse_args()
    default = 'sit.yml' if a.family == 'sit' else 'sit-repa.yml'
    recipe = read_config(a.config or Path(__file__).resolve().parents[2]/'configs'/'train'/default)
    direction = 1 if a.family == 'sit' else 0
    if not recipe['model']['model'].startswith('SiT-') or recipe['model']['prediction'] != 'v' or recipe['model']['data_time'] != direction:
        p.error('Official SiT/REPA initialization requires its native SiT velocity path')
    state, source = load_pretrained_state(a.checkpoint, a.family, a.key)
    model = build_model(recipe)
    model.net.load_state_dict(state, strict=True)
    model.initialize_corrector()
    save_checkpoint(a.output, model.state_dict(), recipe,
                    kind='base_init',
                    source=dict(**source,
                                purpose='official SiT/REPA base initialization; fresh corrector'))
    print(a.output)


if __name__ == '__main__': main()
