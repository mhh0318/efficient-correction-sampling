"""Combine labeled sampling grids; run parents select their latest run."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='Run directories or parent directories')
    parser.add_argument('--labels', nargs='+', help='One label per input directory')
    parser.add_argument('--horizontal', action='store_true', help='Arrange panels side by side')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.inputs):
        parser.error('Provide one label per input directory')
    panels = []
    for directory in args.inputs:
        path = directory / 'grid.png'
        if not path.is_file():
            runs = sorted(directory.glob('*/grid.png'))
            if not runs:
                parser.error(f'No grid.png found in {directory}')
            path = runs[-1]
        with Image.open(path) as image:
            panels.append(image.convert('RGB'))
    labels = args.labels or [path.name for path in args.inputs]
    label_height, gap = 32, 12
    if args.horizontal:
        width = sum(image.width for image in panels) + gap * (len(panels) - 1)
        height = max(image.height for image in panels) + label_height
    else:
        width = max(image.width for image in panels)
        height = sum(image.height + label_height for image in panels) + gap * (len(panels) - 1)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    x, y = 0, 0
    for label, image in zip(labels, panels):
        label_x = x + (image.width - draw.textlength(label, font=font))/2 if args.horizontal else 8
        draw.text((label_x, y + 5), label, fill='black', font=font)
        canvas.paste(image, (x, y + label_height))
        if args.horizontal:
            x += image.width + gap
        else:
            y += image.height + label_height + gap
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f'Saved comparison grid to {args.output}')


if __name__ == '__main__':
    main()
