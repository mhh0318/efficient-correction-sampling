"""Index the official ILSVRC2012 training tar without extracting its images."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    archive = a.directory / 'ILSVRC2012_img_train.tar'
    rows, counts = [], Counter()
    with tarfile.open(archive, 'r:') as outer:
        members = sorted((m for m in outer.getmembers() if m.isfile() and m.name.endswith('.tar')), key=lambda m: m.name)
        assert len(members) == 1000
        names = [Path(m.name).stem for m in members]
        assert len(set(names)) == 1000 and all(n.startswith('n') and n[1:].isdigit() for n in names)
        for label, member in enumerate(members):
            with tarfile.open(fileobj=outer.extractfile(member), mode='r:') as inner:
                for image in inner:
                    if not image.isfile():
                        continue
                    assert image.name.lower().endswith(('.jpg', '.jpeg'))
                    rows.append((member.offset_data + image.offset_data, image.size, label))
                    counts[names[label]] += 1
            if label % 50 == 0:
                print(f'Indexed {label+1} classes, {len(rows)} images', flush=True)
    assert len(rows) == 1281167 and len(counts) == 1000
    temporary = a.directory / 'index.tmp.npy'
    np.save(temporary, np.asarray(rows, dtype=np.int64))
    temporary.replace(a.directory / 'index.npy')
    record = dict(classes=1000, images=len(rows), class_to_idx={n: i for i, n in enumerate(names)},
                  counts=dict(counts),
                  source='https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
                  preprocessing='Original JPEG bytes; ADM center crop 256 and random horizontal flip at training time')
    (a.directory / 'dataset.json').write_text(json.dumps(record, indent=2)+'\n')
    print('PASS: official ImageNet training archive validated and indexed', flush=True)


if __name__ == '__main__':
    main()
