"""Check identical JPEG preprocessing across folder/tar and explicit flip choices."""
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from src.datasets import imagenet as data
from src.datasets.imagenet import ImageNet, LatentMoments


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.constants = patch.multiple(data, IMAGENET_CLASSES=2, IMAGENET_IMAGES=4)
        self.constants.start()
        self.addCleanup(self.constants.stop)
        self.folder = self.root/'folder'
        rows = []
        for label in range(2):
            name = f'n{label+1:08d}'
            (self.folder/'train'/name).mkdir(parents=True)
            for image in range(2):
                pixels = np.random.default_rng(label*2+image).integers(0, 256, (23, 37, 3), dtype=np.uint8)
                path = self.folder/'train'/name/f'{image}.JPEG'
                Image.fromarray(pixels).save(path)
                rows.append((path, label))
        self.tar = self.root/'tar'
        self.tar.mkdir()
        archive = self.tar/'ILSVRC2012_img_train.tar'
        with tarfile.open(archive, 'w:') as f:
            for path, _ in rows:
                f.add(path, arcname=str(path.relative_to(self.folder/'train')))
        with tarfile.open(archive, 'r:') as f:
            index = [(m.offset_data, m.size, rows[i][1])
                     for i, m in enumerate(m for m in f if m.isfile())]
        np.save(self.tar/'index.npy', np.asarray(index, dtype=np.int64))
        (self.tar/'dataset.json').write_text(json.dumps(dict(classes=2, images=4)))

    def close_tar(self, dataset):
        if dataset.fd is not None:
            os.close(dataset.fd)
            dataset.fd = None

    def test_folder_tar_labels_and_pixels_match_in_both_flip_modes(self):
        for flipping in (False, True):
            folder = ImageNet(self.folder, image_size=16, horizontal_flip=flipping)
            tar = ImageNet(self.tar, image_size=16, horizontal_flip=flipping)
            self.addCleanup(self.close_tar, tar)
            for epoch in (0, 1):
                folder.epoch = tar.epoch = epoch
                for i in range(4):
                    a, label_a = folder[i]
                    b, label_b = tar[i]
                    self.assertEqual(label_a, i//2)
                    self.assertEqual(label_a, label_b)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(a.shape, (3, 16, 16))
                    self.assertTrue(a.min() >= -1 and a.max() <= 1)

    def test_flip_is_explicit_and_changes_only_horizontal_orientation(self):
        original = ImageNet(self.folder/'train', image_size=16, horizontal_flip=False)
        augmented = ImageNet(self.folder, image_size=16, horizontal_flip=True)
        observed = set()
        for epoch in range(10):
            original.epoch = augmented.epoch = epoch
            for i in range(4):
                a, _ = original[i]
                b, _ = augmented[i]
                flipped = np.random.default_rng(np.random.SeedSequence([0, epoch, i])).random() < .5
                observed.add(bool(flipped))
                torch.testing.assert_close(b, a.flip(-1) if flipped else a, rtol=0, atol=0)
        self.assertEqual(observed, {False, True})

    def test_cached_views_follow_same_flip_policy(self):
        root = self.root/'cache'
        root.mkdir()
        moments = np.zeros((4, 2, 8, 4, 4), dtype=np.float32)
        moments[:, 1] = 1.
        np.save(root/'moments.npy', moments)
        np.save(root/'labels.npy', np.array([0, 0, 1, 1]))
        (root/'latents.json').write_text(json.dumps(dict(
            format='imagenet-vae-moments/1', complete=True, images=4,
            shards=[dict(start=0, end=4, file='moments.npy')])))
        for flipping in (False, True):
            dataset = LatentMoments(root, horizontal_flip=flipping)
            for epoch in range(3):
                dataset.epoch = epoch
                for i in range(4):
                    params, label = dataset[i]
                    expected = int(flipping and np.random.default_rng(
                        np.random.SeedSequence([0, epoch, i])).random() < .5)
                    self.assertEqual(label, i//2)
                    self.assertTrue(torch.all(params == expected))

    def test_partial_class_directory_is_rejected(self):
        with patch.object(data, 'IMAGENET_CLASSES', 1000), self.assertRaisesRegex(ValueError, 'Expected train/'):
            ImageNet(self.folder)

    def test_empty_class_is_rejected_even_if_total_count_matches(self):
        for path in (self.folder/'train/n00000002').glob('*.JPEG'):
            path.rename(self.folder/'train/n00000001'/f'extra-{path.name}')
        with self.assertRaisesRegex(ValueError, 'Every ImageNet class'):
            ImageNet(self.folder)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
