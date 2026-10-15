# ImageNet data preparation

## Download

Use the complete ImageNet (ILSVRC2012) training set. The
[official ImageNet download page](https://image-net.org/download.php)
links to the [Kaggle ILSVRC dataset](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).
Sign in and accept the dataset terms there before downloading. With the Kaggle
CLI configured for your account:

```bash
pip install kaggle
kaggle competitions download -c imagenet-object-localization-challenge -p /data/imagenet
unzip /data/imagenet/imagenet-object-localization-challenge.zip \
  'ILSVRC/Data/CLS-LOC/train/*' -d /data/imagenet
```

Pass `/data/imagenet/ILSVRC/Data/CLS-LOC/train` to `src/train.py --data`. An existing
ImageFolder layout also works:

```text
imagenet/train/n01440764/*.JPEG
imagenet/train/n01443537/*.JPEG
...
```

Both `imagenet` and `imagenet/train` are accepted. Class indices follow sorted
WordNet directory names, as in torchvision ImageFolder.

If you already have the official `ILSVRC2012_img_train.tar`, indexing it avoids
extracting 1.3M small files:

```bash
python src/tools/prepare_imagenet.py /data/imagenet
```

Pass the indexed directory to training or caching.

## Preprocessing

The default follows [official SiT](https://github.com/willisma/SiT/blob/main/train.py):
ADM center crop to the configured image size, random horizontal flip with
probability 0.5, and RGB normalization to `[-1, 1]`. Latent models use
`stabilityai/sd-vae-ft-ema`, draw a fresh VAE posterior sample on each access,
and multiply latents by 0.18215. VAE computation remains FP32.

The supplied training recipes enable random horizontal flipping. Set
`training.horizontal_flip=false` to disable it, as in the
[public REPA loader](https://github.com/sihyun-yu/REPA/blob/main/dataset.py).

## Optional packed VAE cache

Use this for repeated latent training on filesystems where many small reads are
slow. From either the ImageFolder root or indexed tar directory:

```bash
torchrun --standalone --nproc_per_node=8 src/tools/cache_latents.py \
  --config configs/train/sit.yml --data /data/imagenet \
  --output /data/imagenet-moments
```

For a Kaggle extraction, replace `--data /data/imagenet` with the `train` path
above. Pass `/data/imagenet-moments` to training afterward.

The cache occupies about **84 GB** for ImageNet-256. It stores VAE posterior
moments for original and flipped images, while training draws a fresh posterior
sample on each access. Use the same codec config when caching and training.
