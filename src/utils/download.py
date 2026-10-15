"""Download trained checkpoints: python src/utils/download.py CHECKPOINT_NAME."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from tqdm import tqdm

CATALOG = Path(__file__).resolve().parents[2] / 'configs/checkpoints.json'


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


class AssetRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        redirected = super().redirect_request(request, fp, code, message, headers, url)
        if redirected and urllib.parse.urlsplit(url).netloc != urllib.parse.urlsplit(request.full_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


OPENER = urllib.request.build_opener(AssetRedirect())


def token():
    value = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if value:
        return value
    if shutil.which('gh'):
        result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def fetch(url, path, expected, headers=None):
    """Resume a transport part; publish it only after size and SHA256 verification."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size == expected['size'] and sha256(path) == expected['sha256']:
            return
        raise ValueError(f'Corrupt cached file: {path}; remove it and retry')
    partial = path.with_name(path.name + '.partial')
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected['size']:
            partial.unlink()
            offset = 0
        if offset < expected['size']:
            request_headers = dict(headers or {}, Accept='application/octet-stream', **{'Cache-Control': 'no-cache'})
            if offset:
                request_headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=request_headers)
            try:
                with OPENER.open(request, timeout=120) as response:
                    if response.status == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Invalid download range')
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise ValueError(f'Unexpected HTTP status {response.status}')
                    with partial.open('ab' if offset else 'wb') as output, tqdm(
                        total=expected['size'], initial=offset, desc=path.name,
                        unit='B', unit_scale=True, unit_divisor=1024,
                    ) as progress:
                        while block := response.read(8 * 1024**2):
                            output.write(block)
                            progress.update(len(block))
                            if output.tell() > expected['size']:
                                raise ValueError(f'Download exceeds expected size: {path.name}')
            except (OSError, urllib.error.URLError) as error:
                if attempt == 2:
                    status = getattr(error, 'code', 'network error')
                    raise RuntimeError(f'Download failed ({status}): {path.name}; rerun to resume') from None
                time.sleep(2**attempt)
                continue
        if partial.stat().st_size != expected['size']:
            if attempt == 2:
                raise RuntimeError(f'Incomplete download: {path.name}; rerun to resume')
            continue
        if sha256(partial) != expected['sha256']:
            partial.unlink()
            raise ValueError(f'Checksum mismatch: {path.name}; no checkpoint was installed')
        partial.replace(path)
        return


def assemble(parts, output, expected):
    """Join verified transport parts without changing the original safetensors bytes."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.weights-', delete=False) as file:
        temporary = Path(file.name)
        digest, size = hashlib.sha256(), 0
        try:
            for part in parts:
                with Path(part).open('rb') as incoming:
                    while block := incoming.read(8 * 1024**2):
                        file.write(block)
                        digest.update(block)
                        size += len(block)
            file.flush()
            if size != expected['size'] or digest.hexdigest() != expected['sha256']:
                raise ValueError('Assembled checkpoint checksum mismatch; no checkpoint was installed')
            file.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def download_model(name, output=None, *, repo=None):
    """Resolve one official bundle, using verified local files before any network access."""
    catalog = json.loads(CATALOG.read_text())
    if name not in catalog['models']:
        raise ValueError(f'Unknown official checkpoint: {name}')
    repo = repo or catalog['repository']
    output = Path(output if output is not None else os.environ.get('CORRECTION_CHECKPOINT_DIR', 'checkpoints')).expanduser()
    entry = catalog['models'][name]
    destination = output / name
    config_path, weights_path = destination / 'config.json', destination / 'model.safetensors'
    if config_path.exists():
        current = json.loads(config_path.read_text())
        if any(current.get(key) != value for key, value in entry['config'].items()):
            raise ValueError(f'{destination} contains a different recipe; use another directory')
    if weights_path.exists():
        if weights_path.stat().st_size != entry['weights']['size']:
            raise ValueError(f'{weights_path} contains different weights; use another directory')
        if config_path.is_file():
            return destination
    else:
        print(f'Downloading {name} ({entry["weights"]["size"] / 1024**3:.2f} GiB) to {destination}', flush=True)
        credentials = token()
        headers = {'User-Agent': 'correction-training-downloader'}
        if credentials:
            headers['Authorization'] = f'Bearer {credentials}'
        url = f'https://api.github.com/repos/{repo}/releases/tags/{catalog["release"]}'
        try:
            with OPENER.open(urllib.request.Request(url, headers=headers), timeout=60) as response:
                release = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f'Release unavailable (HTTP {error.code}). '
                f'Check https://github.com/{repo}/releases/tag/{catalog["release"]}. '
                'For API rate limits or a private mirror, run gh auth login or set GITHUB_TOKEN.'
            ) from None
        assets = {asset['name']: asset for asset in release['assets']}
        cache = output / '.downloads' / repo.replace('/', '--') / catalog['release']
        parts = []
        for part in entry['weights']['parts']:
            asset = assets.get(part['name'])
            if not asset or asset['size'] != part['size'] or asset['state'] != 'uploaded':
                raise ValueError(f'Release asset missing or incomplete: {part["name"]}')
            print(f'{name}: {part["name"]} ({part["size"] / 1024**3:.2f} GiB)', flush=True)
            path = cache / part['name']
            asset_url = (asset.get('browser_download_url', asset['url'])
                         if not credentials else asset['url'])
            fetch(asset_url, path, part, headers)
            parts.append(path)
        assemble(parts, weights_path, entry['weights'])
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / '.config.json.tmp'
    temporary.write_text(json.dumps(entry['config'], indent=2) + '\n')
    temporary.replace(config_path)
    return destination


def resolve_checkpoint(value, root=None):
    """Official names may download; explicit local paths never trigger a download."""
    names = json.loads(CATALOG.read_text())['models']
    if str(value) in names:
        return download_model(str(value), root)
    directory = Path(value).expanduser()
    if not all((directory / name).is_file() for name in ('config.json', 'model.safetensors')):
        raise FileNotFoundError(f'Checkpoint bundle not found: {directory}. Expected config.json and model.safetensors, or an official name: {", ".join(names)}')
    return directory


def main():
    catalog = json.loads(CATALOG.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='+', choices=[*catalog['models'], 'all'])
    parser.add_argument('--output', type=Path, help='Root directory for model bundles (default: CORRECTION_CHECKPOINT_DIR or checkpoints)')
    parser.add_argument('--repo', default=catalog['repository'], help='Release repository (for a migrated release)')
    args = parser.parse_args()
    names = list(catalog['models']) if 'all' in args.models else list(dict.fromkeys(args.models))
    for name in names:
        try:
            destination = download_model(name, args.output, repo=args.repo)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        print(f'Ready: {destination}', flush=True)


if __name__ == '__main__':
    main()
