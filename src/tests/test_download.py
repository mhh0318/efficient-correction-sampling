"""Exercise interrupted transfers, credentials, and corrupt checkpoint rejection."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import download
from src.utils.download import assemble, fetch, resolve_checkpoint


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = bytes(range(256)) * 128
        self.expected = dict(size=len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest())
        self.requests = []
        test = self

        class Asset(BaseHTTPRequestHandler):
            def do_GET(self):
                test.requests.append(dict(authorization=self.headers.get('Authorization'),
                                          range=self.headers.get('Range')))
                offset = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
                self.send_response(206 if offset else 200)
                if offset:
                    self.send_header('Content-Range', f'bytes {offset}-{len(test.payload)-1}/{len(test.payload)}')
                self.send_header('Content-Length', str(len(test.payload) - offset))
                self.end_headers()
                self.wfile.write(test.payload[offset:])

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Asset)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f'http://127.0.0.1:{self.server.server_port}/asset'
        self.catalog = self.root / 'checkpoints.json'
        self.config = dict(format='correction-training/2', weights_sha256=self.expected['sha256'])
        self.catalog.write_text(json.dumps(dict(repository='test/repo', release='v1', models={
            'toy': dict(config=self.config, weights=self.expected | dict(parts=[
                self.expected | dict(name='toy.part000')]))})))
        patch = mock.patch.object(download, 'CATALOG', self.catalog)
        patch.start()
        self.addCleanup(patch.stop)

    def install(self, config=None, payload=None):
        destination = self.root / 'checkpoints' / 'toy'
        destination.mkdir(parents=True)
        (destination / 'config.json').write_text(json.dumps(self.config if config is None else config))
        (destination / 'model.safetensors').write_bytes(self.payload if payload is None else payload)
        return destination

    def test_official_name_downloads_then_reuses_verified_bundle_offline(self):
        release = dict(assets=[dict(name='toy.part000', url=self.url,
                                   size=len(self.payload), state='uploaded')])
        open_url = download.OPENER.open
        def respond(request, **kwargs):
            if request.full_url.startswith('https://api.github.com/'):
                return io.BytesIO(json.dumps(release).encode())
            return open_url(request, **kwargs)
        with mock.patch.object(download.OPENER, 'open', side_effect=respond), \
                mock.patch.object(download, 'token', return_value=None):
            destination = resolve_checkpoint('toy', self.root / 'checkpoints')
        self.assertEqual((destination / 'model.safetensors').read_bytes(), self.payload)
        self.assertEqual(json.loads((destination / 'config.json').read_text()), self.config)
        with mock.patch.object(download.OPENER, 'open', side_effect=AssertionError('network used')), \
                mock.patch.object(download, 'token', side_effect=AssertionError('credentials queried')), \
                mock.patch.object(download, 'sha256', side_effect=AssertionError('cached weights scanned')):
            self.assertEqual(resolve_checkpoint('toy', self.root / 'checkpoints'), destination)
        self.assertEqual(len(self.requests), 1)

    def test_anonymous_download_uses_public_asset_url(self):
        release = dict(assets=[dict(name='toy.part000',
            url='https://api.github.com/repos/test/repo/releases/assets/1',
            browser_download_url=self.url, size=len(self.payload), state='uploaded')])
        with mock.patch.object(download.OPENER, 'open',
                return_value=io.BytesIO(json.dumps(release).encode())), \
                mock.patch.object(download, 'token', return_value=None), \
                mock.patch.object(download, 'fetch') as fetch_part, \
                mock.patch.object(download, 'assemble'):
            resolve_checkpoint('toy', self.root / 'checkpoints')
        self.assertEqual(fetch_part.call_args.args[0], self.url)
        self.assertNotIn('Authorization', fetch_part.call_args.args[3])

    def test_explicit_local_path_never_downloads_or_requires_catalog_match(self):
        destination = self.install(config=dict(local_training=True))
        with mock.patch.object(download, 'download_model', side_effect=AssertionError('download used')):
            self.assertEqual(resolve_checkpoint(str(destination)), destination)

    def test_pretrained_names_download_from_source_then_reuse_cache_offline(self):
        import torch
        from src.utils.config import read_config
        from src.utils.pretrained import PRETRAINED, resolve_pretrained
        root = Path(__file__).resolve().parents[2]
        with mock.patch.dict('os.environ', CORRECTION_CHECKPOINT_DIR=str(self.root / 'base-cache')):
            for name in ('sit', 'sit-repa'):
                recipe = read_config(root / f'configs/train/{name}.yml')
                with mock.patch.dict(PRETRAINED, {name: dict(url=self.url)}):
                    path = resolve_pretrained(name, recipe)
                self.assertEqual(path.read_bytes(), self.payload)
                with mock.patch.object(torch.hub, 'download_url_to_file', side_effect=AssertionError('network used')):
                    self.assertEqual(resolve_pretrained(name, recipe), path)
                    self.assertEqual(resolve_pretrained(str(path), recipe), path)
            wrong = read_config(root / 'configs/train/sit-b-v.yml')
            with mock.patch.object(torch.hub, 'download_url_to_file') as download_base:
                with self.assertRaisesRegex(ValueError, 'SiT-XL/2'):
                    resolve_pretrained('sit', wrong)
                download_base.assert_not_called()
        self.assertEqual(len(self.requests), 2)

    def test_huggingface_base_uses_pinned_revision_and_reuses_local_file(self):
        from src.utils.config import read_config
        from src.utils.pretrained import PRETRAINED, resolve_pretrained
        root = Path(__file__).resolve().parents[2]
        recipe = read_config(root / 'configs/train/sit.yml')
        source = dict(repo_id='test/base', filename='weights/model.safetensors', revision='a'*40)
        cache = self.root / 'hf-base-cache'
        def download_file(*, repo_id, filename, revision, local_dir):
            self.assertEqual((repo_id, revision), ('test/base', 'a'*40))
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.payload)
            return str(path)
        with mock.patch.dict('os.environ', CORRECTION_CHECKPOINT_DIR=str(cache)), \
                mock.patch.dict(PRETRAINED, sit=source), \
                mock.patch('huggingface_hub.hf_hub_download', side_effect=download_file) as download_base:
            path = resolve_pretrained('sit', recipe)
            self.assertEqual(path.read_bytes(), self.payload)
            self.assertEqual(resolve_pretrained('sit', recipe), path)
            self.assertEqual(resolve_pretrained(str(path), recipe), path)
            download_base.assert_called_once()

    def test_environment_cache_root_and_explicit_override(self):
        explicit = self.install()
        root = self.root / 'environment-cache'
        destination = root / 'toy'
        destination.mkdir(parents=True)
        (destination / 'config.json').write_text(json.dumps(self.config))
        (destination / 'model.safetensors').write_bytes(self.payload)
        with mock.patch.dict(download.os.environ, {'CORRECTION_CHECKPOINT_DIR': str(root)}), \
                mock.patch.object(download.OPENER, 'open', side_effect=AssertionError('network used')):
            self.assertEqual(resolve_checkpoint('toy'), destination)
            self.assertEqual(resolve_checkpoint('toy', self.root / 'checkpoints'), explicit)

    def test_unknown_name_or_incomplete_path_fails_without_network(self):
        destination = self.root / 'incomplete'
        destination.mkdir()
        (destination / 'config.json').write_text('{}')
        with mock.patch.object(download.OPENER, 'open', side_effect=AssertionError('network used')):
            for value in ('typo-model', str(destination)):
                with self.subTest(value=value), self.assertRaisesRegex(FileNotFoundError, 'Checkpoint bundle not found'):
                    resolve_checkpoint(value)

    def test_cached_official_weights_or_recipe_cannot_be_silently_replaced(self):
        destination = self.install(payload=b'corrupt')
        with mock.patch.object(download.OPENER, 'open', side_effect=AssertionError('network used')):
            with self.assertRaisesRegex(ValueError, 'different weights'):
                resolve_checkpoint('toy', self.root / 'checkpoints')
            self.assertEqual((destination / 'model.safetensors').read_bytes(), b'corrupt')
            (destination / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'different recipe'):
                resolve_checkpoint('toy', self.root / 'checkpoints')

    def test_private_release_auth_failure_reports_login_instructions(self):
        from urllib.error import HTTPError
        with mock.patch.object(download.OPENER, 'open', side_effect=HTTPError('test', 404, 'private', {}, None)), \
                mock.patch.object(download, 'token', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'gh auth login'):
                resolve_checkpoint('toy', self.root / 'checkpoints')

    def test_sample_cli_passes_resolved_bundle_and_device_to_loader(self):
        destination = self.install()
        loader = mock.Mock(side_effect=RuntimeError('stop before model allocation'))
        modules = {'numpy': SimpleNamespace(), 'PIL': SimpleNamespace(Image=None),
                   'torch': SimpleNamespace(), 'src.utils.checkpoint': SimpleNamespace(load_model=loader),
                   'src.datasets.codec': SimpleNamespace(build_codec=None)}
        sample = Path(download.__file__).resolve().parents[1] / 'sample.py'
        with mock.patch.dict(sys.modules, modules), \
                mock.patch('src.utils.config.sampling_recipe', return_value={}), \
                mock.patch.object(sys, 'argv', ['sample.py', '--checkpoint', 'toy', '--checkpoint-dir',
                    str(self.root / 'checkpoints'), '--output', str(self.root / 'samples'), '--device', 'cpu']), \
                self.assertRaisesRegex(RuntimeError, 'stop before model allocation'):
            runpy.run_path(str(sample), run_name='__main__')
        loader.assert_called_once_with(destination, 'cpu')

    def test_interrupted_part_resumes_and_verifies(self):
        path = self.root / 'weights.part000'
        path.with_name(path.name + '.partial').write_bytes(self.payload[:9000])
        fetch(self.url, path, self.expected)
        self.assertEqual(self.requests[0]['range'], 'bytes=9000-')
        self.assertEqual(path.read_bytes(), self.payload)
        self.assertFalse(path.with_name(path.name + '.partial').exists())

    def test_redirect_does_not_send_github_credentials_to_asset_host(self):
        url = self.url

        class Redirect(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header('Location', url)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        fetch(f'http://127.0.0.1:{server.server_port}/asset', self.root/'part',
              self.expected, {'Authorization': 'Bearer test-only-credential'})
        self.assertIsNone(self.requests[0]['authorization'])

    def test_corruption_cannot_publish_a_part_or_replace_existing_weights(self):
        part = self.root / 'part'
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            fetch(self.url, part, self.expected | dict(sha256='0' * 64))
        self.assertFalse(part.exists())
        first, second = self.root/'first', self.root/'second'
        first.write_bytes(self.payload[:10000])
        second.write_bytes(self.payload[10000:])
        weights = self.root/'model.safetensors'
        assemble([first, second], weights, self.expected)
        self.assertEqual(weights.read_bytes(), self.payload)
        second.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            assemble([first, second], weights, self.expected)
        self.assertEqual(weights.read_bytes(), self.payload)


if __name__ == '__main__':
    unittest.main()
