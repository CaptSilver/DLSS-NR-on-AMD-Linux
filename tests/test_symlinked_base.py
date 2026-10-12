"""Base directories reached through a symlink, e.g. /home -> /var/home on ostree
systems. The user's own base (home, XDG_DATA_HOME, --data-dir) is resolved once;
anything the installer owns below it must still reject symlinks."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from dlssnr import cli, deploy, runtime

DIGEST = 'ab' * 32


class SymlinkedHome(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # realpath: the temp dir itself may sit behind a symlink on some hosts.
        self.root = Path(os.path.realpath(temporary.name))
        self.real_home = self.root / 'var/home/user'
        self.real_data = self.real_home / '.local/share'
        self.real_data.mkdir(parents=True)
        (self.root / 'home').symlink_to('var/home')
        self.home = self.root / 'home/user'
        self.elsewhere = self.root / 'elsewhere'
        self.elsewhere.mkdir()
        environ = mock.patch.dict(os.environ, {'HOME': str(self.home)})
        environ.start()
        self.addCleanup(environ.stop)
        os.environ.pop('XDG_DATA_HOME', None)

    def symlinked_xdg(self):
        real = self.root / 'xdg-real'
        real.mkdir()
        (self.root / 'xdg').symlink_to('xdg-real')
        os.environ['XDG_DATA_HOME'] = str(self.root / 'xdg')
        return real

    def data_dir(self, explicit=None):
        return cli.data_dir(argparse.Namespace(data_dir=explicit))

    def assertRejects(self, link, call, *args):
        with self.assertRaisesRegex(RuntimeError, 'Unsafe symlink or wrong path type: ' + re.escape(str(link)) + r'\Z'):
            call(*args)


class BridgeCache(SymlinkedHome):
    def test_cache_path_under_symlinked_home(self):
        self.assertEqual(deploy._cache_path(DIGEST), self.real_data / 'dlssnr-linux/native' / DIGEST / 'bridge.so')

    def test_cache_path_under_symlinked_xdg_data_home(self):
        real = self.symlinked_xdg()
        self.assertEqual(deploy._cache_path(DIGEST), real / 'dlssnr-linux/native' / DIGEST / 'bridge.so')

    def test_bridge_is_cached_under_symlinked_home(self):
        source = self.root / 'bridge.so'
        source.write_bytes(b'bridge')
        digest = hashlib.sha256(b'bridge').hexdigest()
        cache = deploy._cache_path(digest)
        deploy._cache(source, digest, cache)
        self.assertEqual((self.real_data / 'dlssnr-linux/native' / digest / 'bridge.so').read_bytes(), b'bridge')

    def test_relative_xdg_data_home_is_rejected(self):
        os.environ['XDG_DATA_HOME'] = 'relative/share'
        with self.assertRaisesRegex(RuntimeError, 'must be absolute'):
            deploy._cache_path(DIGEST)

    def test_symlinked_app_dir_is_rejected(self):
        link = self.real_data / 'dlssnr-linux'
        link.symlink_to(self.elsewhere)
        self.assertRejects(link, deploy._cache_path, DIGEST)

    def test_symlinked_digest_dir_is_rejected(self):
        (self.real_data / 'dlssnr-linux/native').mkdir(parents=True)
        link = self.real_data / 'dlssnr-linux/native' / DIGEST
        link.symlink_to(self.elsewhere)
        self.assertRejects(link, deploy._cache_path, DIGEST)


class DataDir(SymlinkedHome):
    def write_manifest(self, data_dir):
        runtime._write_json(data_dir / 'runtime-manifest.json', {'ok': True})

    def test_default_data_dir_under_symlinked_home(self):
        self.write_manifest(self.data_dir())
        self.assertTrue((self.real_data / 'dlssnr-linux/runtime-manifest.json').is_file())

    def test_data_dir_under_symlinked_xdg_data_home(self):
        real = self.symlinked_xdg()
        self.write_manifest(self.data_dir())
        self.assertTrue((real / 'dlssnr-linux/runtime-manifest.json').is_file())

    def test_explicit_data_dir_under_symlinked_home(self):
        self.write_manifest(self.data_dir(self.home / 'custom'))
        self.assertTrue((self.real_home / 'custom/runtime-manifest.json').is_file())

    def test_relative_explicit_data_dir_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'must be absolute'):
            self.data_dir(Path('relative/data'))

    def test_symlinked_app_dir_is_rejected(self):
        link = self.real_data / 'dlssnr-linux'
        link.symlink_to(self.elsewhere)
        self.assertRejects(link, self.write_manifest, self.data_dir())
        self.assertEqual(list(self.elsewhere.iterdir()), [])

    def test_symlinked_explicit_data_dir_is_rejected(self):
        # Like dlssnr-linux/ in the default layout, the named directory itself is
        # the installer's root; only its parents are the user's own layout.
        link = self.real_home / 'custom'
        link.symlink_to(self.elsewhere)
        self.assertRejects(link, self.write_manifest, self.data_dir(self.home / 'custom'))
        self.assertEqual(list(self.elsewhere.iterdir()), [])

    def test_symlinked_rocm_venv_is_rejected(self):
        (self.real_home / 'custom').mkdir()
        link = self.real_home / 'custom/rocm-venv'
        link.symlink_to(self.elsewhere)
        data_dir = self.data_dir(self.home / 'custom')
        self.assertRejects(link, runtime._write_json, data_dir / 'rocm-venv/wheel-provenance.json', {})
        self.assertEqual(list(self.elsewhere.iterdir()), [])


class GameDir(SymlinkedHome):
    def setUp(self):
        super().setUp()
        self.game = self.real_home / 'games/Game/Win64'
        self.game.mkdir(parents=True)
        self.exe = self.game / 'game.exe'
        self.exe.write_bytes(b'MZ')

    def test_symlinked_store_is_rejected(self):
        link = self.game / deploy.STORE
        link.symlink_to(self.elsewhere)
        self.assertRejects(link, deploy._load, self.exe)

    def test_symlinked_dll_target_is_not_written_through(self):
        victim = self.elsewhere / 'd3d12.dll'
        victim.write_bytes(b'original')
        link = self.game / 'd3d12.dll'
        link.symlink_to(victim)
        self.assertRejects(link, deploy._atomic_bytes, link, b'payload')
        self.assertEqual(victim.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
