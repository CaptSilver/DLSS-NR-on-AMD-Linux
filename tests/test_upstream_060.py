"""Version, pins and import surface for the 0.6.0 upstream runtime."""
from pathlib import Path
import hashlib
import json
import os
import re
import unittest
from dlssnr import upstream, assets, legacy_converter
from dlssnr.games import pe_info

ROOT = Path(__file__).resolve().parents[1]
NEW_IMPORTS = {'hipHostMalloc', 'hipHostGetDevicePointer'}


class Release060(unittest.TestCase):
    def test_pins(self):
        manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
        self.assertEqual(upstream.VERSION, '0.6.0')
        self.assertEqual(manifest['mod_version'], '0.6.0')
        self.assertEqual(manifest['version'], '0.6.0')
        self.assertEqual(upstream.SETUP_URL,
            'https://github.com/danielblnc/DLSS-NR-on-AMD/releases/download/v0.6.0/dlssnr_on_amd_setup.exe')
        self.assertEqual(upstream.SETUP_BYTES, 59841536)
        self.assertEqual(upstream.SETUP_SHA256,
            '20636c9587e858e2b35b29702ef36bcb0018a985592e24d6ca95b3e1b41d2e71')
        self.assertEqual(manifest['external_files']['dlssnr_on_amd_setup.exe'], upstream.SETUP_SHA256)
        self.assertEqual(upstream.PAYLOAD_OFFSET, 2498583)
        self.assertEqual(upstream.PAYLOAD_BYTES, 56677888)
        self.assertEqual(upstream.PAYLOAD_SHA256,
            '195c4a891b6eac4c1cb7671e10ff62bbbe2b17f1dfae1344dc5a6714e4775721')

    def test_wait_shader_pins_move_but_hashes_do_not(self):
        # Offsets shift with the new payload; the FNV hashes vkd3d matches on do not.
        self.assertEqual(upstream.FLAG_SHADER_OFFSET, 546736)
        self.assertEqual(upstream.FLAG_SHADER_HASH, '135ea1f88cbc832d')
        self.assertEqual(upstream.FLAG_SHADER_BYTES, 1472)
        self.assertEqual(upstream.FLAG_SHADER_SHA256,
            '0b85aab8db0ee8ba14019e32e3839bb2d321cde974371b30eb3742955b53084b')
        self.assertEqual(upstream.GRAPHICS_WAIT_SHADERS['pixel'][:2], (548208, 1152))
        self.assertEqual(upstream.GRAPHICS_WAIT_SHADERS['vertex'][:2], (549360, 616))
        self.assertEqual(upstream.GRAPHICS_WAIT_HASHES,
            {'pixel': 'f2010bee184ea0aa', 'vertex': '97c89ca9f5ead0f9'})

    def test_default_config_unchanged(self):
        self.assertEqual(hashlib.sha256(upstream.DEFAULT_CONFIG).hexdigest(),
            '30b6f49a8274798b43344db204f36750a1fdf6690d13edf5b0a43b6259e8372b')

    def test_host_memory_imports_are_the_whole_new_surface(self):
        self.assertEqual(len(upstream.HIP_IMPORTS), 40)
        self.assertTrue(NEW_IMPORTS <= upstream.HIP_IMPORTS)
        self.assertEqual(upstream.HIP_IMPORTS - legacy_converter.HIP_IMPORTS,
            {'hipGetDevice', 'hipOccupancyMaxActiveBlocksPerMultiprocessor',
             'hipDeviceGetStreamPriorityRange', 'hipStreamCreateWithPriority',
             'hipStreamDestroy'} | NEW_IMPORTS)

    def test_trampoline_source_exports_the_imports_and_the_extras(self):
        source = (ROOT / 'sources/trampoline/amdhip64_7_pe.c').read_text()
        names = set(re.findall(r'^EXPORT\s[^(\n]*?\b(_{0,2}hip\w+)\s*\(', source, re.M))
        self.assertEqual(names, set(upstream.HIP_EXPORTS))
        self.assertTrue(upstream.HIP_IMPORTS <= names)

    @unittest.skipUnless((ROOT / 'assets/amdhip64_7.dll').is_file(), 'built trampoline not present')
    def test_built_trampoline_exports_the_imports_and_the_extras(self):
        info = pe_info(ROOT / 'assets/amdhip64_7.dll')
        self.assertEqual(sorted(info['exports']), sorted(upstream.HIP_EXPORTS))
        self.assertEqual(len(info['exports']), len(upstream.HIP_EXPORTS))

    def test_bridge_abi_8_contract(self):
        manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
        self.assertEqual(manifest['hip_bridge_abi'], 8)
        header = (ROOT / 'native/hip_bridge.h').read_text()
        self.assertIn('0x384849504E524C44ULL', header)
        self.assertIn('"DLNRHIP8"', header)
        self.assertIn('DLSSNR_HIP_ABI_V8', (ROOT / 'native/hip_bridge.c').read_text())
        manifest['files'] = {name: '0' * 64 for name in upstream.COMPONENTS}
        assets.require_deployable(manifest)
        for old in (1, 2, 3, 4, 5, 6, 7):
            with self.subTest(old=old), self.assertRaises(RuntimeError):
                assets.require_deployable(dict(manifest, hip_bridge_abi=old))


@unittest.skipUnless(os.environ.get('DLSSNR_TEST_SETUP'), 'official current fixture required')
class Official060(unittest.TestCase):
    def test_real_setup_is_accepted_and_stale_offsets_are_not(self):
        data = Path(os.environ['DLSSNR_TEST_SETUP']).read_bytes()
        payload, config, report = upstream.inspect_setup(data)
        self.assertEqual(report['mod_version'], '0.6.0')
        self.assertEqual(payload[:2], b'MZ')
        self.assertEqual(config, upstream.DEFAULT_CONFIG)
        # The 0.5.0 flag-shader offset no longer lands on a shader in this payload.
        self.assertNotEqual(payload[522656:522660], b'DXBC')
        self.assertEqual(payload[upstream.FLAG_SHADER_OFFSET:upstream.FLAG_SHADER_OFFSET + 4], b'DXBC')


if __name__ == '__main__':
    unittest.main()
