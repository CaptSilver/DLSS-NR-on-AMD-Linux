"""The vkd3d patch stack: pinned digests, fetch order, packaging, and the shared-buffer hunk."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / 'sources/vkd3d-proton-shared-buffers.patch'


class Vkd3dPatchStack(unittest.TestCase):
    def test_patch_digests_match_provenance(self):
        v = json.loads((ROOT / 'PROVENANCE.json').read_text())['vkd3d']
        for patch, key in (('sources/vkd3d-proton-ordered.patch', 'patch_sha256'),
                           (v['shared_buffers_patch'], 'shared_buffers_patch_sha256')):
            with self.subTest(patch=patch):
                self.assertEqual(hashlib.sha256((ROOT / patch).read_bytes()).hexdigest(), v[key])

    def test_fetch_applies_ordered_patch_before_shared_buffers(self):
        fetch = (ROOT / 'sources/fetch_vkd3d.py').read_text()
        self.assertLess(fetch.index('vkd3d-proton-ordered.patch'),
                        fetch.index('vkd3d-proton-shared-buffers.patch'))

    def test_shared_buffers_patch_is_packaged(self):
        release = (ROOT / 'build_release.py').read_text()
        self.assertIn("'sources/vkd3d-proton-shared-buffers.patch'", release)

    def test_committed_buffer_branch_opens_kmt_export_only_for_shared_heaps(self):
        added = [line[1:] for line in SHARED.read_text().splitlines()
                 if line.startswith('+') and not line.startswith('+++')]
        body = '\n'.join(added)
        self.assertRegex(body, r'if \(heap_flags & D3D12_HEAP_FLAG_SHARED\)\s*\n\s*'
                               r'd3d12_resource_open_export_kmt\(object, device, &object->mem\);')
        self.assertEqual(len(re.findall(r'd3d12_resource_open_export_kmt', body)), 1)


if __name__ == '__main__':
    unittest.main()
