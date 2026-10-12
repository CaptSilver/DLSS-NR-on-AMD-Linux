"""Update-time settings preservation and supported GPU targets (version pins live in test_upstream_060)."""
import unittest
from dlssnr import upstream, deploy

class UpdateBehaviour050(unittest.TestCase):
    def test_preserve_settings_on_update(self):
        config=b'[DlssNrOnAmd]\nQuality=reference\nOverlayKey=F8\nScale=0.04\nLocalStructure=0.3\n'
        out=deploy._ini(config,0,update=True)
        for line in config.splitlines()[1:]:self.assertIn(line+b'\n',out)
        self.assertEqual(deploy._ini(out,0,update=True),out)

    def test_amd_handheld_support_not_silently_enabled(self):
        from dlssnr import cli
        self.assertEqual(cli.TARGETS,frozenset(('gfx1100','gfx1101','gfx1102','gfx1200','gfx1201')))

if __name__=='__main__':unittest.main()
