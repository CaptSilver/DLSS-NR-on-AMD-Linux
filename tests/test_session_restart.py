"""A restarted mod session on the same flag buffer must get fresh frame slots."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class SessionRestartRegression(unittest.TestCase):
    @unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
    def test_actual_bridge_session_restart_contract(self):
        with tempfile.TemporaryDirectory() as d:
            exe=Path(d)/'session-restart'
            subprocess.run(['gcc','-std=gnu11','-O1','-g','-pthread',
                '-Werror=incompatible-pointer-types','-fsanitize=address,undefined',
                '-fno-pie','-no-pie',str(ROOT/'tests/session_restart_contract.c'),
                '-ldl','-o',str(exe)],check=True,capture_output=True,timeout=30)
            for scenario in range(5):
                with self.subTest(scenario=scenario):
                    r=subprocess.run([str(exe),str(scenario)],capture_output=True,
                        text=True,timeout=10,
                        env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0'))
                    self.assertEqual(r.returncode,0,r.stdout+r.stderr)

if __name__=='__main__': unittest.main()
