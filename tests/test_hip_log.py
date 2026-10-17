"""The bridge log must never wait on a lock another thread can leave held."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bridge_060 import compile_and_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_logged(src, timeout=15):
    out = {}

    def run(d, exe):
        log = d / 'hip.log'
        env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0', DLSSNR_HIP_LOG=str(log))
        subprocess.run([exe], check=True, capture_output=True, timeout=timeout, env=env)
        out['log'] = log.read_text()
    compile_and_run(src, ROOT, ('-ldl',), run=run)
    return out['log']


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class BridgeLog(unittest.TestCase):
    def test_log_does_not_wait_on_a_stdio_lock_left_held(self):
        # Wine can suspend a thread in the middle of a log call; a stdio FILE lock
        # it held then stays taken and every later log call blocks forever.
        src = r'''
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
static void stuck_lock(FILE *f) { (void)f; for (;;) pause(); }
#define flockfile(f) stuck_lock(f)
#include "native/hip_bridge.c"
int main(void) {
    hip_log("first %d", 1);
    hip_log("second %s", "line");
    return 0;
}
'''
        log = run_logged(src, timeout=10)
        # The bridge's load-time constructor logs first; only the tail is ours.
        self.assertRegex(log, r'\[pid \d+\] first 1\n\[pid \d+\] second line\n$')

    def test_concurrent_lines_stay_whole_and_long_lines_are_cut(self):
        src = r'''
#define _GNU_SOURCE
#include <pthread.h>
#include <string.h>
#include "native/hip_bridge.c"
static void *worker(void *arg) {
    long id = (long)arg;
    for (int i = 0; i < 200; i++)
        hip_log("thread=%ld line=%d %s", id, i, "abcdefghijklmnopqrstuvwxyz");
    return 0;
}
int main(void) {
    pthread_t t[16];
    char big[8192];
    for (long i = 0; i < 16; i++)
        pthread_create(&t[i], 0, worker, (void *)i);
    for (int i = 0; i < 16; i++)
        pthread_join(t[i], 0);
    memset(big, 'x', sizeof(big) - 1);
    big[sizeof(big) - 1] = 0;
    hip_log("long %s", big);
    return 0;
}
'''
        log = run_logged(src)
        lines = log.split('\n')
        self.assertEqual(lines[-1], '')
        lines = lines[:-1]
        body = [l for l in lines if 'thread=' in l]
        self.assertEqual(len(body), 16 * 200)
        pattern = re.compile(r'^\[pid \d+\] thread=\d+ line=\d+ abcdefghijklmnopqrstuvwxyz$')
        self.assertTrue(all(pattern.match(l) for l in body))
        long_line = [l for l in lines if ' long ' in l]
        self.assertEqual(len(long_line), 1)
        self.assertTrue(long_line[0].startswith('[pid '))
        self.assertLess(len(long_line[0]), 8192)


if __name__ == '__main__':
    unittest.main()
