"""Host-memory pass-throughs added with the 0.6.0 runtime: production wrapper bodies on CPU."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]


def compile_and_run(source, include, extra=(), run=None):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'test.c').write_text(source)
        subprocess.run(['gcc', '-std=gnu11', '-O1', '-g', '-pthread', '-ffunction-sections',
            '-fdata-sections', '-Werror=incompatible-pointer-types',
            '-fsanitize=address,undefined', '-fno-pie', '-no-pie', '-I', str(include),
            str(d / 'test.c'), '-Wl,--gc-sections', *extra, '-o', str(d / 'test')],
            check=True, capture_output=True, text=True, timeout=60)
        if run:
            run(d, str(d / 'test'))
        else:
            subprocess.run([str(d / 'test')], check=True, capture_output=True, timeout=15)


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class HostMemoryBridge(unittest.TestCase):
    def test_pe_wrappers_forward_flags_pointers_errors_and_reject_old_table(self):
        src = (ROOT / 'sources/trampoline/amdhip64_7_pe.c').read_text().replace(
            '#define EXPORT __declspec(dllexport)', '#define EXPORT __attribute__((ms_abi))')
        src = '#define __declspec(x)\n#define __stdcall __attribute__((ms_abi))\n' + src
        src += r'''
#include <assert.h>
static int called, result;
static int SYSV host_malloc(void **out, u64 size, unsigned flags) {
    called++; assert(size == 0x123456789ULL && flags == 0x80000003u);
    if (!result && out) *out = (void *)0x7f0000001000ULL;
    return result;
}
static int SYSV host_devptr(void **dev, void *host, unsigned flags) {
    called++; assert(host == (void *)0x7f0000001000ULL && flags == 0x0u);
    if (!result && dev) *dev = (void *)0x7e0000002000ULL;
    return result;
}
int main(void) {
    DlssnrHipBridge t = {0}; g = &t; void *out = 0, *dev = 0;
    t.p_hipHostMalloc = host_malloc; t.p_hipHostGetDevicePointer = host_devptr;
    t.magic = 0x344849504E524C44ULL; /* previous table revision */
    assert(hipHostMalloc(&out, 0x123456789ULL, 0x80000003u) == 3 && !out);
    assert(hipHostGetDevicePointer(&dev, (void *)0x7f0000001000ULL, 0) == 3 && !dev && !called);
    t.magic = DLSSNR_HIP_MAGIC;
    assert(hipHostMalloc(&out, 0x123456789ULL, 0x80000003u) == 0 && out == (void *)0x7f0000001000ULL);
    assert(hipHostGetDevicePointer(&dev, out, 0) == 0 && dev == (void *)0x7e0000002000ULL);
    result = 801; out = 0; dev = 0;
    assert(hipHostMalloc(&out, 0x123456789ULL, 0x80000003u) == 801 && !out);
    assert(hipHostGetDevicePointer(&dev, (void *)0x7f0000001000ULL, 0) == 801 && !dev);
    t.p_hipHostMalloc = 0; t.p_hipHostGetDevicePointer = 0;
    assert(hipHostMalloc(&out, 1, 0) == 3 && hipHostGetDevicePointer(&dev, 0, 0) == 3);
    g = 0; assert(hipHostMalloc(&out, 1, 0) == 3);
    assert(called == 4); return 0;
}
'''
        compile_and_run(src, ROOT / 'sources/trampoline')

    def test_native_bridge_resolves_and_forwards_host_memory(self):
        src = r'''
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
static int result, called;
static int host_malloc(void **p, u64 size, unsigned flags) {
    called++; assert(size == 4096 && flags == 2);
    if (!result && p) *p = (void *)0x7f0000001000ULL;
    return result;
}
static int host_devptr(void **d, void *h, unsigned flags) {
    called++; assert(h == (void *)0x7f0000001000ULL && flags == 0);
    if (!result && d) *d = (void *)0x7e0000002000ULL;
    return result;
}
int main(void) {
    g_hip_ok = 1; real.p_hipHostMalloc = host_malloc; real.p_hipHostGetDevicePointer = host_devptr;
    void *p = 0, *d = 0;
    assert(g_bridge.magic == 0x354849504E524C44ULL);
    assert(g_bridge.p_hipHostMalloc && g_bridge.p_hipHostGetDevicePointer);
    assert(g_bridge.p_hipHostMalloc(&p, 4096, 2) == 0 && p == (void *)0x7f0000001000ULL);
    assert(g_bridge.p_hipHostGetDevicePointer(&d, p, 0) == 0 && d == (void *)0x7e0000002000ULL);
    result = 801; p = d = 0;
    assert(g_bridge.p_hipHostMalloc(&p, 4096, 2) == 801 && !p);
    assert(g_bridge.p_hipHostGetDevicePointer(&d, (void *)0x7f0000001000ULL, 0) == 801 && !d);
    real.p_hipHostMalloc = 0; real.p_hipHostGetDevicePointer = 0;
    assert(g_bridge.p_hipHostMalloc(&p, 1, 0) == 3 && g_bridge.p_hipHostGetDevicePointer(&d, 0, 0) == 3);
    assert(called == 4); return 0;
}
'''
        compile_and_run(src, ROOT, ('-ldl',))

    def test_native_bridge_loads_host_memory_symbols_from_selected_library(self):
        # The test above fills real.p_* by hand. This one goes through
        # ensure_hip(): a stand-in libamdhip64 exports every name the bridge
        # resolves, so dropping a must_dlsym() for the host-memory calls
        # leaves the pointers null and the stubs return 3.
        names = set(re.findall(r'must_dlsym\("([^"]+)"\)', (ROOT / 'native/hip_bridge.c').read_text()))
        host_calls = {'hipHostMalloc', 'hipHostGetDevicePointer'}
        lib_src = ''.join('int %s(void) { return 0; }\n' % n for n in sorted(names - host_calls)) + r'''
int hipHostMalloc(void **p, unsigned long size, unsigned flags) {
    if (size != 4096 || flags != 2) return 11;
    *p = (void *)0x7f0000001000ULL; return 0;
}
int hipHostGetDevicePointer(void **d, void *h, unsigned flags) {
    if (h != (void *)0x7f0000001000ULL || flags) return 12;
    *d = (void *)0x7e0000002000ULL; return 0;
}
'''
        src = r'''
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
int main(int argc, char **argv) {
    void *p = 0, *d = 0;
    assert(argc == 2 && !g_hip_ok);
    setenv("DLSSNR_HIP_LIBRARY", argv[1], 1);
    assert(g_bridge.p_hipHostMalloc(&p, 4096, 2) == 0 && p == (void *)0x7f0000001000ULL);
    assert(g_hip_ok && real.p_hipHostMalloc && real.p_hipHostGetDevicePointer);
    assert(g_bridge.p_hipHostGetDevicePointer(&d, p, 0) == 0 && d == (void *)0x7e0000002000ULL);
    assert(g_bridge.p_hipHostMalloc(&p, 1, 0) == 11);
    return 0;
}
'''

        def run(d, exe):
            (d / 'fakehip.c').write_text(lib_src)
            lib = d / 'libfakehip.so'
            subprocess.run(['gcc', '-shared', '-fPIC', '-o', str(lib), str(d / 'fakehip.c')],
                check=True, capture_output=True, text=True, timeout=60)
            env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0', DLSSNR_HIP_LOG=str(d / 'hip.log'))
            subprocess.run([exe, str(lib)], check=True, capture_output=True, timeout=15, env=env)
        compile_and_run(src, ROOT, ('-ldl',), run=run)


if __name__ == '__main__':
    unittest.main()
