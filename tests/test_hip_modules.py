"""Module-API, query and lifecycle forwarders plus the device-name override: wrapper bodies on CPU."""
from pathlib import Path
import os
import re
import shutil
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bridge_060 import compile_and_run
from dlssnr import upstream

ROOT = Path(__file__).resolve().parents[1]
PE_SOURCE = ROOT / 'sources/trampoline/amdhip64_7_pe.c'
# Every call the bridge adds beyond the upstream import surface. hipModuleLoad
# is PE-only: it reads the file itself and reuses hipModuleLoadData.
TABLE_CALLS = ('hipInit', 'hipMemGetInfo', 'hipEventDestroy', 'hipGetErrorName', 'hipStreamCreate',
    'hipHostFree', 'hipModuleLoadData', 'hipModuleGetFunction', 'hipModuleLaunchKernel',
    'hipModuleUnload', 'hipStreamQuery', 'hipMemcpy2DAsync', 'hipExtModuleLaunchKernel',
    'hipDeviceGetName')


K32_FAKES = r'''
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
HANDLE __stdcall CreateFileA(const char *name, DWORD acc, DWORD share, void *sa, DWORD disp, DWORD attr,
                             HANDLE tmpl) {
    int fd = disp == 3 ? open(name, O_RDONLY) : -1;
    (void)acc; (void)share; (void)sa; (void)attr; (void)tmpl;
    return fd < 0 ? INVALID_HANDLE_VALUE : (HANDLE)(long long)(fd + 1);
}
DWORD __stdcall GetFileSize(HANDLE h, DWORD *high) {
    struct stat st; (void)high;
    return fstat((int)(long long)h - 1, &st) ? 0xFFFFFFFFu : (DWORD)st.st_size;
}
BOOL __stdcall ReadFile(HANDLE h, void *buf, DWORD n, DWORD *got, void *ov) {
    ssize_t r = read((int)(long long)h - 1, buf, n); (void)ov;
    if (r < 0) return 0;
    *got = (DWORD)r; return 1;
}
BOOL __stdcall CloseHandle(HANDLE h) { close((int)(long long)h - 1); return 1; }
static void *live; static int allocs, frees;
void *__stdcall VirtualAlloc(void *addr, unsigned long long size, DWORD type, DWORD prot) {
    (void)addr; (void)type; (void)prot; allocs++; live = malloc(size); return live;
}
BOOL __stdcall VirtualFree(void *p, unsigned long long size, DWORD type) {
    (void)size; (void)type; frees++; if (p == live) live = 0; free(p); return 1;
}

'''


def pe_source():
    src = PE_SOURCE.read_text().replace(
        '#define EXPORT __declspec(dllexport)', '#define EXPORT __attribute__((ms_abi))')
    return '#define __declspec(x)\n#define __stdcall __attribute__((ms_abi))\n' + src


class ExportSurface(unittest.TestCase):
    def test_extra_exports_are_the_new_calls_and_module_load(self):
        self.assertEqual(upstream.HIP_EXTRA_EXPORTS,
                         frozenset(TABLE_CALLS) | {'hipModuleLoad'} | upstream.HIP_FENCE_EXPORTS)
        self.assertEqual(upstream.HIP_EXPORTS, upstream.HIP_IMPORTS | upstream.HIP_EXTRA_EXPORTS)
        self.assertFalse(upstream.HIP_EXTRA_EXPORTS & upstream.HIP_IMPORTS)

    def test_table_has_a_slot_for_each_call(self):
        header = (ROOT / 'native/hip_bridge.h').read_text()
        for name in TABLE_CALLS:
            with self.subTest(name=name):
                self.assertRegex(header, r'\(\*p_%s\)\(' % name)


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class PeForwarders(unittest.TestCase):
    def test_wrappers_forward_arguments_results_and_reject_old_table(self):
        src = pe_source() + K32_FAKES + r'''
#include <assert.h>
static int calls;
static int SYSV f_init(unsigned f) { calls++; assert(f == 0x5u); return 11; }
static int SYSV f_meminfo(u64 *fr, u64 *tot) { calls++; *fr = 0x100000001ULL; *tot = 0x200000002ULL; return 12; }
static int SYSV f_evdestroy(void *e) { calls++; assert(e == (void *)0x1111); return 13; }
static const char *SYSV f_errname(int e) { calls++; assert(e == 701); return "hipErrorLaunchFailure"; }
static int SYSV f_streamcreate(void **s) { calls++; *s = (void *)0x2222; return 14; }
static int SYSV f_hostfree(void *p) { calls++; assert(p == (void *)0x3333); return 15; }
static int SYSV f_loaddata(void **m, const void *img) {
    calls++; assert(img == (const void *)0x4444); *m = (void *)0x5555; return 16;
}
static int SYSV f_getfn(void **fn, void *m, const char *name) {
    calls++; assert(m == (void *)0x5555 && name == (const char *)0x6666); *fn = (void *)0x7777; return 17;
}
static int SYSV f_launch(void *fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
        unsigned bz, unsigned shared, void *stream, void **params, void **extra) {
    calls++;
    assert(fn == (void *)0x7777 && gx == 1 && gy == 2 && gz == 3 && bx == 4 && by == 5 && bz == 6);
    assert(shared == 0xdeadbeefu && stream == (void *)0x8888);
    assert(params == (void **)0x9999 && extra == (void **)0xaaaa);
    return 18;
}
static int SYSV f_unload(void *m) { calls++; assert(m == (void *)0x5555); return 19; }
static int SYSV f_query(void *s) { calls++; assert(s == (void *)0x8888); return 20; }
static int SYSV f_cpy2d(void *dst, u64 dp, const void *src, u64 sp, u64 w, u64 h, int kind, void *s) {
    calls++;
    assert(dst == (void *)0x10 && dp == 0x100000001ULL && src == (void *)0x20 && sp == 0x200000002ULL);
    assert(w == 0x300000003ULL && h == 0x400000004ULL && kind == 3 && s == (void *)0x8888);
    return 21;
}
static int SYSV f_extlaunch(void *fn, unsigned gx, unsigned gy, unsigned gz, unsigned lx, unsigned ly,
        unsigned lz, u64 shared, void *stream, void **params, void **extra, void *start, void *stop,
        unsigned flags) {
    calls++;
    assert(fn == (void *)0x7777 && gx == 7 && gy == 8 && gz == 9 && lx == 1 && ly == 2 && lz == 3);
    assert(shared == 0x500000005ULL && stream == (void *)0x8888 && params == (void **)0x9999);
    assert(extra == (void **)0xaaaa && start == (void *)0xb1 && stop == (void *)0xb2 && flags == 0x42);
    return 22;
}
static int SYSV f_devname(char *name, int len, int dev) { calls++; assert(len == 64 && dev == 1); name[0] = 'x'; return 23; }

int main(void) {
    DlssnrHipBridge t = {0}; g = &t;
    u64 fr = 0, tot = 0; void *h = 0, *fn = 0; char name[64] = {0};
    t.p_hipInit = f_init; t.p_hipMemGetInfo = f_meminfo; t.p_hipEventDestroy = f_evdestroy;
    t.p_hipGetErrorName = f_errname; t.p_hipStreamCreate = f_streamcreate; t.p_hipHostFree = f_hostfree;
    t.p_hipModuleLoadData = f_loaddata; t.p_hipModuleGetFunction = f_getfn;
    t.p_hipModuleLaunchKernel = f_launch; t.p_hipModuleUnload = f_unload; t.p_hipStreamQuery = f_query;
    t.p_hipMemcpy2DAsync = f_cpy2d; t.p_hipExtModuleLaunchKernel = f_extlaunch;
    t.p_hipDeviceGetName = f_devname;

    t.magic = 0x374849504E524C44ULL; /* previous table revision */
    assert(hipInit(5) == 3 && hipHostFree((void *)0x3333) == 3 && hipStreamQuery((void *)0x8888) == 3);
    assert(!strcmp(hipGetErrorName(701), "hip bridge not initialized") && !calls);

    t.magic = DLSSNR_HIP_MAGIC;
    assert(hipInit(5) == 11);
    assert(hipMemGetInfo(&fr, &tot) == 12 && fr == 0x100000001ULL && tot == 0x200000002ULL);
    assert(hipEventDestroy((void *)0x1111) == 13);
    assert(!strcmp(hipGetErrorName(701), "hipErrorLaunchFailure"));
    assert(hipStreamCreate(&h) == 14 && h == (void *)0x2222);
    assert(hipHostFree((void *)0x3333) == 15);
    assert(hipModuleLoadData(&h, (const void *)0x4444) == 16 && h == (void *)0x5555);
    assert(hipModuleGetFunction(&fn, h, (const char *)0x6666) == 17 && fn == (void *)0x7777);
    assert(hipModuleLaunchKernel(fn, 1, 2, 3, 4, 5, 6, 0xdeadbeefu, (void *)0x8888,
        (void **)0x9999, (void **)0xaaaa) == 18);
    assert(hipModuleUnload(h) == 19);
    assert(hipStreamQuery((void *)0x8888) == 20);
    assert(hipMemcpy2DAsync((void *)0x10, 0x100000001ULL, (void *)0x20, 0x200000002ULL,
        0x300000003ULL, 0x400000004ULL, 3, (void *)0x8888) == 21);
    assert(hipExtModuleLaunchKernel(fn, 7, 8, 9, 1, 2, 3, 0x500000005ULL, (void *)0x8888,
        (void **)0x9999, (void **)0xaaaa, (void *)0xb1, (void *)0xb2, 0x42) == 22);
    assert(hipDeviceGetName(name, 64, 1) == 23 && name[0] == 'x');
    assert(calls == 14);

    /* A missing slot is a not-initialized error, never a crash. */
    memset(&t, 0, sizeof t); t.magic = DLSSNR_HIP_MAGIC;
    assert(hipInit(0) == 3 && hipMemGetInfo(&fr, &tot) == 3 && hipEventDestroy(0) == 3);
    assert(hipStreamCreate(&h) == 3 && hipHostFree(0) == 3 && hipModuleLoadData(&h, 0) == 3);
    assert(hipModuleGetFunction(&fn, 0, 0) == 3 && hipModuleUnload(0) == 3 && hipStreamQuery(0) == 3);
    assert(hipModuleLaunchKernel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == 3);
    assert(hipMemcpy2DAsync(0, 0, 0, 0, 0, 0, 0, 0) == 3);
    assert(hipExtModuleLaunchKernel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == 3);
    assert(hipDeviceGetName(name, 64, 0) == 3);
    assert(!strcmp(hipGetErrorName(1), "hip bridge not initialized"));
    g = 0; assert(hipInit(0) == 3);
    return 0;
}
'''
        compile_and_run('#include <string.h>\n' + src, ROOT / 'sources/trampoline')

    def test_module_load_reads_the_file_and_hands_the_image_to_load_data(self):
        # hipModuleLoad gets a Windows path the Linux runtime cannot open, so the
        # PE side reads the code object and the bridge only sees the bytes. The
        # buffer must outlive the load until hipModuleUnload.
        src = pe_source() + K32_FAKES + r'''
#include <assert.h>
static const char *path; static int result, unloads, next_handle = 0x5555;
static const void *seen_image;
static int SYSV load_data(void **m, const void *img) {
    char want[8] = "\x7f" "ELFcode";
    assert(!memcmp(img, want, 8)); seen_image = img;
    *m = (void *)(long long)next_handle++; return result;
}
static int frees_at_unload;
static int SYSV unload(void *m) { (void)m; unloads++; frees_at_unload = frees; return 77; }
int main(int argc, char **argv) {
    DlssnrHipBridge t = {0}; g = &t; void *m = 0, *mods[64]; (void)argc; path = argv[1];
    t.magic = DLSSNR_HIP_MAGIC;
    assert(hipModuleLoad(&m, path) == 3 && !m && !allocs); /* no load-data slot */
    t.p_hipModuleLoadData = load_data; t.p_hipModuleUnload = unload;

    /* The code object stays allocated while the module is loaded. */
    assert(hipModuleLoad(&m, path) == 0 && m == (void *)0x5555 && allocs == 1 && frees == 0);
    assert(live == seen_image && !memcmp(live, "\x7f" "ELFcode", 8));
    assert(hipModuleUnload(m) == 77 && unloads == 1 && !live);
    assert(frees_at_unload == 0 && frees == 1); /* forwarded first, released after */
    assert(hipModuleUnload(m) == 77 && frees == 1); /* nothing recorded the second time */

    /* A failed load frees at once and leaves nothing to unload. */
    result = 200; m = 0;
    assert(hipModuleLoad(&m, path) == 200 && allocs == 2 && frees == 2 && !live);
    result = 0;
    assert(hipModuleLoad(&m, "/nonexistent/module.hsaco") == 301 && allocs == 2);
    assert(hipModuleLoad(0, path) == 1 && hipModuleLoad(&m, 0) == 1 && allocs == 2);
    assert(hipModuleLoad(&m, argv[2]) == 1 && allocs == 2 && frees == 2); /* empty file */

    /* Each module owns its buffer; a full table refuses the load, not leaks. */
    for (int i = 0; i < 64; i++) assert(hipModuleLoad(&mods[i], path) == 0);
    assert(allocs == 66 && frees == 2);
    assert(hipModuleLoad(&m, path) == 2 && allocs == 66);
    for (int i = 63; i >= 0; i--) hipModuleUnload(mods[i]);
    assert(frees == 66);
    assert(hipModuleLoad(&m, path) == 0 && hipModuleUnload(m) == 77 && frees == 67);

    t.magic = 0x354849504E524C44ULL;
    assert(hipModuleLoad(&m, path) == 3 && hipModuleUnload(m) == 3);
    return 0;
}
'''
        def run(d, exe):
            import subprocess
            (d / 'k.hsaco').write_bytes(b'\x7fELFcode')
            (d / 'empty.hsaco').write_bytes(b'')
            subprocess.run([exe, str(d / 'k.hsaco'), str(d / 'empty.hsaco')], check=True,
                capture_output=True, timeout=15)
        compile_and_run('#include <string.h>\n' + src, ROOT / 'sources/trampoline', run=run)

    def test_source_exports_exactly_the_export_surface(self):
        names = set(re.findall(r'^EXPORT\s[^(\n]*?\b(_{0,2}hip\w+)\s*\(', PE_SOURCE.read_text(), re.M))
        self.assertEqual(names, set(upstream.HIP_EXPORTS))


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class NativeForwarders(unittest.TestCase):
    def test_stubs_forward_arguments_results_and_missing_runtime_calls(self):
        src = r'''
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
static int calls, result;
static int r_init(unsigned f) { calls++; assert(f == 5); return result; }
static int r_meminfo(u64 *fr, u64 *tot) { calls++; *fr = 7; *tot = 9; return result; }
static int r_evdestroy(void *e) { calls++; assert(e == (void *)0x1111); return result; }
static const char *r_errname(int e) { calls++; assert(e == 701); return "hipErrorLaunchFailure"; }
static int r_streamcreate(void **s) { calls++; *s = (void *)0x2222; return result; }
static int r_hostfree(void *p) { calls++; assert(p == (void *)0x3333); return result; }
static int r_loaddata(void **m, const void *img) { calls++; assert(img == (void *)0x4444); *m = (void *)0x5555; return result; }
static const char kName[] = "kernel";
static int r_getfn(void **fn, void *m, const char *n) { calls++; assert(m == (void *)0x5555 && n == kName); *fn = (void *)0x7777; return result; }
static int r_launch(void *fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
        unsigned sh, void *st, void **p, void **x) {
    calls++;
    assert(fn == (void *)0x7777 && gx == 1 && gy == 2 && gz == 3 && bx == 4 && by == 5 && bz == 6);
    assert(sh == 0xdeadbeefu && st == (void *)0x8888 && p == (void **)0x9999 && x == (void **)0xaaaa);
    return result;
}
static int r_unload(void *m) { calls++; assert(m == (void *)0x5555); return result; }
static int r_query(void *s) { calls++; assert(s == (void *)0x8888); return result; }
static int r_cpy2d(void *d, u64 dp, const void *s, u64 sp, u64 w, u64 h, int k, void *st) {
    calls++; assert(d == (void *)0x10 && dp == 1 && s == (void *)0x20 && sp == 2 && w == 3 && h == 4 && k == 3 && st == (void *)0x8888);
    return result;
}
static int r_ext(void *fn, unsigned gx, unsigned gy, unsigned gz, unsigned lx, unsigned ly, unsigned lz,
        u64 sh, void *st, void **p, void **x, void *a, void *b, unsigned fl) {
    calls++;
    assert(fn == (void *)0x7777 && gx == 7 && gy == 8 && gz == 9 && lx == 1 && ly == 2 && lz == 3);
    assert(sh == 0x500000005ULL && st == (void *)0x8888 && p == (void **)0x9999 && x == (void **)0xaaaa);
    assert(a == (void *)0xb1 && b == (void *)0xb2 && fl == 0x42);
    return result;
}
static int r_devname(char *n, int len, int dev) { calls++; assert(len == 16 && dev == 2); strcpy(n, "Real GPU"); return result; }

int main(void) {
    g_hip_ok = 1;
    real.p_hipInit = r_init; real.p_hipMemGetInfo = r_meminfo; real.p_hipEventDestroy = r_evdestroy;
    real.p_hipGetErrorName = r_errname; real.p_hipStreamCreate = r_streamcreate;
    real.p_hipHostFree = r_hostfree; real.p_hipModuleLoadData = r_loaddata;
    real.p_hipModuleGetFunction = r_getfn; real.p_hipModuleLaunchKernel = r_launch;
    real.p_hipModuleUnload = r_unload; real.p_hipStreamQuery = r_query;
    real.p_hipMemcpy2DAsync = r_cpy2d; real.p_hipExtModuleLaunchKernel = r_ext;
    real.p_hipDeviceGetName = r_devname;
    assert(g_bridge.magic == 0x384849504E524C44ULL);
    u64 fr = 0, tot = 0; void *h = 0, *fn = 0; char name[16] = {0};
    for (result = 0; result <= 719; result += 719) {
        calls = 0;
        assert(g_bridge.p_hipInit(5) == result);
        assert(g_bridge.p_hipMemGetInfo(&fr, &tot) == result && fr == 7 && tot == 9);
        assert(g_bridge.p_hipEventDestroy((void *)0x1111) == result);
        assert(!strcmp(g_bridge.p_hipGetErrorName(701), "hipErrorLaunchFailure"));
        assert(g_bridge.p_hipStreamCreate(&h) == result && h == (void *)0x2222);
        assert(g_bridge.p_hipHostFree((void *)0x3333) == result);
        assert(g_bridge.p_hipModuleLoadData(&h, (void *)0x4444) == result && h == (void *)0x5555);
        assert(g_bridge.p_hipModuleGetFunction(&fn, h, kName) == result && fn == (void *)0x7777);
        assert(g_bridge.p_hipModuleLaunchKernel(fn, 1, 2, 3, 4, 5, 6, 0xdeadbeefu, (void *)0x8888,
            (void **)0x9999, (void **)0xaaaa) == result);
        assert(g_bridge.p_hipModuleUnload(h) == result);
        assert(g_bridge.p_hipStreamQuery((void *)0x8888) == result);
        assert(g_bridge.p_hipMemcpy2DAsync((void *)0x10, 1, (void *)0x20, 2, 3, 4, 3, (void *)0x8888) == result);
        assert(g_bridge.p_hipExtModuleLaunchKernel(fn, 7, 8, 9, 1, 2, 3, 0x500000005ULL, (void *)0x8888,
            (void **)0x9999, (void **)0xaaaa, (void *)0xb1, (void *)0xb2, 0x42) == result);
        assert(g_bridge.p_hipDeviceGetName(name, 16, 2) == result && !strcmp(name, "Real GPU"));
        assert(calls == 14);
    }
    memset(&real, 0, sizeof real);
    assert(g_bridge.p_hipInit(0) == 3 && g_bridge.p_hipMemGetInfo(&fr, &tot) == 3);
    assert(g_bridge.p_hipEventDestroy(0) == 3 && g_bridge.p_hipStreamCreate(&h) == 3);
    assert(g_bridge.p_hipHostFree(0) == 3 && g_bridge.p_hipModuleLoadData(&h, 0) == 3);
    assert(g_bridge.p_hipModuleGetFunction(&fn, 0, 0) == 3 && g_bridge.p_hipModuleUnload(0) == 3);
    assert(g_bridge.p_hipStreamQuery(0) == 3);
    assert(g_bridge.p_hipModuleLaunchKernel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == 3);
    assert(g_bridge.p_hipMemcpy2DAsync(0, 0, 0, 0, 0, 0, 0, 0) == 3);
    assert(g_bridge.p_hipExtModuleLaunchKernel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == 3);
    assert(g_bridge.p_hipDeviceGetName(name, 16, 0) == 3);
    assert(strlen(g_bridge.p_hipGetErrorName(1)) > 0);
    return 0;
}
'''
        compile_and_run(src, ROOT, ('-ldl',))

    def test_calls_resolve_from_the_selected_library(self):
        # The test above fills real.p_* by hand; this goes through ensure_hip(),
        # so a dropped dlsym leaves the slot null and the stub returns 3.
        names = set(re.findall(r'must_dlsym\("([^"]+)"\)', (ROOT / 'native/hip_bridge.c').read_text()))
        self.assertTrue(set(TABLE_CALLS) <= names)
        lib_src = ''.join('int %s(void) { return 0; }\n' % n
                          for n in sorted(names - set(TABLE_CALLS))) + ''.join(
            'int %s(void) { return %d; }\n' % (n, 1000 + i)
            for i, n in enumerate(TABLE_CALLS) if n != 'hipGetErrorName') + \
            'const char *hipGetErrorName(int e) { return "resolved"; }\n'
        calls = ', '.join('g_bridge.p_%s(%s)' % (n, ', '.join(['0'] * a)) for n, a in (
            ('hipInit', 1), ('hipMemGetInfo', 2), ('hipEventDestroy', 1), ('hipStreamCreate', 1),
            ('hipHostFree', 1), ('hipModuleLoadData', 2), ('hipModuleGetFunction', 3),
            ('hipModuleLaunchKernel', 11), ('hipModuleUnload', 1), ('hipStreamQuery', 1),
            ('hipMemcpy2DAsync', 8), ('hipExtModuleLaunchKernel', 14), ('hipDeviceGetName', 3)))
        expect = [1000 + TABLE_CALLS.index(n) for n in (
            'hipInit', 'hipMemGetInfo', 'hipEventDestroy', 'hipStreamCreate', 'hipHostFree',
            'hipModuleLoadData', 'hipModuleGetFunction', 'hipModuleLaunchKernel', 'hipModuleUnload',
            'hipStreamQuery', 'hipMemcpy2DAsync', 'hipExtModuleLaunchKernel', 'hipDeviceGetName')]
        src = r'''
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
int main(int argc, char **argv) {
    int want[] = {%s};
    assert(argc == 2);
    setenv("DLSSNR_HIP_LIBRARY", argv[1], 1);
    /* the first call opens the library; the slots below are read afterwards */
    assert(!strcmp(g_bridge.p_hipGetErrorName(1), "resolved"));
    int got[] = {%s};
    for (unsigned i = 0; i < sizeof got / sizeof *got; i++) { if (got[i] != want[i]) fprintf(stderr, "slot %%u got %%d want %%d\n", i, got[i], want[i]); assert(got[i] == want[i]); }
    return 0;
}
''' % (', '.join(map(str, expect)), calls)

        def run(d, exe):
            import subprocess
            (d / 'fakehip.c').write_text(lib_src)
            lib = d / 'libfakehip.so'
            subprocess.run(['gcc', '-shared', '-fPIC', '-o', str(lib), str(d / 'fakehip.c')],
                check=True, capture_output=True, text=True, timeout=60)
            env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0', DLSSNR_HIP_LOG=str(d / 'hip.log'))
            subprocess.run([exe, str(lib)], check=True, capture_output=True, timeout=15, env=env)
        compile_and_run(src, ROOT, ('-ldl',), run=run)


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class DeviceNameOverride(unittest.TestCase):
    def test_env_name_replaces_a_valid_devices_name_and_fits_the_buffer(self):
        src = r"""
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
static int calls;
/* Only devices 0 and 1 exist; anything else is hipErrorInvalidDevice. */
static int r_devname(char *n, int len, int dev) {
    calls++;
    if (dev < 0 || dev > 1) return 101;
    if (!n || len <= 0) return 1;
    snprintf(n, len, "Real GPU"); return 0;
}
int main(void) {
    char buf[32]; g_hip_ok = 1; real.p_hipDeviceGetName = r_devname;
    unsetenv("DLSSNR_HIP_DEVICE_NAME");
    assert(g_bridge.p_hipDeviceGetName(buf, sizeof buf, 0) == 0 && !strcmp(buf, "Real GPU") && calls == 1);
    setenv("DLSSNR_HIP_DEVICE_NAME", "", 1); /* empty means unset */
    assert(g_bridge.p_hipDeviceGetName(buf, sizeof buf, 0) == 0 && !strcmp(buf, "Real GPU") && calls == 2);

    setenv("DLSSNR_HIP_DEVICE_NAME", "AMD Radeon RX 9070 XT", 1);
    memset(buf, 'z', sizeof buf);
    assert(g_bridge.p_hipDeviceGetName(buf, sizeof buf, 1) == 0 && !strcmp(buf, "AMD Radeon RX 9070 XT"));
    assert(calls == 3);
    /* the override never hides an invalid device or its error code */
    memset(buf, 'z', sizeof buf);
    assert(g_bridge.p_hipDeviceGetName(buf, sizeof buf, 7) == 101 && buf[0] == 'z');
    assert(g_bridge.p_hipDeviceGetName(buf, sizeof buf, -1) == 101);

    /* exact fit, one short, one byte, nothing */
    memset(buf, 'z', sizeof buf);
    assert(g_bridge.p_hipDeviceGetName(buf, 22, 0) == 0 && !strcmp(buf, "AMD Radeon RX 9070 XT") && buf[22] == 'z');
    memset(buf, 'z', sizeof buf);
    assert(g_bridge.p_hipDeviceGetName(buf, 21, 0) == 0 && !strcmp(buf, "AMD Radeon RX 9070 X") && buf[21] == 'z');
    memset(buf, 'z', sizeof buf);
    assert(g_bridge.p_hipDeviceGetName(buf, 1, 0) == 0 && buf[0] == 0 && buf[1] == 'z');
    memset(buf, 'z', sizeof buf);
    assert(g_bridge.p_hipDeviceGetName(buf, 0, 0) == 1 && buf[0] == 'z');
    assert(g_bridge.p_hipDeviceGetName(NULL, 8, 0) == 1);

    /* a cut must not leave half of a multi-byte character */
    setenv("DLSSNR_HIP_DEVICE_NAME", "ab\xc3\xa9", 1);
    assert(g_bridge.p_hipDeviceGetName(buf, 4, 0) == 0 && !strcmp(buf, "ab"));
    assert(g_bridge.p_hipDeviceGetName(buf, 5, 0) == 0 && !strcmp(buf, "ab\xc3\xa9"));

    /* without a runtime call there is no name to override */
    real.p_hipDeviceGetName = 0;
    assert(g_bridge.p_hipDeviceGetName(buf, sizeof buf, 0) == 3);
    return 0;
}
"""
        compile_and_run(src, ROOT, ('-ldl',))


if __name__ == '__main__':
    unittest.main()
