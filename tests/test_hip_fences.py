"""D3D12 fence emulation: hipImportExternalSemaphore type 4 on top of DRM timeline syncobjs."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bridge_060 import compile_and_run
from test_hip_modules import pe_source
from dlssnr import upstream

ROOT = Path(__file__).resolve().parents[1]
FENCE_CALLS = ('hipImportExternalSemaphore', 'hipDestroyExternalSemaphore',
               'hipWaitExternalSemaphoresAsync', 'hipSignalExternalSemaphoresAsync')
ROCM_HIP = Path.home() / '.local/share/dlssnr-linux/rocm-venv/lib/python3.14/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7'
RENDER_NODE = Path(os.environ.get('DLSSNR_DRM_RENDER_NODE', '/dev/dri/renderD128'))


class ExportSurface(unittest.TestCase):
    def test_fence_calls_are_exports_with_a_table_slot_each(self):
        self.assertEqual(upstream.HIP_FENCE_EXPORTS, frozenset(FENCE_CALLS))
        self.assertTrue(upstream.HIP_FENCE_EXPORTS <= upstream.HIP_EXTRA_EXPORTS)
        self.assertFalse(upstream.HIP_FENCE_EXPORTS & upstream.HIP_IMPORTS)
        header = (ROOT / 'native/hip_bridge.h').read_text()
        for name in FENCE_CALLS:
            with self.subTest(name=name):
                self.assertRegex(header, r'\(\*p_%s\)\(' % name)

    def test_both_header_copies_agree(self):
        self.assertEqual((ROOT / 'native/hip_bridge.h').read_bytes(),
                         (ROOT / 'sources/trampoline/hip_bridge.h').read_bytes())


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class PeForwarders(unittest.TestCase):
    def test_wrappers_forward_arguments_results_and_reject_old_table(self):
        src = pe_source() + r'''
#include <assert.h>
static int calls;
static int SYSV f_import(void **out, const void *desc) {
    calls++; assert(desc == (const void *)0x1234); *out = (void *)0x5555; return 11;
}
static int SYSV f_destroy(void *sem) { calls++; assert(sem == (void *)0x5555); return 12; }
static int SYSV f_wait(const void *sems, const void *params, unsigned n, void *stream) {
    calls++; assert(sems == (const void *)0x10 && params == (const void *)0x20 && n == 3 && stream == (void *)0x8888);
    return 13;
}
static int SYSV f_signal(const void *sems, const void *params, unsigned n, void *stream) {
    calls++; assert(sems == (const void *)0x30 && params == (const void *)0x40 && n == 2 && stream == (void *)0x9999);
    return 14;
}
int main(void) {
    DlssnrHipBridge t = {0}; g = &t; void *sem = 0;
    t.p_hipImportExternalSemaphore = f_import; t.p_hipDestroyExternalSemaphore = f_destroy;
    t.p_hipWaitExternalSemaphoresAsync = f_wait; t.p_hipSignalExternalSemaphoresAsync = f_signal;
    t.magic = 0x364849504E524C44ULL; /* previous table revision */
    assert(hipImportExternalSemaphore(&sem, (void *)0x1234) == 3 && !sem && !calls);
    assert(hipDestroyExternalSemaphore((void *)1) == 3);
    assert(hipWaitExternalSemaphoresAsync((void *)1, (void *)1, 1, 0) == 3);
    assert(hipSignalExternalSemaphoresAsync((void *)1, (void *)1, 1, 0) == 3 && !calls);
    t.magic = DLSSNR_HIP_MAGIC;
    assert(hipImportExternalSemaphore(&sem, (void *)0x1234) == 11 && sem == (void *)0x5555);
    assert(hipDestroyExternalSemaphore(sem) == 12);
    assert(hipWaitExternalSemaphoresAsync((void *)0x10, (void *)0x20, 3, (void *)0x8888) == 13);
    assert(hipSignalExternalSemaphoresAsync((void *)0x30, (void *)0x40, 2, (void *)0x9999) == 14);
    assert(calls == 4);
    t.p_hipImportExternalSemaphore = 0; t.p_hipDestroyExternalSemaphore = 0;
    t.p_hipWaitExternalSemaphoresAsync = 0; t.p_hipSignalExternalSemaphoresAsync = 0;
    assert(hipImportExternalSemaphore(&sem, 0) == 3 && hipDestroyExternalSemaphore(sem) == 3);
    assert(hipWaitExternalSemaphoresAsync(0, 0, 0, 0) == 3 && hipSignalExternalSemaphoresAsync(0, 0, 0, 0) == 3);
    return 0;
}
'''
        compile_and_run('#include <string.h>\n' + src, ROOT / 'sources/trampoline')


# Fakes for the DRM syncobj ioctls and the HIP entry points the emulation calls. The
# timeline is a counter behind a condition variable; HIP calls assert they are not
# running inside a host callback, which is where HIP may not be re-entered.
FAKES = r'''
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
#include <errno.h>

static pthread_mutex_t tl_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t tl_cv = PTHREAD_COND_INITIALIZER;
static u64 tl_point;
static int node_fd = -1, imported_fd = -1, destroyed, node_opens, cap_checks;
static unsigned imported_handle;
static char opened_path[256];

static int d_open(const char *path) {
    node_opens++; snprintf(opened_path, sizeof opened_path, "%s", path);
    node_fd = open("/dev/null", O_RDWR | O_CLOEXEC); return node_fd;
}
static int d_cap(int drm) { (void)drm; cap_checks++; return 1; }
static int d_fd2h(int drm, int fd, unsigned *h) { assert(drm == node_fd); imported_fd = fd; *h = 77; imported_handle = 77; return 0; }
static int d_destroy(int drm, unsigned h) { assert(h == 77 && drm == node_fd); destroyed++; return 0; }
static long long mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
static int d_wait(int drm, unsigned h, u64 point, long long abs_ns) {
    struct timespec ts = { abs_ns / 1000000000LL, abs_ns % 1000000000LL };
    int r = 0; (void)drm; assert(h == 77);
    pthread_mutex_lock(&tl_mu);
    while (tl_point < point && !r) r = pthread_cond_timedwait(&tl_cv, &tl_mu, &ts);
    pthread_mutex_unlock(&tl_mu);
    return r ? -ETIME : 0;
}
static void tl_set(u64 p) { pthread_mutex_lock(&tl_mu); if (p > tl_point) tl_point = p; pthread_cond_broadcast(&tl_cv); pthread_mutex_unlock(&tl_mu); }
static int d_signal(int drm, unsigned h, u64 p) { (void)drm; assert(h == 77); tl_set(p); return 0; }
static int d_query(int drm, unsigned h, u64 *p) { (void)drm; assert(h == 77); pthread_mutex_lock(&tl_mu); *p = tl_point; pthread_mutex_unlock(&tl_mu); return 0; }
static int nt_fd_result;
static void *nt_seen;
static int d_nt(void *nt) {
    nt_seen = nt;
    return nt_fd_result ? nt_fd_result : open("/dev/null", O_RDONLY | O_CLOEXEC);
}

static __thread int in_cb;
static int wait_calls, hostfree_n, query_busy, query_calls, sync_ok = 1;
static void *wait_stream, *wait_ptr; static u64 wait_val, wait_mask; static unsigned wait_flags; static int wait_ret;
static void (*cb_fn)(void *); static void *cb_data, *cb_stream;
static u64 flag_at_free;
static int h_wait64(void *st, void *ptr, u64 v, unsigned fl, u64 mask) {
    assert(!in_cb); wait_calls++; wait_stream = st; wait_ptr = ptr; wait_val = v; wait_flags = fl; wait_mask = mask; return wait_ret;
}
static int h_hostfunc(void *st, void (*fn)(void *), void *d) { assert(!in_cb); cb_fn = fn; cb_data = d; cb_stream = st; return 0; }
static int h_hostmalloc(void **p, u64 sz, unsigned fl) {
    assert(!in_cb && sz >= 8 && (fl & 2) && (fl & 0x40000000u));
    *p = calloc(1, sz); return 0;
}
static int h_devptr(void **d, void *h, unsigned fl) { assert(!in_cb && !fl); *d = h; return 0; }
static int h_hostfree(void *p) { assert(!in_cb); hostfree_n++; flag_at_free = *(volatile u64 *)p; free(p); return 0; }
static int h_query(void *st) { (void)st; query_calls++; if (query_busy > 0) { query_busy--; return 600; } return sync_ok ? 0 : 600; }
static int h_getdev(int *d) { *d = 0; return 0; }
static const char *pci_busid = "0000:03:00.0";
static int h_pci(char *buf, int len, int dev) { assert(dev == 0); snprintf(buf, len, "%s", pci_busid); return 0; }
static void *fg_import_desc; static void *fg_destroyed; static const void *fg_wait_sems, *fg_wait_params; static unsigned fg_wait_n;
static const void *fg_sig_sems, *fg_sig_params; static unsigned fg_sig_n;
static int fg_import(void **out, const void *d) { fg_import_desc = (void *)d; *out = (void *)0xabc; return 55; }
static int fg_destroy(void *s) { fg_destroyed = s; return 56; }
static int fg_wait(const void *s, const void *p, unsigned n, void *st) { fg_wait_sems = s; fg_wait_params = p; fg_wait_n = n; (void)st; return 57; }
static int fg_signal(const void *s, const void *p, unsigned n, void *st) { fg_sig_sems = s; fg_sig_params = p; fg_sig_n = n; (void)st; return 58; }

static void install(void) {
    g_hip_ok = 1;
    fence_ops.nt_to_fd = d_nt; fence_ops.open_node = d_open; fence_ops.timeline_cap = d_cap;
    fence_ops.fd_to_handle = d_fd2h; fence_ops.destroy = d_destroy; fence_ops.wait = d_wait;
    fence_ops.signal = d_signal; fence_ops.query = d_query;
    p_hipStreamWaitValue64 = h_wait64; p_hipLaunchHostFunc = h_hostfunc; p_hipDeviceGetPCIBusId = h_pci;
    real.p_hipHostMalloc = h_hostmalloc; real.p_hipHostGetDevicePointer = h_devptr; real.p_hipHostFree = h_hostfree;
    real.p_hipStreamQuery = h_query; real.p_hipGetDevice = h_getdev;
    real.p_hipImportExternalSemaphore = fg_import; real.p_hipDestroyExternalSemaphore = fg_destroy;
    real.p_hipWaitExternalSemaphoresAsync = fg_wait; real.p_hipSignalExternalSemaphoresAsync = fg_signal;
}
static void run_cb(void) { void (*fn)(void *) = cb_fn; void *d = cb_data; cb_fn = 0; in_cb = 1; fn(d); in_cb = 0; }
static u64 flag_now(void) { return __atomic_load_n((u64 *)wait_ptr, __ATOMIC_ACQUIRE); }
static int flag_reaches(u64 v, int ms) {
    for (long long end = mono_ns() + ms * 1000000LL; mono_ns() < end; usleep(1000))
        if (flag_now() >= v) return 1;
    return 0;
}
struct sem_desc { int type; int pad; void *handle; const void *name; unsigned flags; unsigned reserved[16]; };
struct wparams { u64 value; unsigned char rest[136]; };
'''


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class NativeEmulation(unittest.TestCase):
    def run_c(self, body, extra=('-ldl',), env=None):
        def run(d, exe):
            subprocess.run([exe, str(d)], check=True, capture_output=True, timeout=60,
                           env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0', DLSSNR_HIP_LOG=str(d / 'hip.log'),
                                    **(env or {})))
        compile_and_run(FAKES + body, ROOT, extra, run=run)

    def test_import_other_types_forward_and_bad_conversions_fail_loudly(self):
        self.run_c(r'''
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 1, .handle = (void *)0x1234 };
    assert(g_bridge.magic == 0x374849504E524C44ULL);
    /* anything but a D3D12 fence goes to HIP untouched */
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 55 && sem == (void *)0xabc && fg_import_desc == &d);
    assert(!node_opens);
    d.type = 4; sem = 0; fg_import_desc = 0;
    nt_fd_result = -ENOSYS;   /* win32u has no d3dkmt sync helpers */
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 801 && !sem && !fg_import_desc && !node_opens);
    nt_fd_result = -EBADF;    /* helpers present, handle is not a sync object */
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 400 && !sem && !node_opens);
    nt_fd_result = 0; d.handle = 0;
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 1 && !sem && !node_opens);
    assert(g_bridge.p_hipImportExternalSemaphore(0, &d) == 1 && g_bridge.p_hipImportExternalSemaphore(&sem, 0) == 1);
    return 0;
}
''')

    def test_render_node_comes_from_the_override_or_the_hip_devices_pci_address(self):
        self.run_c(r'''
int main(int argc, char **argv) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    char tree[512], p[600], dev[600];
    snprintf(tree, sizeof tree, "%s/sys", argv[1]);
    snprintf(p, sizeof p, "mkdir -p %s/pci/0000:03:00.0 %s/pci/0000:0e:00.0 %s/renderD128 %s/renderD129 %s/card0 && "
        "ln -s ../../pci/0000:0e:00.0 %s/renderD128/device && ln -s ../../pci/0000:03:00.0 %s/renderD129/device && "
        "ln -s ../../pci/0000:03:00.0 %s/card0/device", tree, tree, tree, tree, tree, tree, tree, tree);
    assert(system(p) == 0);
    fence_sysfs_root = tree; fence_dev_root = "/dev/fake";
    char out[128];
    assert(fence_render_node(out, sizeof out) == 0 && !strcmp(out, "/dev/fake/renderD129"));
    pci_busid = "0000:0E:00.0";   /* case differs between HIP and sysfs */
    assert(fence_render_node(out, sizeof out) == 0 && !strcmp(out, "/dev/fake/renderD128"));
    pci_busid = "0000:77:00.0";
    assert(fence_render_node(out, sizeof out) != 0);
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 801 && !sem && !node_opens);
    pci_busid = "0000:03:00.0";
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0 && sem && node_opens == 1);
    assert(!strcmp(opened_path, "/dev/fake/renderD129") && nt_seen == (void *)0x1234);
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0);
    setenv("DLSSNR_DRM_RENDER_NODE", "/dev/dri/renderD999", 1);
    sem = 0;
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0 && node_opens == 2 && !strcmp(opened_path, "/dev/dri/renderD999"));
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_wait_releases_the_stream_only_when_the_timeline_reaches_the_value(self):
        self.run_c(r'''
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    struct wparams p = { .value = 5 }; void *stream = (void *)0x8888;
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0 && sem);
    assert(cap_checks == 1 && destroyed == 0 && imported_handle == 77);
    assert(fcntl(imported_fd, F_GETFD) == -1 && errno == EBADF);   /* the syncobj holds its own reference */
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, stream) == 0);
    assert(wait_calls == 1 && wait_val == 5 && wait_stream == stream && wait_flags == 0 /* >= */ && wait_mask == ~0ULL);
    usleep(250000);
    assert(flag_now() < 5);                 /* nothing signalled yet: the GPU keeps waiting */
    tl_set(4); usleep(250000); assert(flag_now() < 5);
    tl_set(5); assert(flag_reaches(5, 2000));
    /* satisfied or zero waits cost the GPU nothing */
    p.value = 3; assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, stream) == 0 && wait_calls == 1);
    p.value = 0; assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, stream) == 0 && wait_calls == 1);
    /* already signalled on the timeline but not yet seen by the helper */
    tl_set(8); p.value = 8; assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, stream) == 0 && wait_calls == 1);
    assert(flag_now() >= 8);
    /* a failing stream wait is reported, not swallowed */
    p.value = 20; wait_ret = 719; assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, stream) == 719);
    wait_ret = 0;
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0 && destroyed == 1);
    return 0;
}
''')

    def test_waits_are_released_in_value_order_not_behind_a_larger_pending_one(self):
        self.run_c(r'''
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    struct wparams p = { .value = 9 };
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0);
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, 0) == 0);
    p.value = 5; assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, 0) == 0 && wait_calls == 2);
    tl_set(6); assert(flag_reaches(5, 2000));
    usleep(300000); assert(flag_now() == 5);          /* wait(9) must not have been released */
    tl_set(9); assert(flag_reaches(9, 2000));
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_signal_runs_on_the_stream_and_stays_out_of_hip(self):
        self.run_c(r'''
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    struct wparams p = { .value = 6 };
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0);
    assert(g_bridge.p_hipSignalExternalSemaphoresAsync(&sem, &p, 1, (void *)0x7777) == 0);
    assert(cb_fn && cb_stream == (void *)0x7777);
    assert(tl_point == 0);                  /* not before the stream gets there */
    run_cb();
    assert(tl_point == 6);
    p.value = 7; assert(g_bridge.p_hipSignalExternalSemaphoresAsync(&sem, &p, 1, 0) == 0);
    run_cb(); assert(tl_point == 7);
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0 && destroyed == 1);
    return 0;
}
''')

    def test_destroy_releases_pending_waits_and_waits_for_the_stream_before_freeing(self):
        self.run_c(r'''
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    struct wparams p = { .value = 100 };
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0);
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, (void *)0x8888) == 0);
    usleep(150000);
    query_busy = 3; query_calls = 0;
    long long t0 = mono_ns();
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0);
    assert(mono_ns() - t0 < 1500000000LL);                 /* never waits for the missing signal */
    assert(hostfree_n == 1 && flag_at_free == ~0ULL);       /* the GPU wait was released before the free */
    assert(query_calls >= 4);                                /* and the stream drained first */
    assert(destroyed == 1);
    /* a stream that never goes idle: give up and keep the flag memory instead of freeing it */
    fence_idle_timeout_ms = 200; sync_ok = 0; hostfree_n = 0; sem = 0;
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0);
    p.value = 1; assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, (void *)0x8888) == 0);
    t0 = mono_ns(); assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0);
    assert(mono_ns() - t0 < 1500000000LL && hostfree_n == 0 && destroyed == 2);
    return 0;
}
''')

    def test_destroy_with_a_signal_still_queued_keeps_the_syncobj_until_it_runs(self):
        self.run_c(r'''
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    struct wparams p = { .value = 7 };
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0);
    assert(g_bridge.p_hipSignalExternalSemaphoresAsync(&sem, &p, 1, 0) == 0);
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0);
    assert(destroyed == 0 && tl_point == 0);                /* callback still owns the semaphore */
    run_cb();
    assert(tl_point == 7 && destroyed == 1);
    assert(fcntl(node_fd, F_GETFD) == -1);                  /* DRM fd closed with the last reference */
    return 0;
}
''')

    def test_foreign_handles_and_mixed_arrays_go_to_hip_with_their_own_parameters(self):
        self.run_c(r'''
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 0);
    void *sems[3] = { (void *)0xf00, sem, (void *)0xf01 };
    struct wparams p[3] = { { .value = 11 }, { .value = 5 }, { .value = 13 } };
    assert(sizeof p[0] == 144);
    /* the first foreign result is returned; the emulated one still ran */
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync((void *)sems, p, 3, (void *)0x8888) == 57);
    assert(fg_wait_sems == &sems[2] && fg_wait_params == &p[2] && fg_wait_n == 1);
    assert(wait_calls == 1 && wait_val == 5);
    assert(g_bridge.p_hipSignalExternalSemaphoresAsync((void *)sems, p, 3, 0) == 58);
    assert(fg_sig_sems == &sems[2] && fg_sig_params == &p[2] && fg_sig_n == 1);
    run_cb();
    assert(tl_point == 5);
    assert(g_bridge.p_hipDestroyExternalSemaphore((void *)0xf00) == 56 && fg_destroyed == (void *)0xf00 && !destroyed);
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0 && destroyed == 1);
    return 0;
}
''')

    def test_missing_runtime_calls_report_not_ready_or_not_supported(self):
        self.run_c(r'''
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    struct wparams p = { .value = 2 };
    p_hipLaunchHostFunc = 0;
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 801 && !sem && !node_opens);   /* nothing to signal with */
    p_hipLaunchHostFunc = h_hostfunc; p_hipStreamWaitValue64 = 0;
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 801 && !sem && !node_opens);
    p_hipStreamWaitValue64 = h_wait64;
    memset(&real, 0, sizeof real);
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 3);
    assert(g_bridge.p_hipDestroyExternalSemaphore((void *)1) == 3);
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &p, 1, 0) == 3);
    assert(g_bridge.p_hipSignalExternalSemaphoresAsync(&sem, &p, 1, 0) == 3);
    return 0;
}
''')

    def test_nt_handle_is_opened_through_win32us_local_sync_helpers(self):
        fake_win32u = r'''
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
/* Local (non-exported) functions, as in the real win32u.so. */
static int log_fd = -1;
static void note(const char *s) { dprintf(log_fd, "%s\n", s); }
__attribute__((noinline, used)) static unsigned d3dkmt_open_sync(unsigned global, void *shared) {
    char b[64]; snprintf(b, sizeof b, "open %u %p", global, shared); note(b); return shared == (void *)0x4321 ? 9 : 0;
}
__attribute__((noinline, used)) static int d3dkmt_object_get_fd(unsigned local) {
    char b[64]; snprintf(b, sizeof b, "fd %u", local); note(b); return open("/dev/null", O_RDONLY); }
__attribute__((noinline, used)) static long d3dkmt_destroy_sync(unsigned local) {
    char b[64]; snprintf(b, sizeof b, "destroy %u", local); note(b); return 0; }
__attribute__((constructor)) static void start(void) { log_fd = open(getenv("FAKE_WIN32U_LOG"), O_WRONLY | O_APPEND); }
int exported_only(void) { return 1; }
'''
        body = r'''
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    int fd = d3dkmt_sync_handle_to_fd((void *)0x4321);
    assert(fd >= 0 && fcntl(fd, F_GETFD) != -1);
    close(fd);
    assert(d3dkmt_sync_handle_to_fd((void *)0x1) == -EBADF);   /* helpers resolved, open refused */
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / 'win32u.c').write_text(fake_win32u)
            subprocess.run(['gcc', '-O0', '-shared', '-fPIC', '-o', str(t / 'win32u.so'), str(t / 'win32u.c')],
                           check=True, capture_output=True, text=True, timeout=60)
            log = t / 'calls.log'
            log.write_text('')
            self.run_c(body, env={'LD_LIBRARY_PATH': str(t), 'FAKE_WIN32U_LOG': str(log)})
            lines = log.read_text().split('\n')
            self.assertEqual(lines[:3], ['open 0 0x4321', 'fd 9', 'destroy 9'])
            self.assertEqual(lines[3], 'open 0 0x1')

    def test_missing_win32u_helpers_are_reported_as_unavailable(self):
        self.run_c(r'''
int main(void) {
    /* no win32u.so on this machine */
    assert(d3dkmt_sync_handle_to_fd((void *)0x4321) == -ENOSYS);
    return 0;
}
''', env={'LD_LIBRARY_PATH': '/nonexistent'})

    def test_syncobj_ioctl_structs_match_the_kernel_headers(self):
        body = r'''
#include <libdrm/drm.h>
int main(void) {
    assert(sizeof(struct fence_syncobj_handle) == sizeof(struct drm_syncobj_handle));
    assert(sizeof(struct fence_syncobj_timeline_wait) == sizeof(struct drm_syncobj_timeline_wait));
    assert(sizeof(struct fence_syncobj_timeline_array) == sizeof(struct drm_syncobj_timeline_array));
    assert(sizeof(struct fence_syncobj_destroy) == sizeof(struct drm_syncobj_destroy));
    assert(FENCE_IOCTL_GET_CAP == DRM_IOCTL_GET_CAP);
    assert(FENCE_IOCTL_FD_TO_HANDLE == DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE);
    assert(FENCE_IOCTL_DESTROY == DRM_IOCTL_SYNCOBJ_DESTROY);
    assert(FENCE_IOCTL_TIMELINE_WAIT == DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT);
    assert(FENCE_IOCTL_TIMELINE_SIGNAL == DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL);
    assert(FENCE_IOCTL_QUERY == DRM_IOCTL_SYNCOBJ_QUERY);
    assert(FENCE_WAIT_ALL == DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL && FENCE_WAIT_FOR_SUBMIT == DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT);
    assert(FENCE_CAP_SYNCOBJ_TIMELINE == DRM_CAP_SYNCOBJ_TIMELINE);
    return 0;
}
'''
        if not Path('/usr/include/libdrm/drm.h').exists():
            self.skipTest('libdrm headers not installed')
        self.run_c(body, extra=('-ldl', '-I/usr/include/libdrm'))


def hip_skip_reason():
    if not shutil.which('gcc'):
        return 'C compiler required'
    if not ROCM_HIP.is_file():
        return 'ROCm HIP runtime not installed'
    if not RENDER_NODE.exists():
        return 'no DRM render node'
    if not Path('/usr/include/xf86drm.h').exists():
        return 'libdrm headers not installed'
    return None


HARDWARE = r'''
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
#include <errno.h>
#include <xf86drm.h>

/* HIP's own parameter layouts: the emulation indexes arrays with a fixed stride. */
#include <stddef.h>
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
_Static_assert(sizeof(hipExternalSemaphoreWaitParams) == FENCE_PARAMS_SIZE, "wait params size");
_Static_assert(sizeof(hipExternalSemaphoreSignalParams) == FENCE_PARAMS_SIZE, "signal params size");
_Static_assert(offsetof(hipExternalSemaphoreHandleDesc, handle) == 8, "desc handle offset");

static int exported_fd(int dfd, uint32_t *sh) {
    int fd = -1;
    assert(drmSyncobjCreate(dfd, 0, sh) == 0);
    assert(drmSyncobjHandleToFD(dfd, *sh, &fd) == 0);
    return fd;
}
static int dup_fd;
static int nt_hook(void *nt) { (void)nt; return dup(dup_fd); }
static int (*hipStreamQuery_)(void *);
static int (*hipMalloc_)(void **, u64);
static int (*hipMemsetAsync_)(void *, int, u64, void *);
static int (*hipStreamCreate_)(void **);
static int (*hipDeviceSynchronize_)(void);
static long long now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000LL + t.tv_nsec / 1000000; }
static int poll_idle(void *s, int ms) {
    for (long long end = now_ms() + ms; now_ms() < end; usleep(2000)) if (hipStreamQuery_(s) == 0) return 1;
    return hipStreamQuery_(s) == 0;
}
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    alarm(90);
    setenv("DLSSNR_HIP_LIBRARY", getenv("HIP_LIB"), 1);
    const char *node = getenv("DLSSNR_DRM_RENDER_NODE");
    if (!node || !node[0]) setenv("DLSSNR_DRM_RENDER_NODE", "/dev/dri/renderD128", 1);
    node = getenv("DLSSNR_DRM_RENDER_NODE");
    int dfd = open(node, O_RDWR | O_CLOEXEC);
    assert(dfd >= 0);
    uint64_t cap = 0;
    if (drmGetCap(dfd, DRM_CAP_SYNCOBJ_TIMELINE, &cap) || !cap) { puts("SKIP no timeline syncobj"); return 77; }
    if (ensure_hip()) { puts("SKIP HIP did not load"); return 77; }
    if (real.p_hipInit(0) || real.p_hipSetDevice(0)) { puts("SKIP HIP device init failed"); return 77; }
    if (!real.p_hipHostMalloc || !p_hipStreamWaitValue64 || !p_hipLaunchHostFunc) { puts("SKIP runtime lacks calls"); return 77; }
    hipStreamQuery_ = real.p_hipStreamQuery;
    hipMalloc_ = real.p_hipMalloc; hipMemsetAsync_ = real.p_hipMemsetAsync;
    hipStreamCreate_ = real.p_hipStreamCreate; hipDeviceSynchronize_ = real.p_hipDeviceSynchronize;
    fence_ops.nt_to_fd = nt_hook;    /* the Wine side is not available outside a prefix */

    /* without the override, the HIP device's PCI address must lead to a render node */
    char picked[256], saved[256];
    snprintf(saved, sizeof saved, "%s", node);
    unsetenv("DLSSNR_DRM_RENDER_NODE");
    assert(fence_render_node(picked, sizeof picked) == 0);
    printf("PCI match picked %s\n", picked);
    setenv("DLSSNR_DRM_RENDER_NODE", saved, 1);

    uint32_t sh = 0; dup_fd = exported_fd(dfd, &sh);
    void *stream = 0, *buf = 0, *sem = 0;
    assert(hipStreamCreate_(&stream) == 0 && hipMalloc_(&buf, 4096) == 0);
    hipExternalSemaphoreHandleDesc desc; memset(&desc, 0, sizeof desc);
    desc.type = hipExternalSemaphoreHandleTypeD3D12Fence; desc.handle.win32.handle = (void *)0x1234;
    int r = g_bridge.p_hipImportExternalSemaphore(&sem, &desc);
    printf("import -> %d\n", r);
    assert(r == 0 && sem);

    /* wait(7): the stream's memset must not run until the host signals point 7 */
    hipExternalSemaphoreWaitParams w; memset(&w, 0, sizeof w);
    hipExternalSemaphoreSignalParams sg; memset(&sg, 0, sizeof sg);
    w.params.fence.value = 7;
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &w, 1, stream) == 0);
    assert(hipMemsetAsync_(buf, 0xAB, 4096, stream) == 0);
    assert(!poll_idle(stream, 1000));
    uint64_t pt = 6; assert(drmSyncobjTimelineSignal(dfd, &sh, &pt, 1) == 0);
    assert(!poll_idle(stream, 700));                 /* 6 < 7 */
    pt = 7; assert(drmSyncobjTimelineSignal(dfd, &sh, &pt, 1) == 0);
    long long t0 = now_ms();
    assert(poll_idle(stream, 5000));
    printf("wait(7) released %lld ms after signal(7)\n", now_ms() - t0);

    /* signal(V+1) from the stream is visible to the kernel timeline */
    sg.params.fence.value = 8;
    assert(g_bridge.p_hipSignalExternalSemaphoresAsync(&sem, &sg, 1, stream) == 0);
    assert(poll_idle(stream, 5000));
    for (int i = 0; i < 200; i++) { uint64_t q = 0; drmSyncobjQuery(dfd, &sh, &q, 1); if (q == 8) break; usleep(5000); }
    uint64_t q = 0; assert(drmSyncobjQuery(dfd, &sh, &q, 1) == 0 && q == 8);
    puts("signal(8) visible to drmSyncobjQuery");

    /* a frame loop: wait V, work, signal V+1, host answers each */
    for (uint64_t v = 20; v < 220; v += 2) {
        w.params.fence.value = v; sg.params.fence.value = v + 1;
        assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &w, 1, stream) == 0);
        assert(hipMemsetAsync_(buf, (int)v, 4096, stream) == 0);
        assert(g_bridge.p_hipSignalExternalSemaphoresAsync(&sem, &sg, 1, stream) == 0);
        uint64_t p = v; assert(drmSyncobjTimelineSignal(dfd, &sh, &p, 1) == 0);
        for (int i = 0; i < 4000; i++) { uint64_t qq = 0; drmSyncobjQuery(dfd, &sh, &qq, 1); if (qq >= v + 1) break; usleep(500); }
        uint64_t qq = 0; drmSyncobjQuery(dfd, &sh, &qq, 1);
        assert(qq >= v + 1);
    }
    assert(poll_idle(stream, 5000));
    puts("200-step wait/signal loop ok");

    /* destroy with a wait that will never be satisfied must not hang */
    w.params.fence.value = 100000;
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(&sem, &w, 1, stream) == 0);
    assert(hipMemsetAsync_(buf, 0xEE, 4096, stream) == 0);
    t0 = now_ms();
    assert(g_bridge.p_hipDestroyExternalSemaphore(sem) == 0);
    printf("destroy with a pending wait took %lld ms\n", now_ms() - t0);
    assert(now_ms() - t0 < 5000);
    assert(poll_idle(stream, 5000));                  /* the released stream finished */
    assert(hipDeviceSynchronize_() == 0);
    close(dup_fd); drmSyncobjDestroy(dfd, sh);
    puts("OK");
    return 0;
}
'''


@unittest.skipIf(hip_skip_reason(), hip_skip_reason() or '')
class RealHardware(unittest.TestCase):
    def test_wait_and_signal_through_real_hip_and_real_timeline_syncobjs(self):
        hip_include = ROCM_HIP.parents[1] / 'include'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'hw.c').write_text(HARDWARE)
            subprocess.run(['gcc', '-std=gnu11', '-O1', '-g', '-pthread', '-no-pie', '-fno-pie',
                            '-I', str(ROOT), '-I', '/usr/include/libdrm', '-I', str(hip_include),
                            '-o', str(d / 'hw'), str(d / 'hw.c'), '-ldl', '-ldrm'],
                           check=True, capture_output=True, text=True, timeout=120)
            env = dict(os.environ, HIP_LIB=str(ROCM_HIP), DLSSNR_HIP_LOG=str(d / 'hip.log'),
                       DLSSNR_DRM_RENDER_NODE=str(RENDER_NODE))
            try:
                run = subprocess.run([str(d / 'hw')], capture_output=True, text=True, timeout=120, env=env)
            except subprocess.TimeoutExpired as e:
                self.fail('hardware run hung: %s' % (e.stdout or b'')[-2000:])
            if run.returncode == 77:
                self.skipTest(run.stdout.strip())
            log = (d / 'hip.log').read_text() if (d / 'hip.log').exists() else ''
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr + log[-3000:])
            self.assertIn('OK', run.stdout)


if __name__ == '__main__':
    unittest.main()
