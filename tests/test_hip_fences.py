"""D3D12 fence emulation, bridge half: the host-mapped counter, the signal callback and the semaphore forwards."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bridge_060 import compile_and_run
from test_hip_modules import pe_source
from test_pe_fence import FAKES as PE_FAKES
from dlssnr import upstream

ROOT = Path(__file__).resolve().parents[1]
FENCE_CALLS = ('hipImportExternalSemaphore', 'hipDestroyExternalSemaphore',
               'hipWaitExternalSemaphoresAsync', 'hipSignalExternalSemaphoresAsync')
def _rocm_hip():
    wheels = sorted((Path.home() / '.local/share/dlssnr-linux/rocm-venv/lib').glob(
        'python*/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7'))
    system = Path('/opt/rocm/lib/libamdhip64.so.7')
    return wheels[-1] if wheels else (system if system.exists() else None)


ROCM_HIP = _rocm_hip()


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
        # The import export reads the descriptor's type before it forwards, so it needs a real one.
        src = pe_source() + PE_FAKES + r'''
static int calls;
static struct { int type; int pad; void *handle; } desc = { 1, 0, (void *)0x1234 };
static int SYSV f_import(void **out, const void *d) {
    calls++; assert(d == (const void *)&desc); *out = (void *)0x5555; return 11;
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
    t.magic = 0x374849504E524C44ULL; /* previous table revision */
    assert(hipImportExternalSemaphore(&sem, &desc) == 3 && !sem && !calls);
    assert(hipDestroyExternalSemaphore((void *)1) == 3);
    assert(hipWaitExternalSemaphoresAsync((void *)1, (void *)1, 1, 0) == 3);
    assert(hipSignalExternalSemaphoresAsync((void *)1, (void *)1, 1, 0) == 3 && !calls);
    t.magic = DLSSNR_HIP_MAGIC;
    assert(hipImportExternalSemaphore(&sem, &desc) == 11 && sem == (void *)0x5555);
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


# Fakes for the HIP entry points the counter half calls. HIP calls assert they are
# not running inside a host callback, which is where HIP may not be re-entered.
FAKES = r"""
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
#include <errno.h>

static __thread int in_cb;
static int wait_calls, hostfree_n, query_busy, query_calls, sync_ok = 1, hostfunc_ret, hostfunc_calls, sync_calls;
static void *wait_stream, *wait_ptr; static u64 wait_val, wait_mask; static unsigned wait_flags; static int wait_ret;
static void (*cb_fn)(void *); static void *cb_data, *cb_stream;
static u64 counter_at_free; static unsigned hostmalloc_flags; static int hostmalloc_ret, devptr_calls;
static int h_wait64(void *st, void *ptr, u64 v, unsigned fl, u64 mask) {
    assert(!in_cb); wait_calls++; wait_stream = st; wait_ptr = ptr; wait_val = v; wait_flags = fl; wait_mask = mask; return wait_ret;
}
static int h_hostfunc(void *st, void (*fn)(void *), void *d) {
    assert(!in_cb); hostfunc_calls++;
    if (hostfunc_ret) return hostfunc_ret;
    cb_fn = fn; cb_data = d; cb_stream = st; return 0;
}
static int h_hostmalloc(void **p, u64 sz, unsigned fl) {
    assert(!in_cb && sz >= 8); hostmalloc_flags = fl;
    if (hostmalloc_ret) return hostmalloc_ret;
    *p = calloc(1, sz); return 0;
}
static int h_devptr(void **d, void *h, unsigned fl) { assert(!in_cb && !fl); devptr_calls++; *d = h; return 0; }
static int h_hostfree(void *p) { assert(!in_cb); hostfree_n++; counter_at_free = *(volatile u64 *)p; free(p); return 0; }
static int h_query(void *st) { (void)st; query_calls++; if (query_busy > 0) { query_busy--; return 600; } return sync_ok ? 0 : 600; }
static int h_devsync(void) { sync_calls++; return 0; }
static long long mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }

static void *fg_import_desc; static void *fg_destroyed; static const void *fg_wait_sems, *fg_wait_params; static unsigned fg_wait_n;
static const void *fg_sig_sems, *fg_sig_params; static unsigned fg_sig_n; static void *fg_wait_stream, *fg_sig_stream;
static int fg_import(void **out, const void *d) { fg_import_desc = (void *)d; *out = (void *)0xabc; return 55; }
static int fg_destroy(void *s) { fg_destroyed = s; return 56; }
static int fg_wait(const void *s, const void *p, unsigned n, void *st) { fg_wait_sems = s; fg_wait_params = p; fg_wait_n = n; fg_wait_stream = st; return 57; }
static int fg_signal(const void *s, const void *p, unsigned n, void *st) { fg_sig_sems = s; fg_sig_params = p; fg_sig_n = n; fg_sig_stream = st; return 58; }

static void install(void) {
    g_hip_ok = 1;
    p_hipStreamWaitValue64 = h_wait64; p_hipLaunchHostFunc = h_hostfunc;
    real.p_hipHostMalloc = h_hostmalloc; real.p_hipHostGetDevicePointer = h_devptr; real.p_hipHostFree = h_hostfree;
    real.p_hipStreamQuery = h_query; real.p_hipDeviceSynchronize = h_devsync;
    real.p_hipImportExternalSemaphore = fg_import; real.p_hipDestroyExternalSemaphore = fg_destroy;
    real.p_hipWaitExternalSemaphoresAsync = fg_wait; real.p_hipSignalExternalSemaphoresAsync = fg_signal;
}
static void run_cb(void) { void (*fn)(void *) = cb_fn; void *d = cb_data; cb_fn = 0; in_cb = 1; fn(d); in_cb = 0; }
static u64 ready_of(DlssnrFenceShared *f) { return __atomic_load_n(&f->signal_ready, __ATOMIC_ACQUIRE); }
static u64 counter_of(DlssnrFenceShared *f) { return __atomic_load_n((u64 *)f->counter, __ATOMIC_ACQUIRE); }
struct wparams { u64 value; unsigned char rest[136]; };
"""


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class NativeCounter(unittest.TestCase):
    def run_c(self, body, extra=('-ldl',), env=None):
        def run(d, exe):
            subprocess.run([exe, str(d)], check=True, capture_output=True, timeout=60,
                           env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0', DLSSNR_HIP_LOG=str(d / 'hip.log'),
                                    **(env or {})))
        compile_and_run(FAKES + body, ROOT, extra, run=run)

    def test_the_table_carries_the_counter_slots_and_the_pe_log(self):
        self.run_c(r"""
int main(void) {
    assert(g_bridge.p_dlssnr_fence_create && g_bridge.p_dlssnr_fence_wait && g_bridge.p_dlssnr_fence_signal);
    assert(g_bridge.p_dlssnr_fence_drain && g_bridge.p_dlssnr_fence_release && g_bridge.p_dlssnr_log);
    assert(g_bridge.p_dlssnr_fence_await_ready);
    return 0;
}
""")

    def test_create_allocates_a_zeroed_host_mapped_counter_and_needs_the_wait_and_callback_calls(self):
        self.run_c(r"""
int main(void) {
    install(); DlssnrFenceShared *f = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0 && f);
    assert(f->magic == DLSSNR_FENCE_SHARED_MAGIC && f->refs == 1);
    assert(hostmalloc_flags == (0x2u | 0x40000000u));        /* mapped + coherent */
    assert(*f->counter == 0 && f->counter_dev == (void *)f->counter && f->signal_ready == 0);
    assert(g_bridge.p_dlssnr_fence_release(f, 1) == 0 && hostfree_n == 1);

    f = 0; p_hipStreamWaitValue64 = 0;     /* the runtime cannot hold a stream on a value */
    assert(g_bridge.p_dlssnr_fence_create(&f) == 801 && !f);
    p_hipStreamWaitValue64 = h_wait64; p_hipLaunchHostFunc = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 801 && !f);
    p_hipLaunchHostFunc = h_hostfunc;
    hostmalloc_ret = 2;                    /* pinned allocation refused */
    assert(g_bridge.p_dlssnr_fence_create(&f) != 0 && !f);
    hostmalloc_ret = 0; real.p_hipHostMalloc = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 3 && !f);
    assert(g_bridge.p_dlssnr_fence_create(0) != 0);
    memset(&real, 0, sizeof real);
    assert(g_bridge.p_dlssnr_fence_create(&f) == 3);
    return 0;
}
""")

    def test_wait_enqueues_a_gte_stream_wait_on_the_counter_and_notes_the_stream(self):
        self.run_c(r"""
int main(void) {
    install(); DlssnrFenceShared *f = 0; void *stream = (void *)0x8888;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0);
    assert(g_bridge.p_dlssnr_fence_wait(f, stream, 5) == 0);
    assert(wait_calls == 1 && wait_val == 5 && wait_stream == stream && wait_ptr == f->counter_dev);
    assert(wait_flags == 0 /* >= */ && wait_mask == ~0ULL);
    wait_ret = 719;                         /* a failing stream wait is reported, not swallowed */
    assert(g_bridge.p_dlssnr_fence_wait(f, stream, 6) == 719);
    wait_ret = 0;
    query_calls = 0;
    assert(g_bridge.p_dlssnr_fence_drain(f) == 0 && query_calls == 1);   /* one stream noted, not two */
    assert(g_bridge.p_dlssnr_fence_release(f, 1) == 0);
    return 0;
}
""")

    def test_signal_runs_a_host_callback_that_raises_signal_ready_and_stays_out_of_hip(self):
        self.run_c(r"""
int main(void) {
    install(); DlssnrFenceShared *f = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0);
    assert(g_bridge.p_dlssnr_fence_signal(f, (void *)0x7777, 6) == 0);
    assert(cb_fn && cb_stream == (void *)0x7777 && f->refs == 2);
    assert(ready_of(f) == 0);               /* not before the stream gets there */
    run_cb();
    assert(ready_of(f) == 6 && f->refs == 1);
    assert(g_bridge.p_dlssnr_fence_signal(f, 0, 4) == 0);
    run_cb(); assert(ready_of(f) == 6);     /* a lower value never moves it back */
    assert(g_bridge.p_dlssnr_fence_signal(f, 0, 9) == 0);
    run_cb(); assert(ready_of(f) == 9);
    /* hipLaunchHostFunc refusing the callback drops the reference it took */
    hostfunc_ret = 719;
    assert(g_bridge.p_dlssnr_fence_signal(f, 0, 12) == 719 && f->refs == 1 && ready_of(f) == 9);
    assert(g_bridge.p_dlssnr_fence_release(f, 1) == 0);
    return 0;
}
""")

    def test_await_ready_sleeps_until_the_callback_raises_signal_ready_or_the_timeout_passes(self):
        self.run_c(r"""
static void *run_cb_later(void *unused) { (void)unused; usleep(150000); run_cb(); return 0; }
int main(void) {
    install(); DlssnrFenceShared *f = 0; pthread_t t;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0);
    assert(g_bridge.p_dlssnr_fence_await_ready(0, 0, 10) != 0);
    /* nothing ready: the full timeout passes, and the caller is told so */
    long long t0 = mono_ns();
    assert(g_bridge.p_dlssnr_fence_await_ready(f, 0, 60) == 600);
    assert(mono_ns() - t0 >= 50000000LL && mono_ns() - t0 < 1000000000LL);
    /* the callback runs on another thread: the sleeper wakes on it, well before its timeout */
    assert(g_bridge.p_dlssnr_fence_signal(f, 0, 5) == 0);
    assert(pthread_create(&t, 0, run_cb_later, 0) == 0);
    t0 = mono_ns();
    assert(g_bridge.p_dlssnr_fence_await_ready(f, 0, 5000) == 0);
    assert(mono_ns() - t0 >= 100000000LL && mono_ns() - t0 < 2000000000LL && ready_of(f) == 5);
    pthread_join(t, 0);
    /* already past `after`: returns at once; not past it: waits */
    t0 = mono_ns();
    assert(g_bridge.p_dlssnr_fence_await_ready(f, 4, 5000) == 0 && mono_ns() - t0 < 100000000LL);
    assert(g_bridge.p_dlssnr_fence_await_ready(f, 5, 30) == 600);
    assert(g_bridge.p_dlssnr_fence_release(f, 1) == 0);
    return 0;
}
""")

    def test_drain_raises_the_counter_and_waits_for_the_streams_then_release_frees_or_leaks(self):
        self.run_c(r"""
int main(void) {
    install(); DlssnrFenceShared *f = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0);
    assert(g_bridge.p_dlssnr_fence_wait(f, (void *)0x8888, 100) == 0);
    query_busy = 3; query_calls = 0;
    long long t0 = mono_ns();
    assert(g_bridge.p_dlssnr_fence_drain(f) == 0);
    assert(mono_ns() - t0 < 1500000000LL);               /* never waits for the missing signal */
    assert(counter_of(f) == ~0ULL && query_calls >= 4);   /* the GPU wait was released, then the stream drained */
    assert(g_bridge.p_dlssnr_fence_release(f, 1) == 0);
    assert(hostfree_n == 1 && counter_at_free == ~0ULL);

    /* a stream that never goes idle: report it, and keep the counter page instead of freeing it */
    fence_idle_timeout_ms = 200; sync_ok = 0; hostfree_n = 0; f = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0);
    assert(g_bridge.p_dlssnr_fence_wait(f, (void *)0x8888, 1) == 0);
    t0 = mono_ns();
    assert(g_bridge.p_dlssnr_fence_drain(f) == 600);
    assert(mono_ns() - t0 < 1500000000LL);
    assert(g_bridge.p_dlssnr_fence_release(f, 0) == 0 && hostfree_n == 0);

    /* more distinct streams than the list holds: fall back to a device-wide sync */
    sync_ok = 1; sync_calls = 0; f = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0);
    for (int i = 0; i < 17; i++) assert(g_bridge.p_dlssnr_fence_wait(f, (void *)(0x9000 + i), 1) == 0);
    assert(g_bridge.p_dlssnr_fence_drain(f) == 0 && sync_calls == 1);
    assert(g_bridge.p_dlssnr_fence_release(f, 1) == 0);
    return 0;
}
""")

    def test_drain_also_waits_for_a_stream_that_only_signals(self):
        self.run_c(r"""
int main(void) {
    install(); DlssnrFenceShared *f = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0);
    assert(g_bridge.p_dlssnr_fence_wait(f, (void *)0x8888, 1) == 0);
    assert(g_bridge.p_dlssnr_fence_signal(f, (void *)0x7777, 2) == 0);   /* another stream, signal only */
    query_calls = 0; query_busy = 0;
    assert(g_bridge.p_dlssnr_fence_drain(f) == 0 && query_calls == 2);
    run_cb();
    assert(g_bridge.p_dlssnr_fence_release(f, 1) == 0);
    return 0;
}
""")

    def test_release_with_a_callback_still_queued_keeps_the_struct_until_it_runs(self):
        self.run_c(r"""
int main(void) {
    install(); DlssnrFenceShared *f = 0;
    assert(g_bridge.p_dlssnr_fence_create(&f) == 0);
    assert(g_bridge.p_dlssnr_fence_signal(f, 0, 7) == 0);
    assert(g_bridge.p_dlssnr_fence_release(f, 1) == 0);
    assert(f->magic == DLSSNR_FENCE_SHARED_MAGIC && f->refs == 1);   /* the callback still owns it */
    run_cb();                                                        /* writes into live memory, then frees */
    return 0;
}
""")

    def test_semaphore_slots_forward_every_call_to_hip_unchanged(self):
        self.run_c(r"""
struct sem_desc { int type; int pad; void *handle; const void *name; unsigned flags; unsigned reserved[16]; };
int main(void) {
    install(); void *sem = 0; struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    struct wparams p[3] = { { .value = 11 }, { .value = 5 }, { .value = 13 } };
    void *sems[3] = { (void *)0xf00, (void *)0xf01, (void *)0xf02 };
    assert(sizeof p[0] == 144);
    /* a type 4 descriptor reaching the bridge is HIP's business now: no emulation here */
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 55 && sem == (void *)0xabc && fg_import_desc == &d);
    assert(g_bridge.p_hipDestroyExternalSemaphore((void *)0xf00) == 56 && fg_destroyed == (void *)0xf00);
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(sems, p, 3, (void *)0x8888) == 57);
    assert(fg_wait_sems == sems && fg_wait_params == p && fg_wait_n == 3 && fg_wait_stream == (void *)0x8888);
    assert(g_bridge.p_hipSignalExternalSemaphoresAsync(sems, p, 3, (void *)0x9999) == 58);
    assert(fg_sig_sems == sems && fg_sig_params == p && fg_sig_n == 3 && fg_sig_stream == (void *)0x9999);
    assert(!wait_calls && !hostfunc_calls);
    /* null arguments are rejected before HIP sees them */
    assert(g_bridge.p_hipImportExternalSemaphore(0, &d) == 1 && g_bridge.p_hipImportExternalSemaphore(&sem, 0) == 1);
    assert(g_bridge.p_hipDestroyExternalSemaphore(0) == 1);
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(0, p, 1, 0) == 1 && g_bridge.p_hipSignalExternalSemaphoresAsync(sems, 0, 1, 0) == 1);
    real.p_hipImportExternalSemaphore = 0; real.p_hipDestroyExternalSemaphore = 0;
    real.p_hipWaitExternalSemaphoresAsync = 0; real.p_hipSignalExternalSemaphoresAsync = 0;
    assert(g_bridge.p_hipImportExternalSemaphore(&sem, &d) == 3 && g_bridge.p_hipDestroyExternalSemaphore(sems[0]) == 3);
    assert(g_bridge.p_hipWaitExternalSemaphoresAsync(sems, p, 1, 0) == 3 && g_bridge.p_hipSignalExternalSemaphoresAsync(sems, p, 1, 0) == 3);
    return 0;
}
""")

    def test_pe_log_lines_are_prefixed_and_formatted_in_the_bridge(self):
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / 'hip.log'
            compile_and_run(FAKES + r"""
int main(void) {
    g_bridge.p_dlssnr_log("fence=%llx waits=%llu done=%llu", 0xabcULL, 17ULL, 3ULL);
    return 0;
}
""", ROOT, ('-ldl',), run=lambda d, exe: subprocess.run(
                [exe], check=True, capture_output=True, timeout=30,
                env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0', DLSSNR_HIP_LOG=str(log))))
            self.assertRegex(log.read_text(), r'\[pid \d+\] pe: fence=abc waits=17 done=3\n')


# The counter path against a real ROCm runtime, with the host standing in for the
# PE worker: it raises the counter the way pe_fence.c's worker does. The fakes above
# accept any pointer and never run a GPU wait, so this is the only place that
# checks the host-mapped allocation flags, hipStreamWaitValue64 and the callback
# thread against the real thing. No D3D12 or DRM involved.
HARDWARE = r"""
#define _GNU_SOURCE
#include "native/hip_bridge.c"
#include <assert.h>
#include <unistd.h>

static long long now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000LL + t.tv_nsec / 1000000; }
static int poll_idle(void *s, int ms) {
    for (long long end = now_ms() + ms; now_ms() < end; usleep(2000)) if (real.p_hipStreamQuery(s) == 0) return 1;
    return real.p_hipStreamQuery(s) == 0;
}
static void raise_counter(DlssnrFenceShared *f, u64 v) { fence_max_raise(f->counter, v); }

int main(void) {
    alarm(90);
    if (ensure_hip()) { puts("SKIP HIP did not load"); return 77; }
    if (real.p_hipInit(0) || real.p_hipSetDevice(0)) { puts("SKIP HIP device init failed"); return 77; }
    if (!real.p_hipHostMalloc || !p_hipStreamWaitValue64 || !p_hipLaunchHostFunc) { puts("SKIP runtime lacks calls"); return 77; }
    void *stream = 0, *buf = 0;
    DlssnrFenceShared *f = 0;
    assert(real.p_hipStreamCreate(&stream) == 0 && real.p_hipMalloc(&buf, 4096) == 0);
    assert(fence_create(&f) == 0 && f);

    /* wait(7): the stream's memset must not run until the counter reaches 7 */
    assert(fence_wait(f, stream, 7) == 0);
    assert(real.p_hipMemsetAsync(buf, 0xAB, 4096, stream) == 0);
    assert(!poll_idle(stream, 800));
    raise_counter(f, 6);
    assert(!poll_idle(stream, 500));                 /* 6 < 7 */
    raise_counter(f, 7);
    assert(poll_idle(stream, 5000));
    puts("wait(7) released by the counter");

    /* signal(9): the callback raises signal_ready on a ROCm thread and wakes the sleeper */
    assert(fence_signal(f, stream, 9) == 0);
    assert(fence_await_ready(f, 0, 5000) == 0 && f->signal_ready == 9);
    assert(poll_idle(stream, 5000));
    assert(fence_await_ready(f, 9, 50) == 600);
    puts("signal(9) reached signal_ready");

    /* a frame loop: wait V, work, signal V+1, the host answers each */
    for (u64 v = 20; v < 120; v += 2) {
        assert(fence_wait(f, stream, v) == 0);
        assert(real.p_hipMemsetAsync(buf, (int)v, 4096, stream) == 0);
        assert(fence_signal(f, stream, v + 1) == 0);
        raise_counter(f, v);
        assert(fence_await_ready(f, v - 1 > 9 ? v - 1 : 9, 5000) == 0 && f->signal_ready >= v + 1);
    }
    assert(poll_idle(stream, 5000));
    puts("50-step wait/signal loop ok");

    /* destroy with a wait nobody will satisfy must not hang */
    assert(fence_wait(f, stream, 100000) == 0);
    assert(real.p_hipMemsetAsync(buf, 0xEE, 4096, stream) == 0);
    long long t0 = now_ms();
    assert(fence_drain(f) == 0 && now_ms() - t0 < 5000);
    assert(fence_release(f, 1) == 0);
    assert(real.p_hipDeviceSynchronize() == 0);
    puts("OK");
    return 0;
}
"""


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
@unittest.skipUnless(ROCM_HIP and Path('/dev/kfd').exists(), 'ROCm runtime or /dev/kfd absent')
class RealHardware(unittest.TestCase):
    def test_counter_wait_and_signal_callback_against_real_rocm(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'hw.c').write_text(HARDWARE)
            subprocess.run(['gcc', '-std=gnu11', '-O1', '-g', '-pthread', '-no-pie', '-fno-pie',
                            '-I', str(ROOT), '-o', str(d / 'hw'), str(d / 'hw.c'), '-ldl'],
                           check=True, capture_output=True, text=True, timeout=120)
            env = dict(os.environ, DLSSNR_HIP_LIBRARY=str(ROCM_HIP), DLSSNR_HIP_LOG=str(d / 'hip.log'))
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
