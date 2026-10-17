"""PE half of the D3D12 fence emulation: the worker, the exports and the COM calls, on CPU with fakes."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bridge_060 import compile_and_run
from test_hip_modules import pe_source

ROOT = Path(__file__).resolve().parents[1]
PE_FENCE = ROOT / 'sources/trampoline/pe_fence.c'
IDL = ROOT / 'build/vkd3d-source/include/vkd3d_d3d12.idl'

# Win32 and COM fakes. Everything is __stdcall (ms_abi in the test build), as the
# production code calls it. All sync objects share one mutex and condition variable,
# which keeps WaitForMultipleObjects trivial and still exercises real threads.
FAKES = r'''
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define WAIT_TIMEOUT_ 0x102u
#define E_FAIL_ 0x80004005u

static pthread_mutex_t gmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gcv = PTHREAD_COND_INITIALIZER;

struct fobj { int refs; int manual; int set; int kind; int tag; };   /* kind 1 event, 2 thread */
static int events_made, thread_never_signalled, closed_done;
static char order[512];
static void note(const char *tok) {
    pthread_mutex_lock(&gmu);
    if (strlen(order) + strlen(tok) + 2 < sizeof order) { strcat(order, tok); strcat(order, " "); }
    pthread_mutex_unlock(&gmu);
}

/* captured PE log lines */
static char logbuf[16384];
static void SYSV fake_log(const char *fmt, u64 a, u64 b, u64 c) {
    char line[256];
    snprintf(line, sizeof line, fmt, a, b, c);
    pthread_mutex_lock(&gmu);
    strncat(logbuf, line, sizeof logbuf - strlen(logbuf) - 2); strcat(logbuf, "\n");
    pthread_mutex_unlock(&gmu);
}
static int log_has(const char *needle) { pthread_mutex_lock(&gmu); int r = strstr(logbuf, needle) != 0; pthread_mutex_unlock(&gmu); return r; }

static long long mono_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000LL + t.tv_nsec / 1000000; }
#define WAIT_UNTIL(cond, ms) ({ int ok_ = 0; for (long long e_ = mono_ms() + (ms); mono_ms() < e_; usleep(1000)) if (cond) { ok_ = 1; break; } ok_ || (cond); })
static void settle(int ms) { usleep(ms * 1000); }

HANDLE __stdcall CreateEventA(void *sa, BOOL manual, BOOL initial, const char *name) {
    struct fobj *o = calloc(1, sizeof *o);
    (void)sa; (void)name;
    o->refs = 1; o->manual = manual; o->set = initial; o->kind = 1;
    pthread_mutex_lock(&gmu); o->tag = events_made++ % 4;   /* wake, done, exited, relay_exited */ pthread_mutex_unlock(&gmu);
    return o;
}
BOOL __stdcall SetEvent(HANDLE h) {
    struct fobj *o = h;
    /* The waiter may close the handle the moment it sees the event, so nothing
     * of `o` is read after it is set. */
    if (o->kind == 1 && o->manual) note("exited");
    pthread_mutex_lock(&gmu);
    o->set = 1;
    pthread_cond_broadcast(&gcv);
    pthread_mutex_unlock(&gmu);
    return 1;
}
static int wait_multi_calls;
void __stdcall Sleep(DWORD ms) { usleep(ms * 1000); }
DWORD __stdcall WaitForMultipleObjects(DWORD n, const HANDLE *h, BOOL all, DWORD ms) {
    struct timespec ts;
    DWORD r = WAIT_TIMEOUT_;
    (void)all;
    __atomic_add_fetch(&wait_multi_calls, 1, __ATOMIC_RELAXED);
    clock_gettime(CLOCK_REALTIME, &ts);
    if (ms != 0xFFFFFFFFu) { ts.tv_sec += ms / 1000; ts.tv_nsec += (long)(ms % 1000) * 1000000L; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; } }
    pthread_mutex_lock(&gmu);
    for (;;) {
        for (DWORD i = 0; i < n; i++) {
            struct fobj *o = h[i];
            if (o->set) { if (!o->manual) o->set = 0; r = i; goto out; }
        }
        if (ms == 0xFFFFFFFFu) pthread_cond_wait(&gcv, &gmu);
        else if (pthread_cond_timedwait(&gcv, &gmu, &ts)) { r = WAIT_TIMEOUT_; goto out; }
    }
out:
    pthread_mutex_unlock(&gmu);
    return r;
}
static void obj_unref(struct fobj *o) {
    pthread_mutex_lock(&gmu); int r = --o->refs; pthread_mutex_unlock(&gmu);
    if (!r) free(o);
}

struct tstart { DWORD (__stdcall *fn)(void *); void *arg; struct fobj *thread; };
static void *tmain(void *p) {
    struct tstart *t = p;
    t->fn(t->arg);
    if (!thread_never_signalled) {
        pthread_mutex_lock(&gmu); t->thread->set = 1; pthread_cond_broadcast(&gcv); pthread_mutex_unlock(&gmu);
    }
    obj_unref(t->thread);
    free(t);
    return 0;
}
static int create_thread_fail, create_thread_fail_at, threads_made;   /* fail_at: 1-based call to refuse */
HANDLE __stdcall CreateThread(void *sa, unsigned long long stack, DWORD (__stdcall *fn)(void *), void *arg, DWORD flags, DWORD *tid) {
    struct fobj *o; struct tstart *t; pthread_t th;
    (void)sa; (void)stack; (void)flags; (void)tid;
    if (create_thread_fail || (create_thread_fail_at && ++threads_made == create_thread_fail_at)) return 0;
    o = calloc(1, sizeof *o); o->refs = 2; o->manual = 1; o->kind = 2;
    t = malloc(sizeof *t); t->fn = fn; t->arg = arg; t->thread = o;
    assert(pthread_create(&th, 0, tmain, t) == 0);
    pthread_detach(th);
    return o;
}
BOOL __stdcall CloseHandle(HANDLE h) {
    struct fobj *o = h;
    if (o->kind == 1 && o->tag == 1) note("close_done");
    obj_unref(o);
    return 1;
}
void *__stdcall VirtualAlloc(void *addr, unsigned long long size, DWORD type, DWORD prot) { (void)addr; (void)type; (void)prot; return calloc(1, size); }
BOOL __stdcall VirtualFree(void *p, unsigned long long size, DWORD type) { (void)size; (void)type; free(p); return 1; }

/* ---- D3D12 fakes ---- */
#define NSLOTS_DEV 44
#define NSLOTS_FENCE 12
struct fdev { const void *const *vtbl; int refs; };
struct reg { u64 value; HANDLE ev; int fired; };
struct ffence {
    const void *const *vtbl; int refs; int released; struct fdev *dev;
    u64 value; struct reg regs[256]; int nregs;
    u64 signals[256]; int nsignals; int signal_fails; int seoc_fail;
    int hold_signal, signal_held;
};
static struct fdev dev;
static struct ffence *fences[16]; static int nfences;
static HANDLE opened_handles[16];
static int pin_fail;
static int create_device_hr, open_hr, create_device_calls, open_calls, loadlib_calls, getproc_calls, pin_calls;
static DWORD pin_flags; static const char *pin_addr;

static void bad_slot(void) { fprintf(stderr, "unexpected COM slot called\n"); abort(); }
static DWORD __stdcall dev_release(void *self) { assert(self == &dev); pthread_mutex_lock(&gmu); int r = --dev.refs; pthread_mutex_unlock(&gmu); return r; }
static int __stdcall dev_open_shared(void *self, HANDLE h, const unsigned char *iid, void **out);
static pe_luid *__stdcall dev_luid_fn(void *self, pe_luid *out) {
    assert(self == &dev && out);              /* the result goes through the hidden out pointer */
    out->low = 0x3f5; out->high = 1;
    return out;
}
static const void *dev_vtbl[NSLOTS_DEV];

/* Wine's fence Release joins vkd3d's waiter thread, and that thread cannot finish
 * exiting while the loader lock is held. A test sets loader_lock_owner to the thread
 * standing in for a DllMain detach; a Release from that thread would never return. */
static int loader_lock_held, release_on_loader_lock_thread;
static pthread_t loader_lock_owner;
static DWORD __stdcall fence_release_fn(void *self) {
    struct ffence *f = self;
    note("fence_release");
    pthread_mutex_lock(&gmu);
    if (loader_lock_held && pthread_equal(pthread_self(), loader_lock_owner)) release_on_loader_lock_thread = 1;
    pthread_mutex_unlock(&gmu);
    f->released = 1;
    pthread_mutex_lock(&gmu); f->dev->refs--; pthread_mutex_unlock(&gmu);
    return 0;
}
static u64 __stdcall fence_completed_fn(void *self) { struct ffence *f = self; pthread_mutex_lock(&gmu); u64 v = f->value; pthread_mutex_unlock(&gmu); return v; }
static void fence_fire(struct ffence *f) {   /* gmu held */
    for (int i = 0; i < f->nregs; i++)
        if (!f->regs[i].fired && f->regs[i].value <= f->value) {
            struct fobj *o = f->regs[i].ev;
            f->regs[i].fired = 1; o->set = 1; pthread_cond_broadcast(&gcv);
        }
}
static void fake_fence_complete(struct ffence *f, u64 v) { pthread_mutex_lock(&gmu); f->value = v; fence_fire(f); pthread_mutex_unlock(&gmu); }
static int __stdcall fence_seoc_fn(void *self, u64 v, HANDLE ev) {
    struct ffence *f = self;
    pthread_mutex_lock(&gmu);
    if (f->seoc_fail) { pthread_mutex_unlock(&gmu); return (int)E_FAIL_; }
    assert(f->nregs < 256);
    f->regs[f->nregs].value = v; f->regs[f->nregs].ev = ev; f->regs[f->nregs].fired = 0; f->nregs++;
    fence_fire(f);
    pthread_mutex_unlock(&gmu);
    return 0;
}
static int __stdcall fence_signal_fn(void *self, u64 v) {
    struct ffence *f = self;
    pthread_mutex_lock(&gmu);
    assert(f->nsignals < 256);
    f->signals[f->nsignals++] = v;
    if (f->hold_signal) { f->signal_held = 1; while (f->hold_signal) { pthread_mutex_unlock(&gmu); usleep(1000); pthread_mutex_lock(&gmu); } }
    if (f->signal_fails > 0) { f->signal_fails--; pthread_mutex_unlock(&gmu); return (int)E_FAIL_; }
    f->value = v;                              /* Wine sets the value outright */
    fence_fire(f);
    pthread_mutex_unlock(&gmu);
    return 0;
}
static const void *fence_vtbl[NSLOTS_FENCE];

static int __stdcall dev_open_shared(void *self, HANDLE h, const unsigned char *iid, void **out) {
    struct ffence *f;
    assert(self == &dev && !memcmp(iid, IID_ID3D12Fence, 16));
    open_calls++;
    if (open_hr) return open_hr;
    f = calloc(1, sizeof *f);
    f->vtbl = fence_vtbl; f->refs = 1; f->dev = &dev;
    pthread_mutex_lock(&gmu); dev.refs++; pthread_mutex_unlock(&gmu);
    fences[nfences] = f; opened_handles[nfences] = h; nfences++;
    *out = f;
    return 0;
}
static int __stdcall fake_create_device(void *adapter, int level, const unsigned char *iid, void **out) {
    assert(!adapter && level == 0xb000 && !memcmp(iid, IID_ID3D12Device, 16));
    create_device_calls++;
    if (create_device_hr) return create_device_hr;
    pthread_mutex_lock(&gmu); dev.refs++; pthread_mutex_unlock(&gmu);
    *out = &dev;
    return 0;
}
HANDLE __stdcall LoadLibraryA(const char *name) { assert(!strcmp(name, "d3d12.dll")); loadlib_calls++; return (HANDLE)0xd3d12; }
void *__stdcall GetProcAddress(HANDLE m, const char *name) {
    assert(m == (HANDLE)0xd3d12 && !strcmp(name, "D3D12CreateDevice")); getproc_calls++; return (void *)fake_create_device;
}
BOOL __stdcall GetModuleHandleExA(DWORD flags, const char *addr, HANDLE *out) { if (pin_fail) return 0; pin_calls++; pin_flags = flags; pin_addr = addr; *out = (HANDLE)0x7e57; return 1; }

static void init_com_fakes(void) {
    for (int i = 0; i < NSLOTS_DEV; i++) dev_vtbl[i] = (void *)bad_slot;
    dev_vtbl[DEV_RELEASE] = (void *)dev_release;
    dev_vtbl[DEV_OPEN_SHARED_HANDLE] = (void *)dev_open_shared;
    dev_vtbl[DEV_GET_ADAPTER_LUID] = (void *)dev_luid_fn;
    dev.vtbl = dev_vtbl;
    for (int i = 0; i < NSLOTS_FENCE; i++) fence_vtbl[i] = (void *)bad_slot;
    fence_vtbl[FENCE_RELEASE] = (void *)fence_release_fn;
    fence_vtbl[FENCE_GET_COMPLETED_VALUE] = (void *)fence_completed_fn;
    fence_vtbl[FENCE_SET_EVENT_ON_COMPLETION] = (void *)fence_seoc_fn;
    fence_vtbl[FENCE_SIGNAL] = (void *)fence_signal_fn;
}

/* ---- the bridge table ---- */
struct fake_wait { DlssnrFenceShared *f; void *stream; u64 value; };
static struct fake_wait waits_seen[64]; static int nwaits_seen;
/* how many waits the PE fence had registered when the bridge was asked to enqueue one */
static unsigned pe_wait_registered_at_call;
static struct { DlssnrFenceShared *f; void *stream; u64 value; } sigs_seen[64]; static int nsigs_seen;
static int release_drained = -1, release_calls, drain_calls, drain_ret, create_ret, wait_ret, signal_ret;
static void (*drain_hook)(DlssnrFenceShared *);
static struct pe_fence *current_pe;
static DlssnrFenceShared *last_shared;
static int fwd_import_calls, fwd_destroy_calls, fwd_wait_calls, fwd_signal_calls;
static const void *fwd_desc, *fwd_wait_sems, *fwd_wait_params, *fwd_sig_sems, *fwd_sig_params; static void *fwd_destroyed;
static unsigned fwd_wait_n, fwd_sig_n; static void *fwd_wait_stream;

static int SYSV b_create(DlssnrFenceShared **out) {
    DlssnrFenceShared *s;
    if (create_ret) return create_ret;
    s = calloc(1, sizeof *s);
    s->magic = DLSSNR_FENCE_SHARED_MAGIC; s->refs = 1;
    s->counter = calloc(1, 8); s->counter_dev = (void *)s->counter;
    *out = last_shared = s;
    return 0;
}
static int SYSV b_wait(DlssnrFenceShared *f, void *stream, u64 v) {
    assert(*f->counter < v);                 /* the PE side already answered satisfied waits */
    waits_seen[nwaits_seen].f = f; waits_seen[nwaits_seen].stream = stream; waits_seen[nwaits_seen].value = v;
    pe_wait_registered_at_call = current_pe ? current_pe->nwaits : 999;
    nwaits_seen++;
    return wait_ret;
}
static int SYSV b_signal(DlssnrFenceShared *f, void *stream, u64 v) {
    sigs_seen[nsigs_seen].f = f; sigs_seen[nsigs_seen].stream = stream; sigs_seen[nsigs_seen].value = v; nsigs_seen++;
    return signal_ret;
}
static int await_calls;
static int SYSV b_await(DlssnrFenceShared *f, u64 after, u32 ms) {
    struct timespec ts; int r;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += ms / 1000; ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    pthread_mutex_lock(&gmu);
    await_calls++;
    while (__atomic_load_n(&f->signal_ready, __ATOMIC_ACQUIRE) <= after)
        if (pthread_cond_timedwait(&gcv, &gmu, &ts)) break;
    r = __atomic_load_n(&f->signal_ready, __ATOMIC_ACQUIRE) > after ? 0 : 600;
    pthread_mutex_unlock(&gmu);
    return r;
}
static int SYSV b_drain(DlssnrFenceShared *f) { note("drain"); drain_calls++; if (drain_hook) drain_hook(f); return drain_ret; }
static int SYSV b_release(DlssnrFenceShared *f, int drained) {
    note("bridge_release"); release_calls++; release_drained = drained;
    free((void *)f->counter); free(f);
    return 0;
}
static int SYSV b_fwd_import(void **out, const void *d) { fwd_import_calls++; fwd_desc = d; *out = (void *)0xabc; return 55; }
static int SYSV b_fwd_destroy(void *s) { fwd_destroy_calls++; fwd_destroyed = s; return 56; }
static int SYSV b_fwd_wait(const void *s, const void *p, unsigned n, void *st) { fwd_wait_calls++; fwd_wait_sems = s; fwd_wait_params = p; fwd_wait_n = n; fwd_wait_stream = st; return 57; }
static int SYSV b_fwd_signal(const void *s, const void *p, unsigned n, void *st) { fwd_signal_calls++; fwd_sig_sems = s; fwd_sig_params = p; fwd_sig_n = n; (void)st; return 58; }

static DlssnrHipBridge table;
static void install(void) {
    init_com_fakes();
    memset(&table, 0, sizeof table);
    table.magic = DLSSNR_HIP_MAGIC;
    table.p_hipImportExternalSemaphore = b_fwd_import; table.p_hipDestroyExternalSemaphore = b_fwd_destroy;
    table.p_hipWaitExternalSemaphoresAsync = b_fwd_wait; table.p_hipSignalExternalSemaphoresAsync = b_fwd_signal;
    table.p_dlssnr_fence_create = b_create; table.p_dlssnr_fence_wait = b_wait; table.p_dlssnr_fence_signal = b_signal;
    table.p_dlssnr_fence_drain = b_drain; table.p_dlssnr_fence_release = b_release; table.p_dlssnr_log = fake_log;
    table.p_dlssnr_fence_await_ready = b_await;
    g = &table;
}

struct sem_desc { int type; int pad; void *handle; const void *name; unsigned flags; unsigned reserved[16]; };
struct wparams { u64 value; unsigned char rest[136]; };
static void *import_fence(HANDLE nt) {
    struct sem_desc d = { .type = 4, .handle = nt };
    void *sem = 0;
    assert(hipImportExternalSemaphore(&sem, &d) == 0 && sem);
    current_pe = sem;
    return sem;
}
static struct ffence *last_fence(void) { return fences[nfences - 1]; }
static int do_wait(void *sem, u64 v, void *stream) { struct wparams p = { .value = v }; void *s = sem; return hipWaitExternalSemaphoresAsync(&s, &p, 1, stream); }
static int do_signal(void *sem, u64 v, void *stream) { struct wparams p = { .value = v }; void *s = sem; return hipSignalExternalSemaphoresAsync(&s, &p, 1, stream); }
static u64 counter_now(DlssnrFenceShared *s) { return __atomic_load_n((u64 *)s->counter, __ATOMIC_ACQUIRE); }
static void set_ready(DlssnrFenceShared *s, u64 v) {   /* what the bridge's callback does */
    pthread_mutex_lock(&gmu); __atomic_store_n(&s->signal_ready, v, __ATOMIC_RELEASE); pthread_cond_broadcast(&gcv); pthread_mutex_unlock(&gmu);
}
static int signals_of(struct ffence *f) { pthread_mutex_lock(&gmu); int n = f->nsignals; pthread_mutex_unlock(&gmu); return n; }
static int armed_for(struct ffence *f, u64 v) {
    pthread_mutex_lock(&gmu); int r = 0; for (int i = 0; i < f->nregs; i++) if (f->regs[i].value == v) r = 1; pthread_mutex_unlock(&gmu); return r;
}
'''

def program(body, defines=''):
    return defines + pe_source() + FAKES + body


@unittest.skipUnless(shutil.which('gcc'), 'C compiler required')
class PeFence(unittest.TestCase):
    def run_c(self, body, defines=''):
        def run(d, exe):
            r = subprocess.run([exe], capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0'))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        compile_and_run(program(body, defines), ROOT / 'sources/trampoline', extra=('-lpthread',), run=run)

    def test_import_opens_the_fence_and_drops_its_own_device_reference(self):
        self.run_c(r'''
int main(void) {
    install();
    struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    void *sem = 0, *sems[8];
    sem = import_fence((HANDLE)0x1234);
    assert(create_device_calls == 1 && open_calls == 1 && opened_handles[0] == (HANDLE)0x1234);
    assert(dev.refs == 1);                              /* only the fence's reference is left */
    assert(log_has("pe fence: d3d12 device") && log_has("luid=1:3f5"));
    assert(pin_calls >= 1 && pin_flags == 5 /* PIN | FROM_ADDRESS */ && pin_addr == (const char *)pe_fence_worker);
    assert(log_has("pe fence: imported handle=1234"));
    assert(loadlib_calls == 1 && getproc_calls == 1);

    sems[0] = import_fence((HANDLE)0x2222);             /* another import: device made again, resolver not */
    assert(create_device_calls == 2 && dev.refs == 2 && loadlib_calls == 1 && getproc_calls == 1);
    assert(hipDestroyExternalSemaphore(sems[0]) == 0);
    assert(dev.refs == 1);
    assert(hipDestroyExternalSemaphore(sem) == 0);
    assert(dev.refs == 0 && release_calls == 2 && release_drained == 1);

    /* anything but a D3D12 fence is HIP's */
    d.type = 1; void *out = 0;
    assert(hipImportExternalSemaphore(&out, &d) == 55 && out == (void *)0xabc && fwd_desc == &d);
    assert(create_device_calls == 2);
    d.type = 4; d.handle = 0; out = 0;
    assert(hipImportExternalSemaphore(&out, &d) == 1 && !out);
    assert(hipImportExternalSemaphore(0, &d) == 1 && hipImportExternalSemaphore(&out, 0) == 1);

    /* D3D12CreateDevice fails: no fence, nothing leaked */
    d.handle = (void *)0x3333; create_device_hr = (int)0x887A0004; int before = nfences;
    assert(hipImportExternalSemaphore(&out, &d) == 801 && !out && nfences == before && dev.refs == 0 && log_has("D3D12CreateDevice hr=887a0004"));
    create_device_hr = 0;
    /* OpenSharedHandle fails: the device reference is dropped anyway */
    open_hr = (int)0x80070057;
    assert(hipImportExternalSemaphore(&out, &d) == 400 && !out && dev.refs == 0 && log_has("OpenSharedHandle hr=80070057"));
    open_hr = 0;
    /* the bridge cannot make its half: the fence and device are released again */
    int rel0 = release_calls;
    create_ret = 801;
    assert(hipImportExternalSemaphore(&out, &d) == 801 && !out && dev.refs == 0 && last_fence()->released);
    assert(release_calls == rel0);                      /* the bridge made nothing, so it has nothing to release */
    create_ret = 0;
    /* no thread: the bridge's block is released, drained, exactly once */
    create_thread_fail = 1; release_drained = -1;
    assert(hipImportExternalSemaphore(&out, &d) == 2 && !out && dev.refs == 0 && last_fence()->released);
    assert(release_calls == rel0 + 1 && release_drained == 1);
    create_thread_fail = 0;
    /* the relay thread cannot start: the first worker is stopped again, then everything unwinds */
    create_thread_fail_at = 2; threads_made = 0; release_drained = -1;
    assert(hipImportExternalSemaphore(&out, &d) == 2 && !out && dev.refs == 0 && last_fence()->released);
    assert(release_calls == rel0 + 2 && release_drained == 1);
    create_thread_fail_at = 0;

    /* a ninth import is refused */
    void *live[8];
    for (int i = 0; i < 8; i++) live[i] = import_fence((HANDLE)(0x4000 + i));
    assert(hipImportExternalSemaphore(&out, &d) == 2 && !out && dev.refs == 8);
    for (int i = 0; i < 8; i++) assert(hipDestroyExternalSemaphore(live[i]) == 0);
    assert(dev.refs == 0);
    /* an old table never gets that far */
    table.magic = 0x374849504E524C44ULL;
    assert(hipImportExternalSemaphore(&out, &d) == 3);
    return 0;
}
''')

    def test_a_failed_pin_releases_the_bridge_block_and_the_fence(self):
        self.run_c(r'''
int main(void) {
    install();
    struct sem_desc d = { .type = 4, .handle = (void *)0x1234 };
    void *out = 0;
    pin_fail = 1;
    assert(hipImportExternalSemaphore(&out, &d) == 801 && !out && log_has("cannot pin the trampoline"));
    assert(release_calls == 1 && release_drained == 1 && dev.refs == 0 && last_fence()->released);
    return 0;
}
''')

    def test_bridge_errors_reach_the_caller_and_a_failed_signal_is_never_delivered(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    wait_ret = 719;
    assert(do_wait(sem, 5, 0) == 719);
    wait_ret = 0;
    signal_ret = 700;
    assert(do_signal(sem, 6, 0) == 700 && log_has("bridge signal 6 failed err=700"));
    signal_ret = 0;
    settle(250);
    assert(signals_of(f) == 0);                         /* no callback was queued, so nothing reaches the fence */
    assert(hipDestroyExternalSemaphore(sem) == 0);
    (void)s;
    return 0;
}
''')

    def test_destroy_never_releases_the_fence_on_the_calling_thread(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence();
    pthread_mutex_lock(&gmu); loader_lock_owner = pthread_self(); loader_lock_held = 1; pthread_mutex_unlock(&gmu);
    assert(hipDestroyExternalSemaphore(sem) == 0);
    pthread_mutex_lock(&gmu); loader_lock_held = 0; pthread_mutex_unlock(&gmu);
    assert(f->released && dev.refs == 0 && release_calls == 1);
    assert(!release_on_loader_lock_thread);             /* the join inside Release would never return there */
    return 0;
}
''')

    def test_the_relay_delivers_a_value_raised_while_destroy_was_stopping_it(self):
        self.run_c(r'''
static void raise_during_drain(DlssnrFenceShared *s) { set_ready(s, 2); }
static void *unhold_once_stopping(void *p) {
    struct ffence *f = p;
    while (!current_pe->stop) usleep(500);
    usleep(20000);                                       /* the relay is still inside its first Signal */
    pthread_mutex_lock(&gmu); f->hold_signal = 0; pthread_mutex_unlock(&gmu);
    return 0;
}
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    pthread_t t;
    f->hold_signal = 1;
    assert(do_signal(sem, 1, 0) == 0 && do_signal(sem, 2, 0) == 0);
    set_ready(s, 1);
    assert(WAIT_UNTIL(f->signal_held, 2000));            /* the relay is busy with 1 */
    drain_hook = raise_during_drain;                     /* 2 arrives, then destroy stops the threads */
    assert(pthread_create(&t, 0, unhold_once_stopping, f) == 0);
    assert(hipDestroyExternalSemaphore(sem) == 0);
    pthread_join(t, 0);
    assert(f->nsignals == 2 && f->signals[0] == 1 && f->signals[1] == 2);
    assert(!log_has("never delivered"));
    return 0;
}
''')

    def test_a_signal_the_stream_never_reached_is_reported_when_the_relay_stops(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence();
    assert(do_signal(sem, 7, 0) == 0);                   /* the stream never gets there */
    assert(hipDestroyExternalSemaphore(sem) == 0);
    assert(signals_of(f) == 0 && log_has("signal 7 posted but never delivered (delivered up to 0)"));
    return 0;
}
''')

    def test_wait_releases_the_counter_when_the_fence_reaches_the_value(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    void *stream = (void *)0x8888;
    fake_fence_complete(f, 4);
    assert(do_wait(sem, 5, stream) == 0);
    assert(nwaits_seen == 1 && waits_seen[0].f == s && waits_seen[0].stream == stream && waits_seen[0].value == 5);
    assert(pe_wait_registered_at_call == 1);            /* registered before the stream wait went in */
    assert(WAIT_UNTIL(armed_for(f, 5), 2000));
    settle(250);
    assert(counter_now(s) < 5);
    fake_fence_complete(f, 5);
    assert(WAIT_UNTIL(counter_now(s) >= 5, 2000));
    /* satisfied and zero waits cost the stream nothing */
    assert(do_wait(sem, 3, stream) == 0 && nwaits_seen == 1);
    assert(do_wait(sem, 0, stream) == 0 && nwaits_seen == 1);
    /* the fence is ahead of the counter: answered without the worker or the stream */
    fake_fence_complete(f, 12);
    assert(do_wait(sem, 9, stream) == 0 && nwaits_seen == 1 && counter_now(s) >= 9);
    assert(hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_waits_are_served_smallest_first_and_every_value_below_completed_is_covered(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    assert(do_wait(sem, 9, 0) == 0 && do_wait(sem, 5, 0) == 0);
    assert(WAIT_UNTIL(armed_for(f, 5), 2000));
    settle(100);
    assert(counter_now(s) == 0);
    fake_fence_complete(f, 6);
    assert(WAIT_UNTIL(counter_now(s) == 6, 2000));      /* released to the fence value, not the awaited one */
    assert(WAIT_UNTIL(armed_for(f, 9), 2000));           /* and the next smallest is armed */
    fake_fence_complete(f, 9);
    assert(WAIT_UNTIL(counter_now(s) == 9, 2000));
    assert(hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_a_failed_registration_is_polled_until_the_fence_gets_there(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    f->seoc_fail = 1;
    assert(do_wait(sem, 3, 0) == 0);
    assert(WAIT_UNTIL(log_has("SetEventOnCompletion(3) hr=80004005"), 2000));
    f->seoc_fail = 0;
    fake_fence_complete(f, 3);                          /* no event is armed; only the poll can notice */
    assert(WAIT_UNTIL(counter_now(s) >= 3, 2000));
    assert(hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_signal_reaches_the_fence_once_the_stream_work_has_finished(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    void *stream = (void *)0x7777;
    assert(do_signal(sem, 6, stream) == 0);
    assert(nsigs_seen == 1 && sigs_seen[0].f == s && sigs_seen[0].stream == stream && sigs_seen[0].value == 6);
    settle(250);
    assert(signals_of(f) == 0);                         /* the stream has not got there yet */
    set_ready(s, 6);
    assert(WAIT_UNTIL(signals_of(f) == 1 && f->signals[0] == 6, 2000));
    assert(fence_completed_fn(f) == 6);
    assert(do_signal(sem, 8, stream) == 0);
    set_ready(s, 8);
    assert(WAIT_UNTIL(signals_of(f) == 2 && f->signals[1] == 8, 2000));
    /* the fence only moves forward: a lower value is reported, forwarded to the stream, and never signalled */
    assert(do_signal(sem, 7, stream) == 0 && nsigs_seen == 3);
    assert(log_has("signal 7 not above 8"));
    settle(250);
    assert(signals_of(f) == 2);
    /* the fence is already there (the game queue got to it first): no Signal call */
    fake_fence_complete(f, 10);
    assert(do_signal(sem, 10, stream) == 0);
    set_ready(s, 10);
    settle(250);
    assert(signals_of(f) == 2);
    /* the worker is idle again: a stream that never reaches its signal is the only reason to poll */
    assert(hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_a_signal_waiting_for_its_stream_costs_no_wakeups(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    settle(100);
    assert(do_signal(sem, 6, 0) == 0);
    int w0 = wait_multi_calls, a0 = await_calls;
    settle(400);
    /* the stream has not reached the signal: nothing polls for it. The relay's
       bounded sleep is one call per slice, the main worker is parked. */
    assert(wait_multi_calls - w0 <= 2);
    assert(await_calls - a0 <= 12);
    set_ready(s, 6);
    assert(WAIT_UNTIL(signals_of(f) == 1, 2000));
    assert(hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_a_failed_signal_is_retried_until_it_lands(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    f->signal_fails = 2;
    assert(do_signal(sem, 6, 0) == 0);
    set_ready(s, 6);
    assert(WAIT_UNTIL(signals_of(f) == 3, 2000));
    assert(f->signals[0] == 6 && f->signals[1] == 6 && f->signals[2] == 6 && fence_completed_fn(f) == 6);
    settle(250);
    assert(signals_of(f) == 3);                         /* landed: the worker goes idle */
    assert(log_has("Signal(6) hr=80004005"));
    assert(hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_destroy_drains_with_the_worker_alive_then_stops_it_then_releases(self):
        self.run_c(r'''
static void slow_drain(DlssnrFenceShared *s) {
    usleep(100000);
    set_ready(s, 4);                                    /* the stream reaches the signal while destroy drains */
    usleep(100000);
}
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence();
    assert(do_signal(sem, 4, 0) == 0);
    drain_hook = slow_drain;
    assert(hipDestroyExternalSemaphore(sem) == 0);
    assert(!strcmp(order, "drain exited fence_release exited close_done bridge_release "));
    assert(f->nsignals == 1 && f->signals[0] == 4 && f->released);   /* delivered before the worker stopped */
    assert(release_drained == 1 && dev.refs == 0);
    /* a stream that would not go idle: the counter page is left to the bridge as a leak */
    drain_hook = 0; drain_ret = 600;
    sem = import_fence((HANDLE)0x1235);
    assert(hipDestroyExternalSemaphore(sem) == 0 && release_drained == 0);
    drain_ret = 0;
    /* the handle is gone: it goes to HIP now, and HIP says what it says */
    assert(hipDestroyExternalSemaphore(sem) == 56 && fwd_destroyed == sem);
    assert(do_wait(sem, 3, 0) == 57 && fwd_wait_sems != 0);
    return 0;
}
''')

    def test_destroy_returns_when_the_worker_has_exited_even_if_its_thread_handle_is_not_signalled(self):
        self.run_c(r'''
int main(void) {
    install();
    thread_never_signalled = 1;                         /* a thread parked on the loader lock */
    void *sem = import_fence((HANDLE)0x1234);
    long long t0 = mono_ms();
    assert(hipDestroyExternalSemaphore(sem) == 0);
    assert(mono_ms() - t0 < 1000 && release_calls == 1 && !log_has("did not stop"));
    return 0;
}
''')

    def test_a_worker_stuck_in_a_d3d12_call_is_leaked_not_freed(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234); struct ffence *f = last_fence(); DlssnrFenceShared *s = last_shared;
    f->hold_signal = 1;
    assert(do_signal(sem, 5, 0) == 0);
    set_ready(s, 5);
    assert(WAIT_UNTIL(f->signal_held, 2000));           /* the worker is inside Signal and stays there */
    long long t0 = mono_ms();
    assert(hipDestroyExternalSemaphore(sem) == 0);
    assert(mono_ms() - t0 < 2000);
    assert(log_has("worker did not stop in"));
    assert(release_calls == 0 && !f->released);          /* nothing freed under a running thread */
    f->hold_signal = 0;
    settle(100);
    return 0;
}
''', defines='#define PE_FENCE_JOIN_MS 300\n')

    def test_a_full_wait_table_refuses_the_wait_without_enqueuing_it(self):
        self.run_c(r'''
int main(void) {
    install();
    void *sem = import_fence((HANDLE)0x1234);
    for (u64 v = 100; v < 132; v++) assert(do_wait(sem, v, 0) == 0);
    assert(nwaits_seen == 32);
    assert(do_wait(sem, 200, 0) == 2 && nwaits_seen == 32 && log_has("wait table full; refusing wait 200"));
    assert(hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_arrays_with_a_live_fence_split_per_element_and_unregistered_handles_go_to_hip(self):
        self.run_c(r'''
int main(void) {
    install();
    /* no PE fence is live: the whole call, arrays and count intact, goes to HIP */
    void *plain[3] = { (void *)0x10, (void *)0x11, (void *)0x12 };
    struct wparams q[3] = { {.value = 1}, {.value = 2}, {.value = 3} };
    assert(hipWaitExternalSemaphoresAsync(plain, q, 3, (void *)0x8888) == 57);
    assert(fwd_wait_sems == plain && fwd_wait_params == q && fwd_wait_n == 3 && fwd_wait_stream == (void *)0x8888);
    assert(hipSignalExternalSemaphoresAsync(plain, q, 3, 0) == 58 && fwd_sig_sems == plain && fwd_sig_n == 3);
    fwd_wait_calls = 0; fwd_signal_calls = 0;

    void *sem = import_fence((HANDLE)0x1234);
    void *sems[3] = { (void *)0xf00, sem, (void *)0xf01 };
    struct wparams p[3] = { { .value = 11 }, { .value = 5 }, { .value = 13 } };
    /* the first foreign result is returned; the fence's own element still ran */
    assert(hipWaitExternalSemaphoresAsync(sems, p, 3, (void *)0x8888) == 57);
    assert(fwd_wait_calls == 2 && fwd_wait_sems == &sems[2] && fwd_wait_params == &p[2] && fwd_wait_n == 1);
    assert(nwaits_seen == 1 && waits_seen[0].value == 5);
    assert(hipSignalExternalSemaphoresAsync(sems, p, 3, 0) == 58);
    assert(fwd_signal_calls == 2 && fwd_sig_sems == &sems[2] && fwd_sig_params == &p[2] && fwd_sig_n == 1);
    assert(nsigs_seen == 1 && sigs_seen[0].value == 5);
    assert(hipDestroyExternalSemaphore((void *)0xf00) == 56 && fwd_destroyed == (void *)0xf00);
    assert(hipDestroyExternalSemaphore(sem) == 0);
    return 0;
}
''')

    def test_pe_log_formats_never_use_pointer_or_string_conversions(self):
        # The bridge formats them with three u64 arguments; %s or %p would read garbage.
        text = PE_FENCE.read_text()
        calls = re.findall(r'pe_fence_(?:log|trace|err)\s*\([^"]*"([^"]*)"', text)
        self.assertTrue(calls)
        for fmt in calls:
            with self.subTest(fmt=fmt):
                self.assertNotRegex(fmt, r'%(?!(?:llu|llx|%))')


def idl_methods(name, text):
    """Method names of an IDL interface, parents first, in vtable order."""
    m = re.search(r'interface\s+%s\s*:\s*(\w+)\s*\{' % name, text)
    if not m:
        return [], None
    start = m.end()
    depth, i = 1, start
    while depth:
        depth += {'{': 1, '}': -1}.get(text[i], 0)
        i += 1
    body = re.sub(r'//[^\n]*', '', text[start:i - 1])
    body = re.sub(r'\[[^\]]*\]', '', body)
    names, level, chunk = [], 0, ''
    for ch in body:
        if ch == '(':
            level += 1
        elif ch == ')':
            level -= 1
        if ch == ';' and level == 0:
            mm = re.search(r'(\w+)\s*\(', chunk)
            if mm and mm.group(1) != 'cpp_quote':
                names.append(mm.group(1))
            chunk = ''
        else:
            chunk += ch
    return names, m.group(1)


def vtable(name, text):
    own, parent = idl_methods(name, text)
    if parent == 'IUnknown':
        return ['QueryInterface', 'AddRef', 'Release'] + own
    return vtable(parent, text) + own


def guid_bytes(uuid):
    a, b, c, d, e = uuid.split('-')
    raw = bytes.fromhex(a)[::-1] + bytes.fromhex(b)[::-1] + bytes.fromhex(c)[::-1] + bytes.fromhex(d) + bytes.fromhex(e)
    return list(raw)


class VtableLayout(unittest.TestCase):
    @unittest.skipUnless(IDL.is_file(), 'vkd3d source tree absent: run sources/fetch_vkd3d.py build/vkd3d-source')
    def test_vtable_slots_and_iids_match_the_d3d12_idl(self):
        idl = IDL.read_text()
        src = PE_FENCE.read_text()

        def define(n):
            return int(re.search(r'#define\s+%s\s+(\d+)' % n, src).group(1))

        dev = vtable('ID3D12Device', idl)
        fence = vtable('ID3D12Fence', idl)
        self.assertEqual(dev.index('Release'), define('DEV_RELEASE'))
        self.assertEqual(dev.index('OpenSharedHandle'), define('DEV_OPEN_SHARED_HANDLE'))
        self.assertEqual(dev.index('GetAdapterLuid'), define('DEV_GET_ADAPTER_LUID'))
        self.assertEqual(fence.index('Release'), define('FENCE_RELEASE'))
        self.assertEqual(fence.index('GetCompletedValue'), define('FENCE_GET_COMPLETED_VALUE'))
        self.assertEqual(fence.index('SetEventOnCompletion'), define('FENCE_SET_EVENT_ON_COMPLETION'))
        self.assertEqual(fence.index('Signal'), define('FENCE_SIGNAL'))
        for iface, const in (('ID3D12Device', 'IID_ID3D12Device'), ('ID3D12Fence', 'IID_ID3D12Fence')):
            uuid = re.search(r'uuid\(([0-9a-f-]{36})\)[^{}]*?interface\s+%s\s*:' % iface, idl, re.S).group(1)
            decl = re.search(r'%s\[16\]\s*=\s*\{([^}]*)\}' % const, src, re.S).group(1)
            have = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]{2}', decl)]
            self.assertEqual(have, guid_bytes(uuid), const)


if __name__ == '__main__':
    unittest.main()
