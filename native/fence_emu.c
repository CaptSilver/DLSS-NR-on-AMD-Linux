/* Counter half of the D3D12 fence emulation, included by hip_bridge.c.
 *
 * A Windows HIP consumer shares a D3D12 fence with HIP (handle type 4), but
 * Linux ROCm cannot import one, and the fence's value lives inside Wine, where
 * only Win32 calls can reach it. So the PE trampoline opens the fence with
 * ID3D12Device::OpenSharedHandle and runs a worker thread on it; this file
 * keeps the part that needs HIP:
 *
 *   wait(V)   hipStreamWaitValue64 on a host-mapped counter. The PE worker
 *             raises the counter once the fence reaches V, which releases the
 *             stream. Nothing here blocks the caller.
 *   signal(V) hipLaunchHostFunc; when the stream reaches it, the callback
 *             raises signal_ready and wakes the PE side's relay thread, which
 *             sleeps in fence_await_ready and passes the value on to the fence.
 *             The callback runs on a ROCm thread, so it touches an atomic and a
 *             condition variable and nothing else: no HIP, and no Win32 (that
 *             thread has no TEB).
 *   drain     raises the counter past anything waited for and lets the streams
 *             that used it go idle, so the memory can be freed.
 *
 * The struct the PE side sees is DlssnrFenceShared; it is the first member of
 * fence_sem, so the two pointers are the same. A reference count (the import
 * plus one per queued callback) keeps it alive for a callback that fires after
 * the import was released. */

#define FENCE_HIP_INVALID_VALUE 1
#define FENCE_HIP_OUT_OF_MEMORY 2
#define FENCE_HIP_NOT_INITIALIZED 3
#define FENCE_HIP_NOT_SUPPORTED 801
#define FENCE_HIP_NOT_READY 600
#define FENCE_HOST_MALLOC_FLAGS (0x2u /* mapped */ | 0x40000000u /* coherent */)
#define FENCE_STREAM_WAIT_GTE 0u
#define FENCE_MAX_STREAMS 16

struct fence_sem {
    DlssnrFenceShared pub; /* first, so a DlssnrFenceShared * is also a fence_sem * */
    pthread_mutex_t mu;    /* guards the stream list */
    pthread_mutex_t ready_mu; /* with ready_cv: wakes fence_await_ready */
    pthread_cond_t ready_cv;
    void *streams[FENCE_MAX_STREAMS];
    int nstreams;
    int sync_device; /* more distinct streams than the list holds */
};

/* How long drain waits for the streams to go idle. */
static unsigned fence_idle_timeout_ms = 2000;

static long long fence_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000LL + t.tv_nsec;
}

/* Rate-limited success trace: the first calls and every 1000th after, enough
 * to see whether a game's waits and signals arrive at all. */
static unsigned long fence_trace_count;
#define fence_trace(...)                                                              \
    do {                                                                              \
        unsigned long n_ = __atomic_add_fetch(&fence_trace_count, 1, __ATOMIC_RELAXED); \
        if (n_ <= 64 || n_ % 1000 == 0)                                               \
            logmsg(__VA_ARGS__);                                                      \
    } while (0)

/* Values only ever move forward, so a late store cannot un-release a wait. */
static void fence_max_raise(volatile u64 *slot, u64 v) {
    u64 cur = __atomic_load_n((u64 *)slot, __ATOMIC_ACQUIRE);
    while (cur < v && !__atomic_compare_exchange_n((u64 *)slot, &cur, v, 0, __ATOMIC_RELEASE,
                                                   __ATOMIC_ACQUIRE))
        ;
}

static void fence_unref(struct fence_sem *s) {
    if (__atomic_sub_fetch(&s->pub.refs, 1, __ATOMIC_ACQ_REL))
        return;
    pthread_mutex_destroy(&s->mu);
    pthread_mutex_destroy(&s->ready_mu);
    pthread_cond_destroy(&s->ready_cv);
    s->pub.magic = 0;
    free(s);
}

static int fence_create(DlssnrFenceShared **out) {
    struct fence_sem *s;
    void *host = NULL;

    if (ensure_hip())
        return FENCE_HIP_NOT_INITIALIZED;
    if (!out)
        return FENCE_HIP_INVALID_VALUE;
    *out = NULL;
    if (!real.p_hipHostMalloc || !real.p_hipHostGetDevicePointer || !real.p_hipHostFree)
        return FENCE_HIP_NOT_INITIALIZED;
    if (!p_hipStreamWaitValue64 || !p_hipLaunchHostFunc) {
        logmsg("fence: runtime lacks hipStreamWaitValue64 or hipLaunchHostFunc");
        return FENCE_HIP_NOT_SUPPORTED;
    }
    s = calloc(1, sizeof *s);
    if (!s)
        return FENCE_HIP_OUT_OF_MEMORY;
    if (real.p_hipHostMalloc(&host, sizeof *s->pub.counter, FENCE_HOST_MALLOC_FLAGS) ||
        real.p_hipHostGetDevicePointer(&s->pub.counter_dev, host, 0)) {
        logmsg("fence: cannot allocate the host-mapped wait counter");
        if (host)
            real.p_hipHostFree(host);
        free(s);
        return FENCE_HIP_OUT_OF_MEMORY;
    }
    pthread_mutex_init(&s->mu, NULL);
    pthread_mutex_init(&s->ready_mu, NULL);
    {
        /* Absolute CLOCK_MONOTONIC deadlines: a wall-clock step must not cut or
         * stretch the relay's sleep. */
        pthread_condattr_t attr;
        pthread_condattr_init(&attr);
        pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
        pthread_cond_init(&s->ready_cv, &attr);
        pthread_condattr_destroy(&attr);
    }
    s->pub.counter = host;
    *s->pub.counter = 0;
    s->pub.magic = DLSSNR_FENCE_SHARED_MAGIC;
    s->pub.refs = 1;
    logmsg("fence: counter %p for shared %p", host, (void *)s);
    *out = &s->pub;
    return 0;
}

static void fence_note_stream(struct fence_sem *s, void *stream) {
    int i;
    pthread_mutex_lock(&s->mu);
    for (i = 0; i < s->nstreams; i++)
        if (s->streams[i] == stream)
            goto out;
    if (s->nstreams < FENCE_MAX_STREAMS)
        s->streams[s->nstreams++] = stream;
    else
        s->sync_device = 1;
out:
    pthread_mutex_unlock(&s->mu);
}

/* The PE side has already answered waits the fence satisfies, so this only
 * enqueues the stream wait. */
static int fence_wait(DlssnrFenceShared *f, void *stream, u64 v) {
    struct fence_sem *s = (struct fence_sem *)f;
    int err;

    if (!f || f->magic != DLSSNR_FENCE_SHARED_MAGIC)
        return FENCE_HIP_INVALID_VALUE;
    fence_note_stream(s, stream);
    err = p_hipStreamWaitValue64(stream, f->counter_dev, v, FENCE_STREAM_WAIT_GTE, ~0ULL);
    if (err)
        logmsg("fence: hipStreamWaitValue64 failed err=%d", err);
    else
        fence_trace("fence: wait %llu enqueued stream=%p", (unsigned long long)v, stream);
    return err;
}

struct fence_signal {
    struct fence_sem *sem;
    u64 value;
};

/* Runs on a ROCm callback thread: no HIP calls, no Win32 calls. */
static void fence_signal_cb(void *p) {
    struct fence_signal *c = p;
    fence_max_raise(&c->sem->pub.signal_ready, c->value);
    /* Taking the mutex orders this after a sleeper's check of signal_ready, so
     * the broadcast cannot fall between its check and its wait. */
    pthread_mutex_lock(&c->sem->ready_mu);
    pthread_cond_broadcast(&c->sem->ready_cv);
    pthread_mutex_unlock(&c->sem->ready_mu);
    fence_trace("fence: stream reached signal %llu", (unsigned long long)c->value);
    fence_unref(c->sem);
    free(c);
}

static int fence_signal(DlssnrFenceShared *f, void *stream, u64 v) {
    struct fence_sem *s = (struct fence_sem *)f;
    struct fence_signal *c;
    int err;

    if (!f || f->magic != DLSSNR_FENCE_SHARED_MAGIC)
        return FENCE_HIP_INVALID_VALUE;
    c = malloc(sizeof *c);
    if (!c)
        return FENCE_HIP_OUT_OF_MEMORY;
    /* A stream that only signals still has to be drained: its callback is what
     * hands the value to the PE relay, and the relay stops once the drain is over. */
    fence_note_stream(s, stream);
    c->sem = s;
    c->value = v;
    __atomic_add_fetch(&f->refs, 1, __ATOMIC_ACQ_REL);
    err = p_hipLaunchHostFunc(stream, fence_signal_cb, c);
    if (err) {
        logmsg("fence: hipLaunchHostFunc failed err=%d", err);
        __atomic_sub_fetch(&f->refs, 1, __ATOMIC_ACQ_REL);
        free(c);
    }
    return err;
}

static int fence_await_ready(DlssnrFenceShared *f, u64 after, u32 timeout_ms) {
    struct fence_sem *s = (struct fence_sem *)f;
    struct timespec ts;
    int ready;

    if (!f || f->magic != DLSSNR_FENCE_SHARED_MAGIC)
        return FENCE_HIP_INVALID_VALUE;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }
    pthread_mutex_lock(&s->ready_mu);
    while (__atomic_load_n(&f->signal_ready, __ATOMIC_ACQUIRE) <= after &&
           pthread_cond_timedwait(&s->ready_cv, &s->ready_mu, &ts) == 0)
        ;
    ready = __atomic_load_n(&f->signal_ready, __ATOMIC_ACQUIRE) > after;
    pthread_mutex_unlock(&s->ready_mu);
    return ready ? 0 : FENCE_HIP_NOT_READY;
}

/* Destroying a semaphore the GPU may still be waiting on: raise the counter
 * past anything waited for, then let the streams drain. A stream that never
 * drains (blocked behind some other semaphore) is given fence_idle_timeout_ms;
 * the caller then leaks the counter page rather than free it under the GPU. */
static int fence_drain(DlssnrFenceShared *f) {
    struct fence_sem *s = (struct fence_sem *)f;
    int i, busy = 0;

    if (!f || f->magic != DLSSNR_FENCE_SHARED_MAGIC)
        return FENCE_HIP_INVALID_VALUE;
    fence_max_raise(f->counter, ~0ULL);
    for (i = 0; i < s->nstreams; i++) {
        long long end = fence_now_ns() + fence_idle_timeout_ms * 1000000LL;
        int r = 0;
        while (real.p_hipStreamQuery && (r = real.p_hipStreamQuery(s->streams[i])) == FENCE_HIP_NOT_READY &&
               fence_now_ns() < end)
            usleep(1000);
        if (real.p_hipStreamQuery && r == FENCE_HIP_NOT_READY)
            busy = 1;
    }
    if (s->sync_device && real.p_hipDeviceSynchronize)
        real.p_hipDeviceSynchronize();
    return busy ? FENCE_HIP_NOT_READY : 0;
}

static int fence_release(DlssnrFenceShared *f, int drained) {
    if (!f || f->magic != DLSSNR_FENCE_SHARED_MAGIC)
        return FENCE_HIP_INVALID_VALUE;
    if (drained && real.p_hipHostFree)
        real.p_hipHostFree((void *)f->counter);
    else
        logmsg("fence: stream still busy; leaking the wait counter of %p", (void *)f);
    fence_unref((struct fence_sem *)f);
    return 0;
}

/* PE-side log lines. The format only uses %llu and %llx, so the three values
 * can be passed through whatever the line consumes. */
static void bridge_pe_log(const char *fmt, u64 a, u64 b, u64 c) {
    char line[256];
    snprintf(line, sizeof line, "pe: %s", fmt);
    logmsg(line, a, b, c);
}

/* Every handle type goes to HIP unchanged; D3D12 fences are opened on the PE
 * side and never reach these. */
static int stub_impsem(void **out, const void *desc) {
    if (ensure_hip())
        return FENCE_HIP_NOT_INITIALIZED;
    if (!out || !desc)
        return FENCE_HIP_INVALID_VALUE;
    if (!real.p_hipImportExternalSemaphore)
        return FENCE_HIP_NOT_INITIALIZED;
    return real.p_hipImportExternalSemaphore(out, desc);
}

static int stub_desem(void *sem) {
    if (ensure_hip())
        return FENCE_HIP_NOT_INITIALIZED;
    if (!sem)
        return FENCE_HIP_INVALID_VALUE;
    if (!real.p_hipDestroyExternalSemaphore)
        return FENCE_HIP_NOT_INITIALIZED;
    return real.p_hipDestroyExternalSemaphore(sem);
}

static int stub_waitsem(const void *sems, const void *params, unsigned n, void *stream) {
    if (ensure_hip())
        return FENCE_HIP_NOT_INITIALIZED;
    if (n && (!sems || !params))
        return FENCE_HIP_INVALID_VALUE;
    if (!real.p_hipWaitExternalSemaphoresAsync)
        return FENCE_HIP_NOT_INITIALIZED;
    return real.p_hipWaitExternalSemaphoresAsync(sems, params, n, stream);
}

static int stub_signalsem(const void *sems, const void *params, unsigned n, void *stream) {
    if (ensure_hip())
        return FENCE_HIP_NOT_INITIALIZED;
    if (n && (!sems || !params))
        return FENCE_HIP_INVALID_VALUE;
    if (!real.p_hipSignalExternalSemaphoresAsync)
        return FENCE_HIP_NOT_INITIALIZED;
    return real.p_hipSignalExternalSemaphoresAsync(sems, params, n, stream);
}
