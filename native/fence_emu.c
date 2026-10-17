/* D3D12 fence emulation, included by hip_bridge.c.
 *
 * Linux ROCm refuses hipImportExternalSemaphore for a DRM timeline syncobj fd
 * under every handle type (timelineSemaphoreInteropSupported is 0), so a
 * Windows HIP consumer that shares a D3D12 fence with HIP (handle type 4) gets
 * the semaphore from here instead. The kernel's timeline syncobj ioctls work
 * fine; only HIP's interop is missing.
 *
 *   import   NT handle -> Wine's local d3dkmt sync object -> its syncobj fd ->
 *            syncobj handle on the HIP device's render node.
 *   wait(V)  hipStreamWaitValue64 on a host-mapped counter, with a helper
 *            thread that waits on the syncobj and then raises the counter. The
 *            caller never blocks and no HIP call runs on the helper thread.
 *   signal   hipLaunchHostFunc; the callback only does the syncobj signal ioctl.
 *   destroy  releases every GPU wait, drains the streams that used the
 *            semaphore, then frees. Callbacks still queued keep the syncobj
 *            alive through a reference count.
 *
 * The DRM calls are raw ioctls so the preloaded bridge does not gain a libdrm
 * dependency inside the Proton container. */

#define FENCE_D3D12_TYPE 4
#define FENCE_PARAMS_SIZE 144 /* sizeof hipExternalSemaphore{Wait,Signal}Params */
#define FENCE_HIP_INVALID_VALUE 1
#define FENCE_HIP_OUT_OF_MEMORY 2
#define FENCE_HIP_NOT_INITIALIZED 3
#define FENCE_HIP_INVALID_HANDLE 400
#define FENCE_HIP_NOT_SUPPORTED 801
#define FENCE_HIP_NOT_READY 600
#define FENCE_HOST_MALLOC_FLAGS (0x2u /* mapped */ | 0x40000000u /* coherent */)
#define FENCE_STREAM_WAIT_GTE 0u
#define FENCE_MAX_STREAMS 16
#define FENCE_MAGIC 0x46454e43u

/* Kernel UAPI (drm.h), copied so the build needs no kernel or libdrm headers. */
struct fence_syncobj_destroy { uint32_t handle, pad; };
struct fence_syncobj_handle { uint32_t handle, flags; int32_t fd; uint32_t pad; u64 point; };
struct fence_syncobj_timeline_wait {
    u64 handles, points;
    int64_t timeout_nsec;
    uint32_t count_handles, flags, first_signaled, pad;
    u64 deadline_nsec;
};
struct fence_syncobj_timeline_array { u64 handles, points; uint32_t count_handles, flags; };
struct fence_get_cap { u64 capability, value; };
#define FENCE_IOCTL_GET_CAP _IOWR('d', 0x0c, struct fence_get_cap)
#define FENCE_IOCTL_DESTROY _IOWR('d', 0xC0, struct fence_syncobj_destroy)
#define FENCE_IOCTL_FD_TO_HANDLE _IOWR('d', 0xC2, struct fence_syncobj_handle)
#define FENCE_IOCTL_TIMELINE_WAIT _IOWR('d', 0xCA, struct fence_syncobj_timeline_wait)
#define FENCE_IOCTL_QUERY _IOWR('d', 0xCB, struct fence_syncobj_timeline_array)
#define FENCE_IOCTL_TIMELINE_SIGNAL _IOWR('d', 0xCD, struct fence_syncobj_timeline_array)
#define FENCE_WAIT_ALL (1u << 0)
#define FENCE_WAIT_FOR_SUBMIT (1u << 1)
#define FENCE_CAP_SYNCOBJ_TIMELINE 0x14

struct fence_ops {
    /* NT handle -> syncobj fd the caller owns; -ENOSYS if Wine's helpers are
     * missing, -EBADF if the handle is not a sync object. */
    int (*nt_to_fd)(void *nt_handle);
    int (*open_node)(const char *path);
    int (*timeline_cap)(int drm);
    int (*fd_to_handle)(int drm, int fd, unsigned *handle);
    int (*destroy)(int drm, unsigned handle);
    /* absolute CLOCK_MONOTONIC deadline; 0 reached, -ETIME not yet */
    int (*wait)(int drm, unsigned handle, u64 point, long long abs_ns);
    int (*signal)(int drm, unsigned handle, u64 point);
    int (*query)(int drm, unsigned handle, u64 *point);
};

static long long fence_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000LL + t.tv_nsec;
}

static int fence_open_node(const char *path) {
    return open(path, O_RDWR | O_CLOEXEC);
}

static int fence_drm_timeline_cap(int drm) {
    struct fence_get_cap cap = { FENCE_CAP_SYNCOBJ_TIMELINE, 0 };
    return ioctl(drm, FENCE_IOCTL_GET_CAP, &cap) == 0 && cap.value == 1;
}

static int fence_drm_fd_to_handle(int drm, int fd, unsigned *handle) {
    struct fence_syncobj_handle a = { .fd = fd };
    if (ioctl(drm, FENCE_IOCTL_FD_TO_HANDLE, &a))
        return -errno;
    *handle = a.handle;
    return 0;
}

static int fence_drm_destroy(int drm, unsigned handle) {
    struct fence_syncobj_destroy a = { .handle = handle };
    return ioctl(drm, FENCE_IOCTL_DESTROY, &a) ? -errno : 0;
}

/* WAIT_FOR_SUBMIT: the consumer may wait for a value before anyone has
 * submitted the signal that produces it. */
static int fence_drm_wait(int drm, unsigned handle, u64 point, long long abs_ns) {
    u64 h = handle;
    struct fence_syncobj_timeline_wait a = {
        .handles = (uintptr_t)&h, .points = (uintptr_t)&point, .timeout_nsec = abs_ns,
        .count_handles = 1, .flags = FENCE_WAIT_ALL | FENCE_WAIT_FOR_SUBMIT,
    };
    if (ioctl(drm, FENCE_IOCTL_TIMELINE_WAIT, &a) == 0)
        return 0;
    /* A signal interrupting the wait is not a failure; the helper just waits again. */
    return errno == ETIME || errno == EINTR ? -ETIME : -errno;
}

static int fence_drm_signal(int drm, unsigned handle, u64 point) {
    u64 h = handle;
    struct fence_syncobj_timeline_array a = {
        .handles = (uintptr_t)&h, .points = (uintptr_t)&point, .count_handles = 1,
    };
    return ioctl(drm, FENCE_IOCTL_TIMELINE_SIGNAL, &a) ? -errno : 0;
}

static int fence_drm_query(int drm, unsigned handle, u64 *point) {
    u64 h = handle;
    struct fence_syncobj_timeline_array a = {
        .handles = (uintptr_t)&h, .points = (uintptr_t)point, .count_handles = 1,
    };
    return ioctl(drm, FENCE_IOCTL_QUERY, &a) ? -errno : 0;
}

/* Wine keeps D3D12 shared-fence objects as d3dkmt sync objects; win32u's own
 * GL semaphore import (import_semaphore) opens one with d3dkmt_open_sync(0,
 * handle), asks for its host fd and drops the local object. Same sequence. */
static int d3dkmt_sync_handle_to_fd(void *nt_handle) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static int resolved;
    static d3dkmt_open_sync_t open_sync;
    static d3dkmt_object_get_fd_t get_fd;
    static d3dkmt_destroy_sync_t destroy_sync;
    d3dkmt_handle_t local;
    int fd;

    pthread_mutex_lock(&mu);
    if (!resolved) {
        void *h = dlopen("win32u.so", RTLD_NOW | RTLD_NOLOAD);
        resolved = 1;
        if (!h)
            h = dlopen("win32u.so", RTLD_NOW);
        open_sync = (d3dkmt_open_sync_t)elf_local_sym(h, "d3dkmt_open_sync");
        get_fd = (d3dkmt_object_get_fd_t)elf_local_sym(h, "d3dkmt_object_get_fd");
        destroy_sync = (d3dkmt_destroy_sync_t)elf_local_sym(h, "d3dkmt_destroy_sync");
        logmsg("d3dkmt sync win32u=%p open=%p get_fd=%p destroy=%p", h, (void *)open_sync,
               (void *)get_fd, (void *)destroy_sync);
    }
    pthread_mutex_unlock(&mu);
    if (!open_sync || !get_fd || !destroy_sync)
        return -ENOSYS;
    /* One line per win32u call: this path has stopped mid-way in games, and
     * the last line written shows which call it was. */
    logmsg("d3dkmt sync open handle=%p", nt_handle);
    local = open_sync(0, nt_handle);
    if (!local) {
        logmsg("d3dkmt_open_sync failed handle=%p", nt_handle);
        return -EBADF;
    }
    logmsg("d3dkmt sync get_fd local=%u", local);
    fd = get_fd(local);
    logmsg("d3dkmt sync destroy local=%u fd=%d", local, fd);
    destroy_sync(local);
    logmsg("d3dkmt sync local=%u fd=%d", local, fd);
    return fd < 0 ? -EBADF : fd;
}

static struct fence_ops fence_ops = {
    d3dkmt_sync_handle_to_fd, fence_open_node, fence_drm_timeline_cap, fence_drm_fd_to_handle,
    fence_drm_destroy, fence_drm_wait, fence_drm_signal, fence_drm_query,
};

/* Tests point these at a scratch tree. */
static const char *fence_sysfs_root = "/sys/class/drm";
static const char *fence_dev_root = "/dev/dri";
/* How long destroy waits for streams to drain, and how long the helper blocks
 * in one syncobj wait before it re-checks for new work and shutdown. */
static unsigned fence_idle_timeout_ms = 2000;
static unsigned fence_slice_ms = 100;

/* The render node whose PCI device is the one HIP is using. */
static int fence_render_node(char *out, size_t size) {
    char bus[64], link[512], target[512];
    struct dirent **names;
    const char *override = getenv("DLSSNR_DRM_RENDER_NODE");
    int device = 0, n, i, found = -1;

    if (override && override[0]) {
        snprintf(out, size, "%s", override);
        return 0;
    }
    if (!real.p_hipGetDevice || !p_hipDeviceGetPCIBusId || real.p_hipGetDevice(&device) ||
        p_hipDeviceGetPCIBusId(bus, sizeof bus, device)) {
        logmsg("fence: cannot read the HIP device's PCI address; set DLSSNR_DRM_RENDER_NODE");
        return -1;
    }
    n = scandir(fence_sysfs_root, &names, NULL, alphasort);
    for (i = 0; i < n; i++) {
        ssize_t len;
        const char *base;
        if (found < 0 && !strncmp(names[i]->d_name, "renderD", 7)) {
            snprintf(link, sizeof link, "%s/%s/device", fence_sysfs_root, names[i]->d_name);
            len = readlink(link, target, sizeof target - 1);
            if (len > 0) {
                target[len] = 0;
                base = strrchr(target, '/');
                base = base ? base + 1 : target;
                if (!strcasecmp(base, bus)) {
                    snprintf(out, size, "%s/%s", fence_dev_root, names[i]->d_name);
                    found = 0;
                }
            }
        }
        free(names[i]);
    }
    if (n >= 0)
        free(names);
    if (found)
        logmsg("fence: no render node for PCI %s under %s", bus, fence_sysfs_root);
    return found;
}

struct fence_sem {
    uint32_t magic;
    struct fence_sem *next;
    int refs; /* the import, plus each queued signal callback and in-flight call */
    int drm_fd;
    unsigned handle;
    volatile u64 *flag; /* host-mapped; the GPU waits for flag >= value */
    void *flag_dev;
    pthread_t thread;
    int thread_started;
    pthread_mutex_t mu; /* queue, stopping and the stream list */
    pthread_cond_t cv;
    u64 *queue;
    size_t count, cap;
    int stopping;
    void *streams[FENCE_MAX_STREAMS];
    int nstreams;
    int sync_device; /* more distinct streams than the list holds */
};

static struct fence_sem *fence_list;
static pthread_mutex_t fence_list_mu = PTHREAD_MUTEX_INITIALIZER;

static u64 fence_flag_load(struct fence_sem *s) {
    return __atomic_load_n((u64 *)s->flag, __ATOMIC_ACQUIRE);
}

/* Values only ever move forward, so a late store cannot un-release a wait. */
static void fence_flag_raise(struct fence_sem *s, u64 v) {
    u64 cur = fence_flag_load(s);
    while (cur < v && !__atomic_compare_exchange_n((u64 *)s->flag, &cur, v, 0, __ATOMIC_RELEASE,
                                                   __ATOMIC_ACQUIRE))
        ;
}

static struct fence_sem *fence_acquire(const void *p) {
    struct fence_sem *s;
    pthread_mutex_lock(&fence_list_mu);
    for (s = fence_list; s && s != p; s = s->next)
        ;
    if (s)
        __atomic_add_fetch(&s->refs, 1, __ATOMIC_ACQ_REL);
    pthread_mutex_unlock(&fence_list_mu);
    return s;
}

static void fence_release(struct fence_sem *s) {
    if (__atomic_sub_fetch(&s->refs, 1, __ATOMIC_ACQ_REL))
        return;
    fence_ops.destroy(s->drm_fd, s->handle);
    close(s->drm_fd);
    pthread_mutex_destroy(&s->mu);
    pthread_cond_destroy(&s->cv);
    free(s->queue);
    s->magic = 0;
    free(s);
}

/* Smallest requested value above what the counter already covers; drops
 * everything at or below the counter. Caller holds s->mu. */
static int fence_next_value(struct fence_sem *s, u64 *out) {
    u64 covered = fence_flag_load(s);
    size_t i, keep = 0;
    int have = 0;
    for (i = 0; i < s->count; i++) {
        if (s->queue[i] <= covered)
            continue;
        s->queue[keep++] = s->queue[i];
        if (!have || s->queue[i] < *out) {
            *out = s->queue[i];
            have = 1;
        }
    }
    s->count = keep;
    return have;
}

/* Waits are served smallest value first, so a wait for 5 is not held up behind
 * an earlier-queued wait for 9 on another stream. The syncobj wait is cut into
 * short slices so a smaller value that arrives meanwhile is picked up. */
static void *fence_helper(void *arg) {
    struct fence_sem *s = arg;
    unsigned long stalled = 0, errors = 0;
    u64 waiting = 0;

    pthread_mutex_lock(&s->mu);
    for (;;) {
        u64 v = 0;
        int r;
        while (!s->stopping && !fence_next_value(s, &v))
            pthread_cond_wait(&s->cv, &s->mu);
        if (s->stopping)
            break;
        pthread_mutex_unlock(&s->mu);
        r = fence_ops.wait(s->drm_fd, s->handle, v, fence_now_ns() + fence_slice_ms * 1000000LL);
        pthread_mutex_lock(&s->mu);
        if (r == -ETIME) {
            if (v != waiting) {
                waiting = v;
                stalled = 0;
            }
            if (++stalled % (5000 / fence_slice_ms) == 0)
                logmsg("fence: still waiting for value %llu (%lu s)", (unsigned long long)v,
                       stalled * fence_slice_ms / 1000);
            continue;
        }
        if (r) {
            /* A broken syncobj must not wedge the stream forever; release it
             * and say so. */
            if (errors++ < 8 || errors % 1000 == 0)
                logmsg("fence: wait for value %llu failed err=%d; releasing the stream",
                       (unsigned long long)v, r);
        }
        fence_flag_raise(s, v);
    }
    pthread_mutex_unlock(&s->mu);
    return NULL;
}

static void fence_note_stream(struct fence_sem *s, void *stream) {
    int i;
    for (i = 0; i < s->nstreams; i++)
        if (s->streams[i] == stream)
            return;
    if (s->nstreams < FENCE_MAX_STREAMS)
        s->streams[s->nstreams++] = stream;
    else
        s->sync_device = 1;
}

static int fence_wait_one(struct fence_sem *s, u64 v, void *stream) {
    u64 last = 0;
    int err;

    if (!v || fence_flag_load(s) >= v)
        return 0;
    /* Already signalled: skip the GPU wait and the helper entirely. */
    if (fence_ops.query(s->drm_fd, s->handle, &last) == 0 && last >= v) {
        fence_flag_raise(s, v);
        return 0;
    }
    pthread_mutex_lock(&s->mu);
    if (s->count == s->cap) {
        size_t cap = s->cap ? s->cap * 2 : 8;
        u64 *q = realloc(s->queue, cap * sizeof *q);
        if (!q) {
            pthread_mutex_unlock(&s->mu);
            return FENCE_HIP_OUT_OF_MEMORY;
        }
        s->queue = q;
        s->cap = cap;
    }
    s->queue[s->count++] = v;
    fence_note_stream(s, stream);
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    err = p_hipStreamWaitValue64(stream, s->flag_dev, v, FENCE_STREAM_WAIT_GTE, ~0ULL);
    if (err)
        logmsg("fence: hipStreamWaitValue64 failed err=%d", err);
    return err;
}

struct fence_signal {
    struct fence_sem *sem;
    u64 value;
};

/* Runs on a HIP callback thread, so it must not call HIP. */
static void fence_signal_cb(void *p) {
    struct fence_signal *c = p;
    int r = fence_ops.signal(c->sem->drm_fd, c->sem->handle, c->value);
    if (r)
        logmsg("fence: timeline signal of %llu failed err=%d", (unsigned long long)c->value, r);
    fence_release(c->sem);
    free(c);
}

static int fence_signal_one(struct fence_sem *s, u64 v, void *stream) {
    struct fence_signal *c = malloc(sizeof *c);
    int err;
    if (!c)
        return FENCE_HIP_OUT_OF_MEMORY;
    c->sem = s;
    c->value = v;
    __atomic_add_fetch(&s->refs, 1, __ATOMIC_ACQ_REL);
    err = p_hipLaunchHostFunc(stream, fence_signal_cb, c);
    if (err) {
        logmsg("fence: hipLaunchHostFunc failed err=%d", err);
        __atomic_sub_fetch(&s->refs, 1, __ATOMIC_ACQ_REL);
        free(c);
    }
    return err;
}

/* Takes ownership of sync_fd. */
static int fence_import_syncobj_fd(void **out, int sync_fd) {
    char node[512];
    struct fence_sem *s = NULL;
    void *host = NULL;
    int err = FENCE_HIP_NOT_SUPPORTED;

    if (fence_render_node(node, sizeof node))
        goto fail;
    s = calloc(1, sizeof *s);
    if (!s) {
        err = FENCE_HIP_OUT_OF_MEMORY;
        goto fail;
    }
    s->magic = FENCE_MAGIC;
    s->refs = 1;
    s->drm_fd = fence_ops.open_node(node);
    if (s->drm_fd < 0) {
        logmsg("fence: cannot open %s: %s", node, strerror(errno));
        goto fail;
    }
    if (!fence_ops.timeline_cap(s->drm_fd)) {
        logmsg("fence: %s has no timeline syncobj support", node);
        goto fail;
    }
    if (fence_ops.fd_to_handle(s->drm_fd, sync_fd, &s->handle)) {
        logmsg("fence: syncobj import of fd %d on %s failed", sync_fd, node);
        err = FENCE_HIP_INVALID_HANDLE;
        goto fail;
    }
    close(sync_fd);
    sync_fd = -1;
    if (real.p_hipHostMalloc(&host, sizeof *s->flag, FENCE_HOST_MALLOC_FLAGS) ||
        real.p_hipHostGetDevicePointer(&s->flag_dev, host, 0)) {
        logmsg("fence: cannot allocate the host-mapped wait counter");
        err = FENCE_HIP_OUT_OF_MEMORY;
        goto fail;
    }
    s->flag = host;
    *s->flag = 0;
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv, NULL);
    if (pthread_create(&s->thread, NULL, fence_helper, s)) {
        err = FENCE_HIP_OUT_OF_MEMORY;
        goto fail_mu;
    }
    s->thread_started = 1;
    pthread_mutex_lock(&fence_list_mu);
    s->next = fence_list;
    fence_list = s;
    pthread_mutex_unlock(&fence_list_mu);
    logmsg("fence: imported syncobj %u on %s as %p", s->handle, node, (void *)s);
    *out = s;
    return 0;

fail_mu:
    pthread_mutex_destroy(&s->mu);
    pthread_cond_destroy(&s->cv);
fail:
    if (host)
        real.p_hipHostFree(host);
    if (sync_fd >= 0)
        close(sync_fd);
    if (s) {
        if (s->handle)
            fence_ops.destroy(s->drm_fd, s->handle);
        if (s->drm_fd > 0)
            close(s->drm_fd);
        free(s);
    }
    return err;
}

static int fence_import_nt(void **out, void *nt_handle) {
    int fd;
    if (!real.p_hipHostMalloc || !real.p_hipHostGetDevicePointer || !real.p_hipHostFree)
        return FENCE_HIP_NOT_INITIALIZED;
    if (!p_hipStreamWaitValue64 || !p_hipLaunchHostFunc) {
        logmsg("fence: runtime lacks hipStreamWaitValue64 or hipLaunchHostFunc");
        return FENCE_HIP_NOT_SUPPORTED;
    }
    if (!nt_handle) {
        logmsg("fence: D3D12 fence import without a handle");
        return FENCE_HIP_INVALID_VALUE;
    }
    fd = fence_ops.nt_to_fd(nt_handle);
    if (fd == -ENOSYS) {
        logmsg("fence: win32u has no d3dkmt sync helpers; D3D12 fences cannot be imported");
        return FENCE_HIP_NOT_SUPPORTED;
    }
    if (fd < 0) {
        logmsg("fence: handle %p is not a shareable sync object", nt_handle);
        return FENCE_HIP_INVALID_HANDLE;
    }
    return fence_import_syncobj_fd(out, fd);
}

/* Destroying a semaphore the GPU may still be waiting on: raise the counter
 * past anything waited for, then let the streams drain before the memory goes.
 * A stream that never drains (blocked behind some other semaphore) is given
 * fence_idle_timeout_ms, after which the counter page is leaked rather than
 * freed under the GPU. */
static void fence_destroy(struct fence_sem *s) {
    struct fence_sem **pp;
    int i, drained = 1;

    pthread_mutex_lock(&fence_list_mu);
    for (pp = &fence_list; *pp && *pp != s; pp = &(*pp)->next)
        ;
    if (*pp)
        *pp = s->next;
    pthread_mutex_unlock(&fence_list_mu);

    pthread_mutex_lock(&s->mu);
    s->stopping = 1;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    fence_flag_raise(s, ~0ULL);
    if (s->thread_started)
        pthread_join(s->thread, NULL);

    for (i = 0; i < s->nstreams; i++) {
        long long end = fence_now_ns() + fence_idle_timeout_ms * 1000000LL;
        int r = 0;
        while (real.p_hipStreamQuery && (r = real.p_hipStreamQuery(s->streams[i])) == FENCE_HIP_NOT_READY &&
               fence_now_ns() < end)
            usleep(1000);
        if (real.p_hipStreamQuery && r == FENCE_HIP_NOT_READY)
            drained = 0;
    }
    if (s->sync_device && real.p_hipDeviceSynchronize)
        real.p_hipDeviceSynchronize();
    if (drained)
        real.p_hipHostFree((void *)s->flag);
    else
        logmsg("fence: stream still busy at destroy; leaking the wait counter of %p", (void *)s);
    fence_release(s); /* the import's reference; queued callbacks hold the rest */
}

static int stub_impsem(void **out, const void *desc) {
    /* hipExternalSemaphoreHandleDesc: int type, union handle at +8 */
    struct { int type; int pad; void *handle; const void *name; } const *d = desc;
    if (ensure_hip())
        return FENCE_HIP_NOT_INITIALIZED;
    if (!out || !d)
        return FENCE_HIP_INVALID_VALUE;
    if (d->type == FENCE_D3D12_TYPE)
        return fence_import_nt(out, d->handle);
    if (!real.p_hipImportExternalSemaphore)
        return FENCE_HIP_NOT_INITIALIZED;
    return real.p_hipImportExternalSemaphore(out, desc);
}

static int stub_desem(void *sem) {
    struct fence_sem *s;
    if (ensure_hip())
        return FENCE_HIP_NOT_INITIALIZED;
    s = fence_acquire(sem);
    if (!s) {
        if (!real.p_hipDestroyExternalSemaphore)
            return FENCE_HIP_NOT_INITIALIZED;
        return real.p_hipDestroyExternalSemaphore(sem);
    }
    __atomic_sub_fetch(&s->refs, 1, __ATOMIC_ACQ_REL); /* undo our lookup reference */
    fence_destroy(s);
    return 0;
}

/* Wait and signal share one walk: emulated semaphores are handled here, the
 * rest go to HIP one at a time with their own parameter block. The first
 * error is returned, but every element is still attempted. */
static int fence_each(const void *sems, const void *params, unsigned n, void *stream, int is_wait) {
    void *const *list = sems;
    int (*forward)(const void *, const void *, unsigned, void *) =
        is_wait ? real.p_hipWaitExternalSemaphoresAsync : real.p_hipSignalExternalSemaphoresAsync;
    int first = 0;
    unsigned i;

    if (ensure_hip())
        return FENCE_HIP_NOT_INITIALIZED;
    if ((n && (!sems || !params)))
        return FENCE_HIP_INVALID_VALUE;
    for (i = 0; i < n; i++) {
        const char *param = (const char *)params + (size_t)i * FENCE_PARAMS_SIZE;
        struct fence_sem *s = fence_acquire(list[i]);
        int err;
        if (s) {
            u64 v = *(const u64 *)param;
            err = is_wait ? fence_wait_one(s, v, stream) : fence_signal_one(s, v, stream);
            fence_release(s);
        } else if (forward) {
            err = forward(&list[i], param, 1, stream);
        } else {
            err = FENCE_HIP_NOT_INITIALIZED;
        }
        if (err && !first)
            first = err;
    }
    return first;
}

static int stub_waitsem(const void *sems, const void *params, unsigned n, void *stream) {
    return fence_each(sems, params, n, stream, 1);
}

static int stub_signalsem(const void *sems, const void *params, unsigned n, void *stream) {
    return fence_each(sems, params, n, stream, 0);
}
