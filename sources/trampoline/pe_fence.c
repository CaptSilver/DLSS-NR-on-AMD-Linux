/* PE half of the D3D12 fence emulation, included by amdhip64_7_pe.c.
 *
 * hipImportExternalSemaphore with handle type 4 receives an NT handle for a
 * D3D12 fence the game created with D3D12_FENCE_FLAG_SHARED. Wine keeps that
 * fence's value in a per-process structure that only D3D12 calls reach, so this
 * side opens the handle as an ID3D12Fence (ID3D12Device::OpenSharedHandle) and
 * runs two threads per fence. An opened fence shares its value with the game's
 * own fence object, so waiting on it and signalling it is the same as the game
 * doing so.
 *
 *   wait(V)    registered here, then the bridge enqueues a stream wait on its
 *              host-mapped counter. The worker raises the counter to the
 *              fence's completed value once the fence gets there.
 *   signal(V)  the bridge enqueues a host callback that sets signal_ready when
 *              the stream reaches it. The relay thread passes that to
 *              Fence->Signal.
 *
 * The ROCm callback thread is not a Wine thread, so it cannot call SetEvent.
 * The relay is a Wine thread that sleeps inside the bridge (on a condition
 * variable that callback wakes) and then makes the D3D12 call, so a signal
 * reaches the game queue without anything polling for it. It is a separate
 * thread because the worker must stay free to serve waits: a worker blocked on
 * a signal that sits behind a later wait in stream order would deadlock.
 * Everything that touches Win32 or D3D12 runs on Wine threads: the caller's
 * (through the exports), the worker and the relay (CreateThread).
 *
 * There is no CRT here: memory comes from VirtualAlloc and the COM interfaces
 * are called through hand-written vtable indices (checked against
 * vkd3d_d3d12.idl by a test). */

#define PECALL __stdcall
#define FENCE_D3D12_TYPE 4
#define FENCE_PARAMS_SIZE 144 /* sizeof hipExternalSemaphore{Wait,Signal}Params */
#define HIP_ERROR_INVALID_HANDLE 400
#define HIP_ERROR_NOT_SUPPORTED 801
#define PE_WAIT_TIMEOUT 0x102u
#define PE_INFINITE 0xFFFFFFFFu
#define PE_PIN_MODULE 0x1u          /* GET_MODULE_HANDLE_EX_FLAG_PIN */
#define PE_FROM_ADDRESS 0x4u        /* GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS */

#define PE_FENCE_MAGIC 0x50454643u /* "PEFC" */
#define PE_FENCE_WAITS 32
#define PE_FENCE_SLOTS 8 /* live imports; lmxxf has one */
#define PE_FENCE_SLICE_MS 1 /* worker poll period while a wait has no event armed; relay retry delay */
#define PE_FENCE_RELAY_SLEEP_MS 50 /* longest the relay sleeps before it looks at `stop` */
#ifndef PE_FENCE_JOIN_MS
#define PE_FENCE_JOIN_MS 5000
#endif

__declspec(dllimport) HANDLE __stdcall CreateThread(void *sa, unsigned long long stack,
                                                    DWORD(__stdcall *fn)(void *), void *arg,
                                                    DWORD flags, DWORD *tid);
__declspec(dllimport) HANDLE __stdcall CreateEventA(void *sa, BOOL manual, BOOL initial,
                                                    const char *name);
__declspec(dllimport) BOOL __stdcall SetEvent(HANDLE h);
__declspec(dllimport) void __stdcall Sleep(DWORD ms);
__declspec(dllimport) DWORD __stdcall WaitForMultipleObjects(DWORD n, const HANDLE *h, BOOL all,
                                                             DWORD ms);
__declspec(dllimport) HANDLE __stdcall LoadLibraryA(const char *name);
__declspec(dllimport) void *__stdcall GetProcAddress(HANDLE module, const char *name);
__declspec(dllimport) BOOL __stdcall GetModuleHandleExA(DWORD flags, const char *name_or_addr,
                                                        HANDLE *module);

/* ---- COM without headers ------------------------------------------------ */

typedef int pe_hr; /* HRESULT; failure is negative */
typedef struct {
    const void *const *vtbl;
} pe_com;
typedef struct {
    u32 low;
    int high; /* not `long`: that is 64 bits in the Linux test build */
} pe_luid;

/* Slot indices from include/vkd3d_d3d12.idl: IUnknown 0-2, ID3D12Object 3-6,
 * ID3D12DeviceChild::GetDevice 7, then each interface's own methods. */
#define DEV_RELEASE 2
#define DEV_OPEN_SHARED_HANDLE 32 /* HRESULT (ID3D12Device*, HANDLE, REFIID, void**) */
#define DEV_GET_ADAPTER_LUID 43   /* LUID *(ID3D12Device*, LUID *out): out pointer, not RAX */
#define FENCE_RELEASE 2
#define FENCE_GET_COMPLETED_VALUE 8     /* UINT64 (ID3D12Fence*) */
#define FENCE_SET_EVENT_ON_COMPLETION 9 /* HRESULT (ID3D12Fence*, UINT64, HANDLE) */
#define FENCE_SIGNAL 10                 /* HRESULT (ID3D12Fence*, UINT64) */
#define FEATURE_LEVEL_11_0 0xb000

/* 189819f1-1db6-4b57-be54-1821339b85f7 */
static const unsigned char IID_ID3D12Device[16] = {0xf1, 0x19, 0x98, 0x18, 0xb6, 0x1d, 0x57, 0x4b,
                                                   0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7};
/* 0a753dcf-c4d8-4b91-adf6-be5a60d95a76 */
static const unsigned char IID_ID3D12Fence[16] = {0xcf, 0x3d, 0x75, 0x0a, 0xd8, 0xc4, 0x91, 0x4b,
                                                  0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76};

typedef pe_hr(PECALL *d3d12_create_device_t)(void *adapter, int level, const void *iid, void **out);
typedef DWORD(PECALL *com_release_t)(pe_com *self);
typedef pe_hr(PECALL *dev_open_shared_t)(pe_com *dev, HANDLE h, const void *iid, void **out);
typedef pe_luid *(PECALL *dev_get_adapter_luid_t)(pe_com *dev, pe_luid *out);
typedef u64(PECALL *fence_completed_t)(pe_com *fence);
typedef pe_hr(PECALL *fence_set_event_t)(pe_com *fence, u64 value, HANDLE event);
typedef pe_hr(PECALL *fence_signal_t)(pe_com *fence, u64 value);

#define COM_CALL(obj, idx, type) ((type)((obj)->vtbl[idx]))

static u64 fence_completed(pe_com *fence) {
    return COM_CALL(fence, FENCE_GET_COMPLETED_VALUE, fence_completed_t)(fence);
}

/* ---- state -------------------------------------------------------------- */

struct pe_fence {
    u32 magic;
    pe_com *fence;             /* ID3D12Fence* from OpenSharedHandle */
    DlssnrFenceShared *shared; /* bridge-owned */
    HANDLE thread, wake, done; /* wake: set by the exports; done: SetEventOnCompletion target */
    HANDLE exited;             /* manual-reset; the worker's last act before returning */
    HANDLE relay, relay_exited; /* the signal relay thread, and its own exit event */
    volatile long lock;       /* guards waits[] and nwaits */
    u64 waits[PE_FENCE_WAITS];
    unsigned nwaits;
    volatile u64 signal_posted; /* highest value asked for; only used to report out-of-order signals */
    volatile int stop;
    u64 armed;     /* worker-private: value `done` is armed for, 0 none */
    u64 signalled; /* relay-private: last value passed to Signal, or completed at import */
    unsigned long trace;      /* rate limit for pe_fence_trace */
    unsigned long errors;     /* rate limit for pe_fence_err */
};

static struct pe_fence *pe_fences[PE_FENCE_SLOTS]; /* registry, under pe_fences_lock */
static volatile long pe_fences_lock;
static volatile int pe_fences_live;
static d3d12_create_device_t pe_create_device; /* resolved once; a race resolves to the same pointer */
static volatile int pe_pinned;

static void pe_spin_lock(volatile long *l) {
    while (__sync_lock_test_and_set(l, 1)) {
    }
}

static void pe_spin_unlock(volatile long *l) {
    __sync_lock_release(l);
}

/* ---- logging ------------------------------------------------------------ */

/* The bridge formats these with three u64 arguments, so a format may only use
 * %llu and %llx. */
static void pe_fence_log(const char *fmt, u64 a, u64 b, u64 c) {
    if (bridge_ok() && g->p_dlssnr_log)
        ((void SYSV (*)(const char *, u64, u64, u64))g->p_dlssnr_log)(fmt, a, b, c);
}

/* First 64 lines, then every 1000th: enough to see a game's calls arrive
 * without writing a line per frame. */
static void pe_fence_trace(struct pe_fence *f, const char *fmt, u64 a, u64 b, u64 c) {
    unsigned long n = __atomic_add_fetch(&f->trace, 1, __ATOMIC_RELAXED);
    if (n <= 64 || n % 1000 == 0)
        pe_fence_log(fmt, a, b, c);
}

/* Errors always log, but a failing call retried every millisecond must not
 * fill the file: the first 16, then every 1000th. */
static void pe_fence_err(struct pe_fence *f, const char *fmt, u64 a, u64 b, u64 c) {
    unsigned long n = __atomic_add_fetch(&f->errors, 1, __ATOMIC_RELAXED);
    if (n <= 16 || n % 1000 == 0)
        pe_fence_log(fmt, a, b, c);
}

static u64 pe_hr_bits(pe_hr hr) {
    return (u64)(u32)hr;
}

/* Values only ever move forward, so a late store cannot un-release a wait. */
static void counter_raise(DlssnrFenceShared *s, u64 v) {
    u64 cur = __atomic_load_n((u64 *)s->counter, __ATOMIC_ACQUIRE);
    while (cur < v && !__atomic_compare_exchange_n((u64 *)s->counter, &cur, v, 0, __ATOMIC_RELEASE,
                                                   __ATOMIC_ACQUIRE))
        ;
}

static u64 counter_load(DlssnrFenceShared *s) {
    return __atomic_load_n((u64 *)s->counter, __ATOMIC_ACQUIRE);
}

/* ---- registry ----------------------------------------------------------- */

static struct pe_fence *pe_fence_lookup(const void *sem) {
    struct pe_fence *hit = 0;
    int i;
    if (!sem)
        return 0;
    pe_spin_lock(&pe_fences_lock);
    for (i = 0; i < PE_FENCE_SLOTS; i++) {
        if (pe_fences[i] == sem) {
            hit = pe_fences[i];
            break;
        }
    }
    pe_spin_unlock(&pe_fences_lock);
    return hit;
}

static int pe_fence_register(struct pe_fence *f) {
    int i, ok = 0;
    pe_spin_lock(&pe_fences_lock);
    for (i = 0; i < PE_FENCE_SLOTS; i++) {
        if (!pe_fences[i]) {
            pe_fences[i] = f;
            __atomic_add_fetch(&pe_fences_live, 1, __ATOMIC_RELEASE);
            ok = 1;
            break;
        }
    }
    pe_spin_unlock(&pe_fences_lock);
    return ok;
}

static void pe_fence_unregister(struct pe_fence *f) {
    int i;
    pe_spin_lock(&pe_fences_lock);
    for (i = 0; i < PE_FENCE_SLOTS; i++) {
        if (pe_fences[i] == f) {
            pe_fences[i] = 0;
            __atomic_sub_fetch(&pe_fences_live, 1, __ATOMIC_RELEASE);
            break;
        }
    }
    pe_spin_unlock(&pe_fences_lock);
}

static int pe_fences_any_live(void) {
    return __atomic_load_n(&pe_fences_live, __ATOMIC_ACQUIRE) != 0;
}

/* ---- worker ------------------------------------------------------------- */

/* Drops waits the fence has passed and returns the smallest that is left, or 0. */
static u64 pe_fence_prune_waits(struct pe_fence *f, u64 completed) {
    unsigned i, keep = 0;
    u64 min = 0;
    pe_spin_lock(&f->lock);
    for (i = 0; i < f->nwaits; i++) {
        if (f->waits[i] <= completed)
            continue;
        f->waits[keep++] = f->waits[i];
        if (!min || f->waits[i] < min)
            min = f->waits[i];
    }
    f->nwaits = keep;
    pe_spin_unlock(&f->lock);
    return min;
}

static DWORD PECALL pe_fence_worker(void *arg) {
    struct pe_fence *f = arg;
    HANDLE handles[2];
    handles[0] = f->wake;
    handles[1] = f->done;
    for (;;) {
        u64 completed = fence_completed(f->fence);
        u64 min;
        int poll;

        counter_raise(f->shared, completed); /* releases every wait <= completed */
        /* Waits are pruned by the fence's value, not the counter: after a drain
         * raised the counter to ~0, entries above the fence stay and are harmless. */
        min = pe_fence_prune_waits(f, completed);
        if (!min) {
            f->armed = 0;
        } else if (min != f->armed) {
            pe_hr hr = COM_CALL(f->fence, FENCE_SET_EVENT_ON_COMPLETION, fence_set_event_t)(
                f->fence, min, f->done); /* re-arm for the new smallest */
            if (hr < 0) {
                pe_fence_err(f, "pe fence: SetEventOnCompletion(%llu) hr=%llx", min,
                             pe_hr_bits(hr), 0);
                f->armed = 0;
            } else {
                f->armed = min;
            }
        }
        if (__atomic_load_n(&f->stop, __ATOMIC_ACQUIRE))
            break;
        /* Poll only while a wait is pending that no event is armed for
         * (SetEventOnCompletion failed); otherwise sleep until an export or the
         * fence wakes us. */
        poll = min && !f->armed;
        WaitForMultipleObjects(2, handles, 0, poll ? PE_FENCE_SLICE_MS : PE_INFINITE);
    }
    /* The fence is released here, not by whoever called destroy. Releasing it
     * joins vkd3d's waiter thread, and that thread cannot finish exiting while
     * the loader lock is held, which is exactly when a DllMain detach destroys
     * the semaphore. On this thread the join only blocks this thread; destroy,
     * which waits on `exited` with a timeout, gets out and drops the lock. The
     * relay goes first because it is the other user of the fence. */
    if (f->relay) {
        HANDLE relay_done = f->relay_exited;
        WaitForMultipleObjects(1, &relay_done, 0, PE_INFINITE);
    }
    /* Releasing drops vkd3d's registrations on `done`, so afterwards nothing
     * will SetEvent it and destroy can close the handle. */
    COM_CALL(f->fence, FENCE_RELEASE, com_release_t)(f->fence);
    SetEvent(f->exited); /* nothing of f is touched after this */
    return 0;
}

/* Passes signal_ready on to the fence. Sleeps in the bridge until a stream
 * callback raises it, or for PE_FENCE_RELAY_SLEEP_MS so `stop` is noticed. */
static DWORD PECALL pe_fence_relay(void *arg) {
    struct pe_fence *f = arg;
    for (;;) {
        /* `stop` is read before `signal_ready`: whatever a stream callback
         * published before destroy set `stop` is then seen by this pass, and the
         * pass after a stop is the last one. Reading it after the Signal would
         * let a value raised in between be dropped. */
        int stopping = __atomic_load_n(&f->stop, __ATOMIC_ACQUIRE);
        u64 ready = __atomic_load_n(&f->shared->signal_ready, __ATOMIC_ACQUIRE);
        int failed = 0;

        if (ready > f->signalled) {
            pe_hr hr = 0;
            /* Never move the fence backwards: Signal sets the value outright, so
             * a stale one would un-complete what the game already waited for.
             * The read and the Signal are not atomic. That is safe for the one
             * consumer this serves, because every game-queue Signal on the
             * fence sits behind a queue Wait on the value signalled here, so it
             * cannot land between the two. A consumer that signals the same
             * fence from the CPU on another thread could race it. */
            if (fence_completed(f->fence) < ready) {
                hr = COM_CALL(f->fence, FENCE_SIGNAL, fence_signal_t)(f->fence, ready);
                if (hr < 0)
                    pe_fence_err(f, "pe fence: Signal(%llu) hr=%llx", ready, pe_hr_bits(hr), 0);
                else
                    pe_fence_trace(f, "pe fence: signalled %llu hr=%llx", ready, pe_hr_bits(hr), 0);
            }
            /* A failed Signal is retried: dropping it would leave the game queue
             * parked on that value forever. */
            if (hr >= 0)
                f->signalled = ready;
            else
                failed = 1;
        }
        if (stopping)
            break; /* after a final pass over whatever the streams delivered */
        if (failed)
            Sleep(PE_FENCE_SLICE_MS);
        else
            ((int SYSV (*)(DlssnrFenceShared *, u64, u32))g->p_dlssnr_fence_await_ready)(
                f->shared, f->signalled, PE_FENCE_RELAY_SLEEP_MS);
    }
    {
        u64 posted = __atomic_load_n(&f->signal_posted, __ATOMIC_ACQUIRE);
        if (posted > f->signalled)
            pe_fence_log("pe fence: signal %llu posted but never delivered (delivered up to %llu)",
                         posted, f->signalled, 0);
    }
    SetEvent(f->relay_exited); /* nothing of f is touched after this */
    return 0;
}

/* ---- import ------------------------------------------------------------- */

static void pe_fence_free_handles(struct pe_fence *f) {
    if (f->thread)
        CloseHandle(f->thread);
    if (f->wake)
        CloseHandle(f->wake);
    if (f->done)
        CloseHandle(f->done);
    if (f->exited)
        CloseHandle(f->exited);
    if (f->relay)
        CloseHandle(f->relay);
    if (f->relay_exited)
        CloseHandle(f->relay_exited);
}

/* Waiting on the exited event as well as the thread matters when destroy runs
 * under the loader lock: a thread cannot finish exiting until it gets that lock
 * for its detach notification, so a join on the thread handle alone would always
 * time out there. The thread handle covers process exit, where Wine has already
 * killed the thread and the event is never set. */
static int pe_fence_join(HANDLE exited, HANDLE thread) {
    HANDLE both[2];
    both[0] = exited;
    both[1] = thread;
    return WaitForMultipleObjects(2, both, 0, PE_FENCE_JOIN_MS) != PE_WAIT_TIMEOUT;
}

/* Asks both threads to finish and waits for them. The worker outlives the relay
 * and releases the fence, so its exit covers both. 0 means one is stuck inside a
 * D3D12 call; the caller then leaks the fence, because freeing memory a running
 * thread still reads is worse. */
static int pe_fence_stop_threads(struct pe_fence *f) {
    __atomic_store_n(&f->stop, 1, __ATOMIC_RELEASE);
    SetEvent(f->wake);
    return pe_fence_join(f->exited, f->thread);
}

static int pe_fence_open(pe_com **fence, void *nt_handle) {
    HANDLE m;
    pe_com *dev = 0;
    pe_hr hr;
    pe_luid luid;

    if (!pe_create_device) {
        m = LoadLibraryA("d3d12.dll"); /* the game has it loaded; this takes a reference */
        pe_create_device = m ? (d3d12_create_device_t)GetProcAddress(m, "D3D12CreateDevice") : 0;
        if (!pe_create_device) {
            pe_fence_log("pe fence: d3d12.dll not loadable", 0, 0, 0);
            return HIP_ERROR_NOT_SUPPORTED;
        }
    }
    /* With no adapter, vkd3d returns the game's own device when DXGI's first
     * adapter is the game's. */
    hr = pe_create_device(0, FEATURE_LEVEL_11_0, IID_ID3D12Device, (void **)&dev);
    if (hr < 0 || !dev) {
        pe_fence_log("pe fence: D3D12CreateDevice hr=%llx", pe_hr_bits(hr), 0, 0);
        return HIP_ERROR_NOT_SUPPORTED;
    }
    luid.low = 0;
    luid.high = 0;
    COM_CALL(dev, DEV_GET_ADAPTER_LUID, dev_get_adapter_luid_t)(dev, &luid);
    pe_fence_log("pe fence: d3d12 device %llx luid=%llx:%llx", (u64)(unsigned long long)dev,
                 (u64)(u32)luid.high, (u64)luid.low);
    *fence = 0;
    hr = COM_CALL(dev, DEV_OPEN_SHARED_HANDLE, dev_open_shared_t)(dev, nt_handle, IID_ID3D12Fence,
                                                                  (void **)fence);
    /* The opened fence holds its own reference to the device. Keeping ours would
     * pin the game's device after the game releases it. */
    COM_CALL(dev, DEV_RELEASE, com_release_t)(dev);
    if (hr < 0 || !*fence) {
        pe_fence_log("pe fence: OpenSharedHandle hr=%llx", pe_hr_bits(hr), 0, 0);
        *fence = 0;
        return HIP_ERROR_INVALID_HANDLE;
    }
    return 0;
}

static int pe_fence_import(void **out, void *nt_handle) {
    pe_com *fence = 0;
    DlssnrFenceShared *shared = 0;
    struct pe_fence *f;
    int err;

    if (!nt_handle) {
        pe_fence_log("pe fence: import without a handle", 0, 0, 0);
        return HIP_ERROR_INVALID_VALUE;
    }
    if (!g->p_dlssnr_fence_create || !g->p_dlssnr_fence_wait || !g->p_dlssnr_fence_signal ||
        !g->p_dlssnr_fence_drain || !g->p_dlssnr_fence_release || !g->p_dlssnr_fence_await_ready)
        return HIP_ERROR_NOT_INITIALIZED;
    if (__atomic_load_n(&pe_fences_live, __ATOMIC_ACQUIRE) >= PE_FENCE_SLOTS)
        return HIP_ERROR_OUT_OF_MEMORY;
    err = pe_fence_open(&fence, nt_handle);
    if (err)
        return err;
    err = ((int SYSV (*)(DlssnrFenceShared **))g->p_dlssnr_fence_create)(&shared);
    if (err || !shared) {
        COM_CALL(fence, FENCE_RELEASE, com_release_t)(fence);
        return err ? err : HIP_ERROR_OUT_OF_MEMORY;
    }
    /* The threads' code must stay mapped until they have returned, and the
     * bridge's HIP state outlives any one import anyway. */
    if (!pe_pinned) {
        HANDLE self = 0;
        if (!GetModuleHandleExA(PE_PIN_MODULE | PE_FROM_ADDRESS, (const char *)pe_fence_worker, &self)) {
            pe_fence_log("pe fence: cannot pin the trampoline", 0, 0, 0);
            err = HIP_ERROR_NOT_SUPPORTED;
            goto fail_shared;
        }
        pe_pinned = 1;
    }
    f = VirtualAlloc(0, sizeof *f, MEM_COMMIT_RESERVE, PAGE_READWRITE);
    if (!f) {
        err = HIP_ERROR_OUT_OF_MEMORY;
        goto fail_shared;
    }
    f->magic = PE_FENCE_MAGIC;
    f->fence = fence;
    f->shared = shared;
    f->wake = CreateEventA(0, 0, 0, 0);
    f->done = CreateEventA(0, 0, 0, 0);
    f->exited = CreateEventA(0, 1, 0, 0);
    f->relay_exited = CreateEventA(0, 1, 0, 0);
    err = HIP_ERROR_OUT_OF_MEMORY;
    if (!f->wake || !f->done || !f->exited || !f->relay_exited)
        goto fail_alloc;
    f->signalled = fence_completed(fence);
    if (!pe_fence_register(f))
        goto fail_alloc;
    f->thread = CreateThread(0, 0, pe_fence_worker, f, 0, 0);
    if (!f->thread) {
        pe_fence_unregister(f);
        goto fail_alloc;
    }
    f->relay = CreateThread(0, 0, pe_fence_relay, f, 0, 0);
    if (!f->relay) {
        pe_fence_unregister(f);
        if (!pe_fence_stop_threads(f)) {
            pe_fence_log("pe fence: worker did not stop in %llu ms; leaking fence %llx", PE_FENCE_JOIN_MS,
                         (u64)(unsigned long long)f, 0);
            return err;
        }
        goto fail_released; /* the worker released the fence on its way out */
    }
    pe_fence_log("pe fence: imported handle=%llx fence=%llx completed=%llu", (u64)(unsigned long long)nt_handle,
                 (u64)(unsigned long long)fence, f->signalled);
    *out = f;
    return 0;

fail_alloc:
    /* The fence goes first: releasing it drops vkd3d's registrations on `done`. */
    COM_CALL(fence, FENCE_RELEASE, com_release_t)(fence);
fail_released:
    pe_fence_free_handles(f);
    VirtualFree(f, 0, MEM_RELEASE);
    ((int SYSV (*)(DlssnrFenceShared *, int))g->p_dlssnr_fence_release)(shared, 1);
    return err;
fail_shared:
    ((int SYSV (*)(DlssnrFenceShared *, int))g->p_dlssnr_fence_release)(shared, 1);
    COM_CALL(fence, FENCE_RELEASE, com_release_t)(fence);
    return err;
}

/* ---- wait and signal ---------------------------------------------------- */

static int pe_fence_wait(struct pe_fence *f, u64 v, void *stream) {
    u64 completed;
    if (!v || counter_load(f->shared) >= v) {
        pe_fence_trace(f, "pe fence: wait %llu already released", v, 0, 0);
        return 0;
    }
    completed = fence_completed(f->fence);
    if (completed >= v) {
        counter_raise(f->shared, completed);
        pe_fence_trace(f, "pe fence: wait %llu already complete (fence at %llu)", v, completed, 0);
        return 0;
    }
    pe_spin_lock(&f->lock);
    if (f->nwaits == PE_FENCE_WAITS) {
        pe_spin_unlock(&f->lock);
        /* The stream wait is not enqueued, so the stream cannot wedge on a
         * value nobody tracks. */
        pe_fence_log("pe fence: wait table full; refusing wait %llu", v, 0, 0);
        return HIP_ERROR_OUT_OF_MEMORY;
    }
    f->waits[f->nwaits++] = v;
    pe_spin_unlock(&f->lock);
    SetEvent(f->wake);
    pe_fence_trace(f, "pe fence: wait %llu registered (fence at %llu)", v, completed, 0);
    /* Registered before the stream wait goes in, so the worker cannot miss it; a
     * failed enqueue leaves an entry the worker drops once the fence passes it. */
    return ((int SYSV (*)(DlssnrFenceShared *, void *, u64))g->p_dlssnr_fence_wait)(f->shared, stream, v);
}

static int pe_fence_signal(struct pe_fence *f, u64 v, void *stream) {
    u64 posted = __atomic_load_n(&f->signal_posted, __ATOMIC_ACQUIRE);
    int err;
    while (posted < v &&
           !__atomic_compare_exchange_n(&f->signal_posted, &posted, v, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        ;
    if (posted >= v)
        /* Still enqueued: the callback's maximum makes it a no-op, and failing
         * here would make the consumer throw on a frame. */
        pe_fence_err(f, "pe fence: signal %llu not above %llu; the fence only moves forward", v, posted, 0);
    err = ((int SYSV (*)(DlssnrFenceShared *, void *, u64))g->p_dlssnr_fence_signal)(f->shared, stream, v);
    if (err) {
        pe_fence_err(f, "pe fence: bridge signal %llu failed err=%llu", v, (u64)(u32)err, 0);
        return err;
    }
    pe_fence_trace(f, "pe fence: signal %llu posted", v, 0, 0);
    return 0;
}

/* ---- destroy ------------------------------------------------------------ */

static int pe_fence_destroy(struct pe_fence *f) {
    int drained;

    /* Later calls with this handle go to HIP and fail there, as they would for
     * a HIP handle that was already destroyed. */
    pe_fence_unregister(f);
    /* The threads keep running through the drain: a signal whose callback fires
     * now still reaches the fence, and the game queue may be waiting on it. */
    drained = ((int SYSV (*)(DlssnrFenceShared *))g->p_dlssnr_fence_drain)(f->shared) == 0;
    if (!pe_fence_stop_threads(f)) {
        pe_fence_log("pe fence: worker did not stop in %llu ms; leaking fence %llx", PE_FENCE_JOIN_MS,
                     (u64)(unsigned long long)f, 0);
        return 0;
    }
    /* The worker has released the fence before it set `exited`, so no vkd3d
     * thread will SetEvent `done` any more and the handles can go. */
    pe_fence_free_handles(f);
    ((int SYSV (*)(DlssnrFenceShared *, int))g->p_dlssnr_fence_release)(f->shared, drained);
    pe_fence_log("pe fence: destroyed fence=%llx drained=%llu", (u64)(unsigned long long)f, (u64)drained, 0);
    f->magic = 0;
    VirtualFree(f, 0, MEM_RELEASE);
    return 0;
}
