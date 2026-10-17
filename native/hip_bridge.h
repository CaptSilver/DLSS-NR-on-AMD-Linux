/* Shared layout between the Linux HIP bridge .so and the PE amdhip64_7.dll trampoline. */
#ifndef DLSSNR_HIP_BRIDGE_H
#define DLSSNR_HIP_BRIDGE_H

#define DLSSNR_HIP_MAGIC 0x384849504E524C44ULL /* "DLNRHIP8"; incompatible tables are rejected */
#define DLSSNR_HIP_ENV "DLSSNR_HIP_BRIDGE"
#define DLSSNR_HIP_ADDR_FILE "/tmp/dlssnr_hip_bridge.addr"

typedef unsigned int u32;
typedef unsigned long long u64;

typedef struct {
    u32 x, y, z;
} dim3_t;

#define DLSSNR_FENCE_SHARED_MAGIC 0x44485346u /* "FSHD" */

/* Bridge-owned half of an imported D3D12 fence. The PE worker reads counter and
 * signal_ready; ROCm's host-callback thread writes signal_ready; both use atomics
 * only. The bridge frees it when the import and every queued callback are done. */
typedef struct {
    u32 magic;
    u32 refs;                  /* bridge-private */
    volatile u64 *counter;     /* host-mapped: a stream blocks until *counter >= value */
    void *counter_dev;         /* the same page as the GPU sees it */
    volatile u64 signal_ready; /* highest value whose stream work has finished */
} DlssnrFenceShared;

typedef struct {
    u64 magic;
    /* compiler / fatbin */
    void *(*p___hipRegisterFatBinary)(void *data);
    void (*p___hipUnregisterFatBinary)(void **modules);
    void (*p___hipRegisterFunction)(void **modules, const void *hostFunction, char *deviceFunction,
                                    const char *deviceName, unsigned int threadLimit, void *tid,
                                    void *bid, void *blockDim, void *gridDim, int *wSize);
    void (*p___hipRegisterVar)(void **modules, void *var, char *hostVar, char *deviceVar, int ext,
                               u64 size, int constant, int global);
    int (*p___hipPushCallConfiguration)(dim3_t gridDim, dim3_t blockDim, u64 sharedMem,
                                        void *stream);
    int (*p___hipPopCallConfiguration)(dim3_t *gridDim, dim3_t *blockDim, u64 *sharedMem,
                                       void **stream);
    /* runtime */
    int (*p_hipGetDeviceCount)(int *count);
    int (*p_hipSetDevice)(int device);
    int (*p_hipGetDevicePropertiesR0600)(void *prop, int deviceId);
    int (*p_hipDriverGetVersion)(int *driverVersion);
    int (*p_hipRuntimeGetVersion)(int *runtimeVersion);
    int (*p_hipMalloc)(void **ptr, u64 size);
    int (*p_hipFree)(void *ptr);
    int (*p_hipMemcpy)(void *dst, const void *src, u64 sizeBytes, int kind);
    int (*p_hipMemcpyAsync)(void *dst, const void *src, u64 sizeBytes, int kind,
                            void *stream);
    int (*p_hipMemcpyToSymbol)(const void *symbol, const void *src, u64 sizeBytes,
                               u64 offset, int kind);
    int (*p_hipMemset)(void *dst, int value, u64 sizeBytes);
    int (*p_hipMemsetAsync)(void *dst, int value, u64 sizeBytes, void *stream);
    int (*p_hipLaunchKernel)(const void *function_address, dim3_t numBlocks, dim3_t dimBlocks,
                             void **args, u64 sharedMemBytes, void *stream);
    int (*p_hipDeviceSynchronize)(void);
    int (*p_hipGetLastError)(void);
    const char *(*p_hipGetErrorString)(int error);
    int (*p_hipEventCreate)(void **event);
    int (*p_hipEventCreateWithFlags)(void **event, unsigned int flags);
    int (*p_hipEventQuery)(void *event);
    int (*p_hipStreamCreateWithFlags)(void **stream, unsigned int flags);
    int (*p_hipStreamSynchronize)(void *stream);
    int (*p_hipEventRecord)(void *event, void *stream);
    int (*p_hipEventSynchronize)(void *event);
    int (*p_hipEventElapsedTime)(float *ms, void *start, void *stop);
    int (*p_hipImportExternalMemory)(void **extMem_out, const void *memHandleDesc);
    int (*p_hipExternalMemoryGetMappedBuffer)(void **devPtr, void *extMem, const void *bufferDesc);
    int (*p_hipDestroyExternalMemory)(void *extMem);
    /* ABI 3: upstream 0.3.3/0.4.0 queries, forwarding original host addresses. */
    int (*p_hipGetDevice)(int *device);
    int (*p_hipOccupancyMaxActiveBlocksPerMultiprocessor)(int *blocks,
            const void *function_address, int block_size, u64 dynamic_shared_bytes);
    /* ABI 4: upstream 0.4.1 priority-stream lifecycle. */
    int (*p_hipDeviceGetStreamPriorityRange)(int *least, int *greatest);
    int (*p_hipStreamCreateWithPriority)(void **stream, unsigned int flags, int priority);
    int (*p_hipStreamDestroy)(void *stream);
    /* ABI 5: the pinned host-memory calls upstream 0.6.0 imports, forwarded
     * unchanged to HIP. */
    int (*p_hipHostMalloc)(void **ptr, u64 size, unsigned int flags);
    int (*p_hipHostGetDevicePointer)(void **devPtr, void *hstPtr, unsigned int flags);
    /* ABI 6: module-API runtimes and the calls around them. hipModuleLoad has
     * no slot: the PE side reads the file and uses hipModuleLoadData. Pointers
     * (kernelParams, extra, code image, error names) cross unchanged because
     * both sides share one address space. */
    int (*p_hipInit)(unsigned int flags);
    int (*p_hipMemGetInfo)(u64 *freeBytes, u64 *totalBytes);
    int (*p_hipEventDestroy)(void *event);
    const char *(*p_hipGetErrorName)(int error);
    int (*p_hipStreamCreate)(void **stream);
    int (*p_hipHostFree)(void *ptr);
    int (*p_hipModuleLoadData)(void **module, const void *image);
    int (*p_hipModuleGetFunction)(void **function, void *module, const char *name);
    int (*p_hipModuleLaunchKernel)(void *function, unsigned int gridX, unsigned int gridY,
                                   unsigned int gridZ, unsigned int blockX, unsigned int blockY,
                                   unsigned int blockZ, unsigned int sharedMemBytes, void *stream,
                                   void **kernelParams, void **extra);
    int (*p_hipModuleUnload)(void *module);
    int (*p_hipStreamQuery)(void *stream);
    int (*p_hipMemcpy2DAsync)(void *dst, u64 dpitch, const void *src, u64 spitch, u64 width,
                              u64 height, int kind, void *stream);
    int (*p_hipExtModuleLaunchKernel)(void *function, unsigned int globalX, unsigned int globalY,
                                      unsigned int globalZ, unsigned int localX, unsigned int localY,
                                      unsigned int localZ, u64 sharedMemBytes, void *stream,
                                      void **kernelParams, void **extra, void *startEvent,
                                      void *stopEvent, unsigned int flags);
    int (*p_hipDeviceGetName)(char *name, int len, int device);
    /* ABI 7: external semaphores. Every handle type goes to HIP through these
     * four slots unchanged; D3D12 fences (type 4) never reach them, see ABI 8.
     * Arrays and descriptors are in-process pointers and cross unchanged. */
    int (*p_hipImportExternalSemaphore)(void **extSem_out, const void *semHandleDesc);
    int (*p_hipDestroyExternalSemaphore)(void *extSem);
    int (*p_hipWaitExternalSemaphoresAsync)(const void *extSemArray, const void *paramsArray,
                                            unsigned int numExtSems, void *stream);
    int (*p_hipSignalExternalSemaphoresAsync)(const void *extSemArray, const void *paramsArray,
                                              unsigned int numExtSems, void *stream);
    /* ABI 8: D3D12 fences (handle type 4) are opened on the PE side, which owns
     * the ID3D12Fence and a worker thread. The bridge keeps the half that needs
     * HIP: the host-mapped counter a stream blocks on and the host callback
     * that reports finished stream work. */
    int (*p_dlssnr_fence_create)(DlssnrFenceShared **out);
    int (*p_dlssnr_fence_wait)(DlssnrFenceShared *f, void *stream, u64 value);
    int (*p_dlssnr_fence_signal)(DlssnrFenceShared *f, void *stream, u64 value);
    int (*p_dlssnr_fence_drain)(DlssnrFenceShared *f);          /* 0 idle, 600 still busy */
    int (*p_dlssnr_fence_release)(DlssnrFenceShared *f, int drained);
    /* PE-side lines for hip.log. fmt uses only %llu / %llx and consumes a, b, c. */
    void (*p_dlssnr_log)(const char *fmt, u64 a, u64 b, u64 c);
    /* Sleeps until signal_ready exceeds `after` or timeout_ms passes: 0 when it
     * does, 600 on timeout. The PE side's signal relay thread lives in this
     * call; the ROCm callback that raises signal_ready wakes it. */
    int (*p_dlssnr_fence_await_ready)(DlssnrFenceShared *f, u64 after, u32 timeout_ms);
} DlssnrHipBridge;

#endif
