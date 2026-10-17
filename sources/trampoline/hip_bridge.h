/* Shared layout between the Linux HIP bridge .so and the PE amdhip64_7.dll trampoline. */
#ifndef DLSSNR_HIP_BRIDGE_H
#define DLSSNR_HIP_BRIDGE_H

#define DLSSNR_HIP_MAGIC 0x364849504E524C44ULL /* "DLNRHIP6"; incompatible tables are rejected */
#define DLSSNR_HIP_ENV "DLSSNR_HIP_BRIDGE"
#define DLSSNR_HIP_ADDR_FILE "/tmp/dlssnr_hip_bridge.addr"

typedef unsigned int u32;
typedef unsigned long long u64;

typedef struct {
    u32 x, y, z;
} dim3_t;

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
} DlssnrHipBridge;

#endif
