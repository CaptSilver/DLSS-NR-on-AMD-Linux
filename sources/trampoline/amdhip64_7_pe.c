/* PE trampoline: Win64 exports → Linux HIP via DlssnrHipBridge (same process).
 * Compiled with clang -target x86_64-pc-windows-msvc (or gnu). */
#include "hip_bridge.h"

#define EXPORT __declspec(dllexport)
#define SYSV __attribute__((sysv_abi))

typedef int BOOL;
typedef unsigned int DWORD;
typedef void *HANDLE;
#define DLL_PROCESS_ATTACH 1
#define GENERIC_READ 0x80000000u
#define GENERIC_WRITE 0x40000000u
#define FILE_SHARE_READ 1u
#define OPEN_EXISTING 3
#define MEM_COMMIT_RESERVE 0x3000u
#define MEM_RELEASE 0x8000u
#define PAGE_READWRITE 4u
#define CREATE_ALWAYS 2
#define FILE_ATTRIBUTE_NORMAL 0x80u
#define INVALID_HANDLE_VALUE ((HANDLE)(long long)-1)

__declspec(dllimport) DWORD __stdcall GetEnvironmentVariableA(const char *name, char *buf, DWORD size);
__declspec(dllimport) int __stdcall __wine_get_unix_env(const char *var, char *buf, unsigned long long size);
__declspec(dllimport) HANDLE __stdcall CreateFileA(const char *name, DWORD acc, DWORD share, void *sa,
                                                   DWORD disp, DWORD attr, HANDLE tmpl);
__declspec(dllimport) BOOL __stdcall WriteFile(HANDLE h, const void *buf, DWORD n, DWORD *w, void *ov);
__declspec(dllimport) BOOL __stdcall CloseHandle(HANDLE h);
__declspec(dllimport) BOOL __stdcall ReadFile(HANDLE h, void *buf, DWORD n, DWORD *got, void *ov);
__declspec(dllimport) DWORD __stdcall GetFileSize(HANDLE h, DWORD *high);
__declspec(dllimport) void *__stdcall VirtualAlloc(void *addr, unsigned long long size, DWORD type,
                                                   DWORD protect);
__declspec(dllimport) BOOL __stdcall VirtualFree(void *addr, unsigned long long size, DWORD type);

static DlssnrHipBridge *g;

static unsigned long long parse_hex_ptr(const char *s) {
    unsigned long long v = 0;
    if (!s)
        return 0;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
        s += 2;
    while (*s) {
        unsigned char c = (unsigned char)*s++;
        v <<= 4;
        if (c >= '0' && c <= '9')
            v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v |= (unsigned)(c - 'A' + 10);
        else
            break;
    }
    return v;
}

static int bridge_ok(void) {
    return g && g->magic == DLSSNR_HIP_MAGIC;
}

static void pe_log(const char *s, unsigned n) {
    HANDLE h = CreateFileA("Z:\\tmp\\dlssnr-pe.log", GENERIC_WRITE, 0, 0, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, 0);
    if (h == INVALID_HANDLE_VALUE)
        return;
    DWORD w = 0;
    WriteFile(h, s, n, &w, 0);
    CloseHandle(h);
}

static void init_bridge(void) {
    char buf[64];
    char msg[160];
    unsigned i;
    buf[0] = 0;
    /* Live Unix environ (LD_PRELOAD setenv). Wine PE env is a stale snapshot. */
    if (__wine_get_unix_env(DLSSNR_HIP_ENV, buf, sizeof(buf)) != 0 || !buf[0]) {
        DWORD n = GetEnvironmentVariableA(DLSSNR_HIP_ENV, buf, (DWORD)sizeof(buf));
        if (n == 0 || n >= sizeof(buf))
            buf[0] = 0;
    }
    if (buf[0])
        g = (DlssnrHipBridge *)(unsigned long long)parse_hex_ptr(buf);
    /* tiny hex dump into pe log */
    msg[0] = 'e'; msg[1] = 'n'; msg[2] = 'v'; msg[3] = '=';
    for (i = 0; i < 48 && buf[i]; i++)
        msg[4 + i] = buf[i];
    msg[4 + i] = ' ';
    msg[5 + i] = 'o';
    msg[6 + i] = 'k';
    msg[7 + i] = '=';
    msg[8 + i] = (char)('0' + (bridge_ok() ? 1 : 0));
    msg[9 + i] = '\n';
    pe_log(msg, 10 + i);
}

BOOL __stdcall DllMain(void *h, DWORD reason, void *r) {
    (void)h;
    (void)r;
    if (reason == DLL_PROCESS_ATTACH)
        init_bridge();
    return 1;
}

#define HIP_ERROR_NOT_INITIALIZED 3
#define HIP_ERROR_INVALID_VALUE 1
#define HIP_SUCCESS 0
#define HIP_ERROR_OUT_OF_MEMORY 2
#define HIP_ERROR_FILE_NOT_FOUND 301

EXPORT void *__hipRegisterFatBinary(void *data) {
    if (!bridge_ok() || !g->p___hipRegisterFatBinary)
        return 0;
    return ((void *SYSV (*)(void *))g->p___hipRegisterFatBinary)(data);
}

EXPORT void __hipUnregisterFatBinary(void **modules) {
    if (!bridge_ok() || !g->p___hipUnregisterFatBinary)
        return;
    ((void SYSV (*)(void **))g->p___hipUnregisterFatBinary)(modules);
}

EXPORT void __hipRegisterFunction(void **modules, const void *hostFunction, char *deviceFunction,
                                  const char *deviceName, unsigned int threadLimit, void *tid,
                                  void *bid, void *blockDim, void *gridDim, int *wSize) {
    if (!bridge_ok() || !g->p___hipRegisterFunction)
        return;
    ((void SYSV (*)(void **, const void *, char *, const char *, unsigned int, void *, void *, void *,
                    void *, int *))g->p___hipRegisterFunction)(
        modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim,
        wSize);
}

EXPORT void __hipRegisterVar(void **modules, void *var, char *hostVar, char *deviceVar, int ext,
                             u64 size, int constant, int global) {
    if (!bridge_ok() || !g->p___hipRegisterVar)
        return;
    ((void SYSV (*)(void **, void *, char *, char *, int, u64, int, int))g->p___hipRegisterVar)(
        modules, var, hostVar, deviceVar, ext, size, constant, global);
}

EXPORT int __hipPushCallConfiguration(dim3_t gridDim, dim3_t blockDim, u64 sharedMem,
                                      void *stream) {
    if (!bridge_ok() || !g->p___hipPushCallConfiguration)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(dim3_t, dim3_t, u64, void *))g->p___hipPushCallConfiguration)(
        gridDim, blockDim, sharedMem, stream);
}

EXPORT int __hipPopCallConfiguration(dim3_t *gridDim, dim3_t *blockDim, u64 *sharedMem,
                                     void **stream) {
    if (!bridge_ok() || !g->p___hipPopCallConfiguration)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(dim3_t *, dim3_t *, u64 *, void **))g->p___hipPopCallConfiguration)(
        gridDim, blockDim, sharedMem, stream);
}

EXPORT int hipGetDeviceCount(int *count) {
    if (!bridge_ok() || !g->p_hipGetDeviceCount)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(int *))g->p_hipGetDeviceCount)(count);
}

EXPORT int hipGetDevice(int *device) {
    if (!bridge_ok() || !g->p_hipGetDevice) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(int *))g->p_hipGetDevice)(device);
}

EXPORT int hipOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks,
        const void *function_address, int block_size, u64 dynamic_shared_bytes) {
    if (!bridge_ok() || !g->p_hipOccupancyMaxActiveBlocksPerMultiprocessor)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(int *, const void *, int, u64))
        g->p_hipOccupancyMaxActiveBlocksPerMultiprocessor)(
        blocks, function_address, block_size, dynamic_shared_bytes);
}

EXPORT int hipSetDevice(int device) {
    if (!bridge_ok() || !g->p_hipSetDevice)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(int))g->p_hipSetDevice)(device);
}

EXPORT int hipGetDevicePropertiesR0600(void *prop, int deviceId) {
    if (!bridge_ok() || !g->p_hipGetDevicePropertiesR0600)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, int))g->p_hipGetDevicePropertiesR0600)(prop, deviceId);
}

EXPORT int hipDriverGetVersion(int *driverVersion) {
    if (!bridge_ok() || !g->p_hipDriverGetVersion)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(int *))g->p_hipDriverGetVersion)(driverVersion);
}

EXPORT int hipRuntimeGetVersion(int *runtimeVersion) {
    if (!bridge_ok() || !g->p_hipRuntimeGetVersion)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(int *))g->p_hipRuntimeGetVersion)(runtimeVersion);
}

EXPORT int hipMalloc(void **ptr, u64 size) {
    if (!bridge_ok() || !g->p_hipMalloc)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, u64))g->p_hipMalloc)(ptr, size);
}

EXPORT int hipFree(void *ptr) {
    if (!bridge_ok() || !g->p_hipFree)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipFree)(ptr);
}

EXPORT int hipMemcpy(void *dst, const void *src, u64 sizeBytes, int kind) {
    if (!bridge_ok() || !g->p_hipMemcpy)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, const void *, u64, int))g->p_hipMemcpy)(dst, src, sizeBytes,
                                                                                    kind);
}

EXPORT int hipMemcpyAsync(void *dst, const void *src, u64 sizeBytes, int kind, void *stream) {
    if (!bridge_ok() || !g->p_hipMemcpyAsync)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, const void *, u64, int, void *))g->p_hipMemcpyAsync)(
        dst, src, sizeBytes, kind, stream);
}

EXPORT int hipMemcpyToSymbol(const void *symbol, const void *src, u64 sizeBytes,
                             u64 offset, int kind) {
    if (!bridge_ok() || !g->p_hipMemcpyToSymbol)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(const void *, const void *, u64, u64, int))
                g->p_hipMemcpyToSymbol)(symbol, src, sizeBytes, offset, kind);
}

EXPORT int hipMemset(void *dst, int value, u64 sizeBytes) {
    if (!bridge_ok() || !g->p_hipMemset)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, int, u64))g->p_hipMemset)(dst, value, sizeBytes);
}

EXPORT int hipMemsetAsync(void *dst, int value, u64 sizeBytes, void *stream) {
    if (!bridge_ok() || !g->p_hipMemsetAsync)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, int, u64, void *))g->p_hipMemsetAsync)(dst, value, sizeBytes,
                                                                                   stream);
}

EXPORT int hipLaunchKernel(const void *function_address, dim3_t numBlocks, dim3_t dimBlocks, void **args,
                           u64 sharedMemBytes, void *stream) {
    if (!bridge_ok() || !g->p_hipLaunchKernel)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(const void *, dim3_t, dim3_t, void **, u64, void *))
                g->p_hipLaunchKernel)(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
                                      stream);
}

EXPORT int hipDeviceSynchronize(void) {
    if (!bridge_ok() || !g->p_hipDeviceSynchronize)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void))g->p_hipDeviceSynchronize)();
}

EXPORT int hipGetLastError(void) {
    if (!bridge_ok() || !g->p_hipGetLastError)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void))g->p_hipGetLastError)();
}

EXPORT const char *hipGetErrorString(int error) {
    if (!bridge_ok() || !g->p_hipGetErrorString)
        return "hip bridge not initialized";
    return ((const char *SYSV (*)(int))g->p_hipGetErrorString)(error);
}

EXPORT int hipEventCreate(void **event) {
    if (!bridge_ok() || !g->p_hipEventCreate)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **))g->p_hipEventCreate)(event);
}

EXPORT int hipEventCreateWithFlags(void **event, unsigned int flags) {
    if (!bridge_ok() || !g->p_hipEventCreateWithFlags) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, unsigned int))g->p_hipEventCreateWithFlags)(event, flags);
}

EXPORT int hipEventQuery(void *event) {
    if (!bridge_ok() || !g->p_hipEventQuery) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipEventQuery)(event);
}

EXPORT int hipStreamCreateWithFlags(void **stream, unsigned int flags) {
    if (!bridge_ok() || !g->p_hipStreamCreateWithFlags) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, unsigned int))g->p_hipStreamCreateWithFlags)(stream, flags);
}

EXPORT int hipDeviceGetStreamPriorityRange(int *least, int *greatest) {
    if (!bridge_ok() || !g->p_hipDeviceGetStreamPriorityRange) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(int *, int *))g->p_hipDeviceGetStreamPriorityRange)(least, greatest);
}

EXPORT int hipStreamCreateWithPriority(void **stream, unsigned int flags, int priority) {
    if (!bridge_ok() || !g->p_hipStreamCreateWithPriority) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, unsigned int, int))g->p_hipStreamCreateWithPriority)(stream, flags, priority);
}

EXPORT int hipStreamDestroy(void *stream) {
    if (!bridge_ok() || !g->p_hipStreamDestroy) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipStreamDestroy)(stream);
}

EXPORT int hipHostMalloc(void **ptr, u64 size, unsigned int flags) {
    if (!bridge_ok() || !g->p_hipHostMalloc) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, u64, unsigned int))g->p_hipHostMalloc)(ptr, size, flags);
}

EXPORT int hipHostGetDevicePointer(void **devPtr, void *hstPtr, unsigned int flags) {
    if (!bridge_ok() || !g->p_hipHostGetDevicePointer) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, void *, unsigned int))g->p_hipHostGetDevicePointer)(devPtr, hstPtr, flags);
}

EXPORT int hipStreamSynchronize(void *stream) {
    if (!bridge_ok() || !g->p_hipStreamSynchronize) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipStreamSynchronize)(stream);
}

EXPORT int hipEventRecord(void *event, void *stream) {
    if (!bridge_ok() || !g->p_hipEventRecord)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, void *))g->p_hipEventRecord)(event, stream);
}

EXPORT int hipEventSynchronize(void *event) {
    if (!bridge_ok() || !g->p_hipEventSynchronize)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipEventSynchronize)(event);
}

EXPORT int hipEventElapsedTime(float *ms, void *start, void *stop) {
    if (!bridge_ok() || !g->p_hipEventElapsedTime)
        return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(float *, void *, void *))g->p_hipEventElapsedTime)(ms, start, stop);
}

EXPORT int hipImportExternalMemory(void **extMem_out, const void *memHandleDesc) {
    /* Windows D3D12/NT handles cannot be imported by Linux HIP. Let the mod fall back to readback. */
    if (!bridge_ok() || !g->p_hipImportExternalMemory)
        return HIP_ERROR_INVALID_VALUE;
    return ((int SYSV (*)(void **, const void *))g->p_hipImportExternalMemory)(extMem_out, memHandleDesc);
}

EXPORT int hipExternalMemoryGetMappedBuffer(void **devPtr, void *extMem, const void *bufferDesc) {
    if (!bridge_ok() || !g->p_hipExternalMemoryGetMappedBuffer)
        return HIP_ERROR_INVALID_VALUE;
    return ((int SYSV (*)(void **, void *, const void *))g->p_hipExternalMemoryGetMappedBuffer)(
        devPtr, extMem, bufferDesc);
}

EXPORT int hipDestroyExternalMemory(void *extMem) {
    if (!bridge_ok() || !g->p_hipDestroyExternalMemory)
        return HIP_ERROR_INVALID_VALUE;
    return ((int SYSV (*)(void *))g->p_hipDestroyExternalMemory)(extMem);
}

EXPORT int hipInit(unsigned int flags) {
    if (!bridge_ok() || !g->p_hipInit) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(unsigned int))g->p_hipInit)(flags);
}

EXPORT int hipMemGetInfo(u64 *freeBytes, u64 *totalBytes) {
    if (!bridge_ok() || !g->p_hipMemGetInfo) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(u64 *, u64 *))g->p_hipMemGetInfo)(freeBytes, totalBytes);
}

EXPORT int hipEventDestroy(void *event) {
    if (!bridge_ok() || !g->p_hipEventDestroy) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipEventDestroy)(event);
}

EXPORT const char *hipGetErrorName(int error) {
    if (!bridge_ok() || !g->p_hipGetErrorName)
        return "hip bridge not initialized";
    return ((const char *SYSV (*)(int))g->p_hipGetErrorName)(error);
}

EXPORT int hipStreamCreate(void **stream) {
    if (!bridge_ok() || !g->p_hipStreamCreate) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **))g->p_hipStreamCreate)(stream);
}

EXPORT int hipHostFree(void *ptr) {
    if (!bridge_ok() || !g->p_hipHostFree) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipHostFree)(ptr);
}

EXPORT int hipStreamQuery(void *stream) {
    if (!bridge_ok() || !g->p_hipStreamQuery) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipStreamQuery)(stream);
}

EXPORT int hipMemcpy2DAsync(void *dst, u64 dpitch, const void *src, u64 spitch, u64 width,
                            u64 height, int kind, void *stream) {
    if (!bridge_ok() || !g->p_hipMemcpy2DAsync) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, u64, const void *, u64, u64, u64, int, void *))
                g->p_hipMemcpy2DAsync)(dst, dpitch, src, spitch, width, height, kind, stream);
}

EXPORT int hipDeviceGetName(char *name, int len, int device) {
    if (!bridge_ok() || !g->p_hipDeviceGetName) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(char *, int, int))g->p_hipDeviceGetName)(name, len, device);
}

/* Module API. The code object is an in-process pointer, and kernelParams /
 * extra are arrays of pointers into the caller's memory; both sides share the
 * address space, so they cross unchanged. */
EXPORT int hipModuleLoadData(void **module, const void *image) {
    if (!bridge_ok() || !g->p_hipModuleLoadData) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, const void *))g->p_hipModuleLoadData)(module, image);
}

/* Code objects read by hipModuleLoad. The runtime may keep reading the image
 * after the load call (lazy code-object parsing), so each buffer lives until
 * hipModuleUnload of its module. */
#define MODULE_IMAGE_SLOTS 64
static struct { void *module; void *image; } module_images[MODULE_IMAGE_SLOTS];
static volatile long module_images_lock;

static void module_images_acquire(void) {
    while (__sync_lock_test_and_set(&module_images_lock, 1)) {}
}

static void module_images_release(void) {
    __sync_lock_release(&module_images_lock);
}

/* Claim a slot before anything is read, so a full table cannot leak a buffer. */
static int module_image_claim(void) {
    int i, slot = -1;
    module_images_acquire();
    for (i = 0; i < MODULE_IMAGE_SLOTS; i++) {
        if (!module_images[i].module && !module_images[i].image) {
            module_images[i].image = (void *)1; /* reserved */
            slot = i;
            break;
        }
    }
    module_images_release();
    return slot;
}

/* The path is a Windows path the Linux runtime cannot open, so read the code
 * object here and hand the bytes to hipModuleLoadData. */
EXPORT int hipModuleLoad(void **module, const char *fname) {
    HANDLE file;
    DWORD size, got = 0;
    void *image;
    int result, slot;
    if (!bridge_ok() || !g->p_hipModuleLoadData) return HIP_ERROR_NOT_INITIALIZED;
    if (!module || !fname) return HIP_ERROR_INVALID_VALUE;
    slot = module_image_claim();
    if (slot < 0) return HIP_ERROR_OUT_OF_MEMORY;
    file = CreateFileA(fname, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL, 0);
    result = HIP_ERROR_FILE_NOT_FOUND;
    image = 0;
    if (file != INVALID_HANDLE_VALUE) {
        size = GetFileSize(file, 0);
        result = HIP_ERROR_INVALID_VALUE;
        if (size && size != 0xFFFFFFFFu) {
            image = VirtualAlloc(0, size, MEM_COMMIT_RESERVE, PAGE_READWRITE);
            result = HIP_ERROR_OUT_OF_MEMORY;
            if (image) {
                result = HIP_ERROR_INVALID_VALUE;
                if (ReadFile(file, image, size, &got, 0) && got == size)
                    result = ((int SYSV (*)(void **, const void *))g->p_hipModuleLoadData)(module,
                                                                                         image);
            }
        }
        CloseHandle(file);
    }
    module_images_acquire();
    if (result == HIP_SUCCESS) {
        module_images[slot].module = *module;
        module_images[slot].image = image;
    } else {
        module_images[slot].image = 0;
    }
    module_images_release();
    if (result != HIP_SUCCESS && image)
        VirtualFree(image, 0, MEM_RELEASE);
    return result;
}

EXPORT int hipModuleGetFunction(void **function, void *module, const char *name) {
    if (!bridge_ok() || !g->p_hipModuleGetFunction) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, void *, const char *))g->p_hipModuleGetFunction)(function, module,
                                                                                  name);
}

EXPORT int hipModuleLaunchKernel(void *function, unsigned int gridX, unsigned int gridY,
                                 unsigned int gridZ, unsigned int blockX, unsigned int blockY,
                                 unsigned int blockZ, unsigned int sharedMemBytes, void *stream,
                                 void **kernelParams, void **extra) {
    if (!bridge_ok() || !g->p_hipModuleLaunchKernel) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, unsigned int, unsigned int, unsigned int, unsigned int,
                          unsigned int, unsigned int, unsigned int, void *, void **, void **))
                g->p_hipModuleLaunchKernel)(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                            sharedMemBytes, stream, kernelParams, extra);
}

EXPORT int hipExtModuleLaunchKernel(void *function, unsigned int globalX, unsigned int globalY,
                                    unsigned int globalZ, unsigned int localX, unsigned int localY,
                                    unsigned int localZ, u64 sharedMemBytes, void *stream,
                                    void **kernelParams, void **extra, void *startEvent,
                                    void *stopEvent, unsigned int flags) {
    if (!bridge_ok() || !g->p_hipExtModuleLaunchKernel) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *, unsigned int, unsigned int, unsigned int, unsigned int,
                          unsigned int, unsigned int, u64, void *, void **, void **, void *, void *,
                          unsigned int))g->p_hipExtModuleLaunchKernel)(
        function, globalX, globalY, globalZ, localX, localY, localZ, sharedMemBytes, stream,
        kernelParams, extra, startEvent, stopEvent, flags);
}

EXPORT int hipModuleUnload(void *module) {
    int result, i;
    void *image = 0;
    if (!bridge_ok() || !g->p_hipModuleUnload) return HIP_ERROR_NOT_INITIALIZED;
    result = ((int SYSV (*)(void *))g->p_hipModuleUnload)(module);
    module_images_acquire();
    for (i = 0; i < MODULE_IMAGE_SLOTS; i++) {
        if (module && module_images[i].module == module) {
            image = module_images[i].image;
            module_images[i].module = 0;
            module_images[i].image = 0;
            break;
        }
    }
    module_images_release();
    if (image)
        VirtualFree(image, 0, MEM_RELEASE);
    return result;
}

/* External semaphores. The bridge turns D3D12 fence imports (type 4) into DRM
 * timeline syncobjs; the descriptor, handle arrays and parameter blocks are
 * in-process memory and cross unchanged. */
EXPORT int hipImportExternalSemaphore(void **extSem, const void *semHandleDesc) {
    if (!bridge_ok() || !g->p_hipImportExternalSemaphore) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void **, const void *))g->p_hipImportExternalSemaphore)(extSem,
                                                                                semHandleDesc);
}

EXPORT int hipDestroyExternalSemaphore(void *extSem) {
    if (!bridge_ok() || !g->p_hipDestroyExternalSemaphore) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(void *))g->p_hipDestroyExternalSemaphore)(extSem);
}

EXPORT int hipWaitExternalSemaphoresAsync(const void *extSemArray, const void *paramsArray,
                                          unsigned int numExtSems, void *stream) {
    if (!bridge_ok() || !g->p_hipWaitExternalSemaphoresAsync) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(const void *, const void *, unsigned int, void *))
                g->p_hipWaitExternalSemaphoresAsync)(extSemArray, paramsArray, numExtSems, stream);
}

EXPORT int hipSignalExternalSemaphoresAsync(const void *extSemArray, const void *paramsArray,
                                            unsigned int numExtSems, void *stream) {
    if (!bridge_ok() || !g->p_hipSignalExternalSemaphoresAsync) return HIP_ERROR_NOT_INITIALIZED;
    return ((int SYSV (*)(const void *, const void *, unsigned int, void *))
                g->p_hipSignalExternalSemaphoresAsync)(extSemArray, paramsArray, numExtSems, stream);
}
