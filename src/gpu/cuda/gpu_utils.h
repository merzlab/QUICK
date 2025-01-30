#if !defined(__QUICK_GPU_UTILS_H_)
#define __QUICK_GPU_UTILS_H_

#include <stdio.h>

#include "../gpu_common.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
//#include "nvToolsExt.h"


void _gpuGetDeviceCount(int *, const char * const, int);
void _gpuSetDevice(int, const char * const, int);
void _gpuMalloc(void **, size_t, const char * const, int);
void _gpuHostAlloc(void **, size_t, unsigned int, const char * const, int);
void _gpuFree(void *, const char * const, int);
void _gpuFreeHost(void *, const char * const, int);
void _gpuMemset(void *, int, size_t, const char * const, int);
void _gpuMemsetAsync(void *, int, size_t, cudaStream_t, const char * const, int);
void _gpuCheckMalloc(void **, size_t *, size_t, const char * const, int);
void _gpuMemcpy(void * const, void const * const, size_t,
        enum cudaMemcpyKind, const char * const, int);
void _gpuMemcpyAsync(void * const, void const * const, size_t,
        enum cudaMemcpyKind, cudaStream_t, const char * const, int);
void _gpuMemcpyToSymbol(void const * const, void const * const, size_t,
        const char * const, int);
void _gpuHostGetDevicePointer(void **, void * const, unsigned int,
        const char * const, int);
void _gpuHostAllocCheck(void **, size_t *, size_t, unsigned int, int, double,
        const char * const, int);
void _gpuHostReallocCheck(void **, size_t *, size_t, unsigned int, int, double,
        const char * const, int);
void _gpuEventCreate(cudaEvent_t *, const char * const, int);
void _gpuEventDestroy(cudaEvent_t, const char * const, int);
void _gpuEventElapsedTime(float *, cudaEvent_t, cudaEvent_t, const char * const, int);
void _gpuEventRecord(cudaEvent_t, cudaStream_t, const char * const, int);
void _gpuEventSynchronize(cudaEvent_t, const char * const, int);

#define gpuGetDeviceCount(c) _gpuGetDeviceCount((c), __FILE__, __LINE__);
#define gpuSetDevice(d) _gpuSetDevice((d), __FILE__, __LINE__);
#define gpuMalloc(p, s) _gpuMalloc((p), (s), __FILE__, __LINE__);
#define gpuHostAlloc(p, s, f) _gpuHostAlloc((p), (s), (f), __FILE__, __LINE__);
#define gpuFree(p) _gpuFree((p), __FILE__, __LINE__);
#define gpuFreeHost(p) _gpuFreeHost((p), __FILE__, __LINE__);
#define gpuMemset(p, d, c) _gpuMemset((p), (d), (c), __FILE__, __LINE__);
#define gpuMemsetAsync(p, d, c, s) _gpuMemsetAsync((p), (d), (c), (s), __FILE__, __LINE__);
#define gpuCheckMalloc(p, cs, ns) _gpuCheckMalloc((p), (cs), (ns), __FILE__, __LINE__);
#define gpuMemcpy(d, s, c, dr) _gpuMemcpy((d), (s), (c), (dr), __FILE__, __LINE__);
#define gpuMemcpyAsync(d, s, c, dr, st) _gpuMemcpyAsync((d), (s), (c), (dr), (st), __FILE__, __LINE__);
#define gpuMemcpyToSymbol(d, s, c) _gpuMemcpyToSymbol((d), (s), (c), __FILE__, __LINE__);
#define gpuHostGetDevicePointer(pd, ph, f) _gpuHostGetDevicePointer((pd), (ph), (f), __FILE__, __LINE__);
#define gpuHostAllocCheck(p, cs, ns, f, oa, oaf) _gpuHostAllocCheck((p), (cs), (ns), (f), (oa), (oaf), __FILE__, __LINE__);
#define gpuHostReallocCheck(p, cs, ns, f, oa, oaf) _gpuHostReallocCheck((p), (cs), (ns), (f), (oa), (oaf), __FILE__, __LINE__);
#define gpuEventCreate(e) _gpuEventCreate((e), __FILE__, __LINE__);
#define gpuEventDestroy(e) _gpuEventDestroy((e), __FILE__, __LINE__);
#define gpuEventElapsedTime(t, s, e) _gpuEventElapsedTime((t), (s), (e), __FILE__, __LINE__);
#define gpuEventRecord(e, s) _gpuEventRecord((e), (s), __FILE__, __LINE__);
#define gpuEventSynchronize(e) _gpuEventSynchronize((e), __FILE__, __LINE__);


#define PRINTERROR(err, s) \
{ \
    if ((err) != cudaSuccess) { \
        printf("%s: %s in %s at line %d\n", (s), cudaGetErrorString((err)), __FILE__, __LINE__); \
    } \
}

#if defined(DEBUG)
  #define PRINTDEBUGNS(s) \
{ \
    fprintf(debugFile, "FILE: %15s, LINE: %5d, DATE: %s, TIME: %s, DEBUG: %s.\n", \
            __FILE__, __LINE__, __DATE__, __TIME__, (s)); \
}

  #define PRINTDEBUG(s) \
{ \
    fprintf(gpu->debugFile, "FILE: %15s, LINE: %5d, DATE: %s, TIME: %s, DEBUG: %s.\n", \
            __FILE__, __LINE__, __DATE__, __TIME__, (s)); \
}

  #define PRINTUSINGTIME(s,time) \
{ \
    fprintf(gpu->debugFile, "TIME: %15s, LINE: %5d, DATE: %s, TIME: %s, TIMING: %20s ======= %f ms =======.\n", \
            __FILE__, __LINE__, __DATE__, __TIME__, (s), (time)); \
}

  #define PRINTMEM(s,a) \
{ \
    fprintf(gpu->debugFile, "MEM: %15s, LINE: %5d, DATE: %s, TIME: %s, MEM   : %10s %lli.\n", \
            __FILE__, __LINE__, __DATE__, __TIME__, (s), (a)); \
}
#else
  #define PRINTDEBUGNS(s)
  #define PRINTDEBUG(s)
  #define PRINTUSINGTIME(s,time)
  #define PRINTMEM(s,a)
#endif


/* timer code for GPU kernels */
#define GPU_TIMER_CREATE() \
  cudaEvent_t start, end; \
  float time; \
  gpuEventCreate(&start); \
  gpuEventCreate(&end);

#define GPU_TIMER_START() \
  gpuEventRecord(start, 0);

#define GPU_TIMER_STOP() \
  gpuEventRecord(end, 0); \
  gpuEventSynchronize(end); \
  gpuEventElapsedTime(&time, start, end);

#define GPU_TIMER_DESTROY() \
  gpuEventDestroy(start); \
  gpuEventDestroy(end);

/* GPU safe call */
#if defined(DEBUG) || defined(DEBUGTIME)
  #define QUICK_SAFE_CALL(x) \
{ \
    GPU_TIMER_CREATE(); \
    GPU_TIMER_START(); \
    fprintf(gpu->debugFile, "%s.%s.%d: %s\n", __FILE__, __FUNCTION__, __LINE__, #x); \
    fprintf(gpu->debugFile, "  LAUNCHBOUND = %i %i\n", gpu->blocks, gpu->twoEThreadsPerBlock); \
    fprintf(gpu->debugFile, "  METHOD = %i\n", gpu->gpu_sim.method); \
    fflush(gpu->debugFile); \
    x; \
    GPU_TIMER_STOP(); \
    totTime += time; \
    fprintf(gpu->debugFile, "this cycle:%f ms total time:%f ms\n", time, totTime); \
    cudaError_t error = cudaDeviceSynchronize(); \
    if (error != cudaSuccess) { \
        printf("%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
    GPU_TIMER_DESTROY(); \
}
#else
  #define QUICK_SAFE_CALL(x) {x;}
#endif

#define SAFE_DELETE(a) \
    if ((a) != NULL) delete (a); \
    (a) = NULL;


#endif
