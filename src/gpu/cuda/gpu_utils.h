#if !defined(__QUICK_GPU_UTILS_H__)
#define __QUICK_GPU_UTILS_H__

#include <stdio.h>

#include "../gpu_common.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
//#include "nvToolsExt.h"

#if defined(DEBUG) || defined(DEBUGTIME)
static FILE *debugFile = NULL;
#endif

#define PRINTERROR(err, s) \
{ \
    if ((err) != cudaSuccess) { \
        printf( "%s: %s in %s at line %d\n", (s), cudaGetErrorString((err)), __FILE__, __LINE__ ); \
    } \
}

#if defined(DEBUG)
  #define PRINTDEBUGNS(s) \
  { \
      fprintf(debugFile,"FILE:%15s, LINE:%5d DATE: %s TIME:%s DEBUG : %s. \n", __FILE__, __LINE__, __DATE__, __TIME__, (s) ); \
  }

  #define PRINTDEBUG(s) \
  { \
      fprintf(gpu->debugFile,"FILE:%15s, LINE:%5d DATE: %s TIME:%s DEBUG : %s. \n", __FILE__, __LINE__, __DATE__, __TIME__, (s) ); \
  }

  #define PRINTUSINGTIME(s,time) \
  { \
      fprintf(gpu->debugFile,"TIME:%15s, LINE:%5d DATE: %s TIME:%s TIMING:%20s ======= %f ms =======.\n", __FILE__, __LINE__, __DATE__, __TIME__, (s), (time)); \
  }

  #define PRINTMEM(s,a) \
  { \
  	fprintf(gpu->debugFile,"MEM :%15s, LINE:%5d DATE: %s TIME:%s MEM   : %10s %lli\n", __FILE__, __LINE__, __DATE__, __TIME__, (s), (a)); \
  }
#else
  #define PRINTDEBUGNS(s)
  #define PRINTDEBUG(s)
  #define PRINTUSINGTIME(s,time)
  #define PRINTMEM(s,a)
#endif


/* debug timer */
#if defined(DEBUG) || defined(DEBUGTIME)
  #define TIMERSTART() \
  cudaEvent_t start, end; \
  float time; \
  cudaEventCreate(&start); \
  cudaEventCreate(&end); \
  cudaEventRecord(start, 0);

  #define TIMERSTOP() \
  cudaEventRecord(end, 0); \
  cudaEventSynchronize(end); \
  cudaEventElapsedTime(&time, start, end); \
  totTime += time; \
  fprintf(gpu->debugFile, "this cycle:%f ms total time:%f ms\n", time, totTime); \
  cudaEventDestroy(start); \
  cudaEventDestroy(end); 
#endif

/* GPU safe call */
#if defined(DEBUG) || defined(DEBUGTIME)
  #define QUICK_SAFE_CALL(x) \
  { \
    TIMERSTART() \
    fprintf(gpu->debugFile, "%s.%s.%d: %s\n", __FILE__, __FUNCTION__, __LINE__, #x); \
    fprintf(gpu->debugFile, "LAUCHBOUND = %i %i\n", gpu->blocks, gpu->twoEThreadsPerBlock); \
    fprintf(gpu->debugFile, "METHOD = %i\n", gpu->gpu_sim.method); \
    fflush(gpu->debugFile); \
    x; \
    TIMERSTOP() \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess && error != cudaErrorNotReady) { \
      printf("%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error)); \
      exit(1); \
    } \
  }
#else
  #define QUICK_SAFE_CALL(x) {x;}
#endif

#define SAFE_DELETE(a) \
  if ( (a) != NULL ) delete (a); \
  (a) = NULL;


#endif
