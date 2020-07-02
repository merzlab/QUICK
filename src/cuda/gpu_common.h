/*
 *  gpu_common.h
 *  new_quick
 *
 *  Created by Yipu Miao on 6/3/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */

#ifndef QUICK_GPU_COMMON_H
#define QUICK_GPU_COMMON_H

#include "../config.h"
#include <stdio.h>
#include <cuda.h>
#include "nvToolsExt.h"
#include "../octree/gpack_common.h"

#ifdef DEBUG
static FILE *debugFile = NULL;
#endif

#define PRINTMETHOD(mtd)\
{\
fprintf(debugFile,"METHOD:%s\n",mtd);\
fflush(stdout);\
}

// Define TEST for the CPU host test, and undef it when you need to run it on device
//#define TEST

#define VDIM1 1
#define VDIM2 1
//#define VDIM3 10
#define VDIM3 16

#define STOREDIM_S 35

#ifdef CUDA_SPDF
#define STOREDIM_L 84
#else
#define STOREDIM_L 84
#endif

#define TRANSDIM 8
#define MCALDIM 120

#define MAXPRIM 10

#define BUFFERSIZE 150000

// Macro for two- and three- dimension array, d1,d2 and d3 are the dimension and i1,i2 and i3 are the indices
#define LOC2(A,i1,i2,d1,d2)  A[i1+(i2)*(d1)]
#define LOC3(A,i1,i2,i3,d1,d2,d3) A[i3+((i2)+(i1)*(d2))*(d3)]
#define LOC4(A,i1,i2,i3,i4,d1,d2,d3,d4) A[i4+(i3+((i2)+(i1)*(d2))*(d3))*(d4)]

#define MAX(A,B)    (A>=B?A:B)
#define MIN(A,B)    (A<B?A:B)

#define VY(a,b,c) LOC3(YVerticalTemp, a, b, c, VDIM1, VDIM2, VDIM3)

#define PRINTERROR(err, s) \
{\
    if (err != cudaSuccess) {\
        printf( "%s: %s in %s at line %d\n", s, cudaGetErrorString(err), __FILE__, __LINE__ ); \
    }\
}

#ifdef DEBUG
#define PRINTDEBUG(s) \
{\
    fprintf(debugFile,"FILE:%15s, LINE:%5d DATE: %s TIME:%s DEBUG : %s. \n", __FILE__,__LINE__,__DATE__,__TIME__,s );\
}

#define PRINTUSINGTIME(s,time)\
{\
    fprintf(debugFile,"TIME:%15s, LINE:%5d DATE: %s TIME:%s TIMING:%20s ======= %f ms =======.\n", __FILE__, __LINE__, __DATE__,__TIME__,s,time);\
}

#define PRINTMEM(s,a) \
{\
	fprintf(debugFile,"MEM :%15s, LINE:%5d DATE: %s TIME:%s MEM   : %10s %lli\n", __FILE__,__LINE__,__DATE__,__TIME__,s,a);\
}
#else
#define PRINTDEBUG(s)
#define PRINTUSINGTIME(s,time)
#define PRINTMEM(s,a)
#endif


// Timer for debug
#ifdef DEBUG
#define TIMERSTART() \
cudaEvent_t start,end;\
float time;\
cudaEventCreate(&start); \
cudaEventCreate(&end);\
cudaEventRecord(start, 0);

#define TIMERSTOP() \
cudaEventRecord(end, 0); \
cudaEventSynchronize(end); \
cudaEventElapsedTime(&time, start, end); \
totTime+=time; \
fprintf(gpu->debugFile,"this cycle:%f ms total time:%f ms\n", time, totTime); \
cudaEventDestroy(start); \
cudaEventDestroy(end); 

#endif


// Atomic add macro
#ifdef TEST
#define QUICKADD(address, val) address += (val)
#define QUICKSUB(address, val) address -= (val)
#else
#define QUICKADD(address, val)  atomicAdd(&(address),(val))
#define QUICKSUB(address, val)  atomicAdd(&(address),(val))
#endif

#define TEXDENSE(a,b) fetch_texture_double(textureDense, (a-1)*devSim.nbasis+(b-1))



#define GRADADD(address, val) \
{ \
    QUICKULL val2 = (QUICKULL) (fabs((val)*GRADSCALE) + (QUICKDouble)0.5); \
    if ( val < (QUICKDouble)0.0) val2 = 0ull - val2; \
    QUICKADD(address, val2); \
}


// CUDA safe call
#ifdef DEBUG
#define QUICK_SAFE_CALL(x)\
{\
TIMERSTART()\
fprintf(gpu->debugFile,"%s.%s.%d: %s\n", __FILE__, __FUNCTION__, __LINE__, #x);fflush(gpu->debugFile);\
fprintf(gpu->debugFile,"LAUCHBOUND = %i %i\n", gpu->blocks, gpu->twoEThreadsPerBlock);\
fprintf(gpu->debugFile,"METHOD = %i\n", gpu->gpu_sim.method);\
x;\
TIMERSTOP()\
cudaError_t error = cudaGetLastError();\
if (error != cudaSuccess && error != cudaErrorNotReady)\
{ printf("%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error));  \
  exit(1);                                                                                              \
}\
}
#else
#define QUICK_SAFE_CALL(x) {x;}
#endif

#define SAFE_DELETE(a) if( (a) != NULL ) delete (a); (a) = NULL;

/*
 ****************************************************************
 *  common variables
 ****************************************************************
 */
/*  define quick type 
 for DPDP: QUICKDouble = double
 QUICKSingle = float
 for SPSP: QUICKDouble = float
 QUICKSingle = float
 */
//typedef double QUICKDouble;
#define QUICKDouble double
//typedef float  QUICKDouble;
typedef float  QUICKSingle;
#define QUICKULL \
unsigned long long int

/* 
 ****************************************************************
 *  constant define
 ****************************************************************
 */
// constant for general purpose
static const int SM_13_THREADS_PER_BLOCK    =   256;
static const int SM_2X_THREADS_PER_BLOCK    =   256;

// constant for 2e-integral
static const int SM_13_2E_THREADS_PER_BLOCK =   256;
static const int SM_2X_2E_THREADS_PER_BLOCK =   256;

// constant for DFT Exchange-Correlation part
static const int MAX_GRID                   =   194;
static const int SM_13_XC_THREADS_PER_BLOCK =   256;
static const int SM_2X_XC_THREADS_PER_BLOCK =   256;


// constant for grad
static const int SM_13_GRAD_THREADS_PER_BLOCK =   256;
static const int SM_2X_GRAD_THREADS_PER_BLOCK =   256;

//Launch parameters for octree based Exchange-Correlation part
static const int SM_2X_XCGRAD_THREADS_PER_BLOCK = MAX_POINTS_PER_CLUSTER;

// physical constant, the same with quick_constants_module
//static const QUICKDouble PI                 =   (QUICKDouble)3.1415926535897932384626433832795;
//static const QUICKSingle PI_FLOAT           =   (QUICKSingle)3.1415926535897932384626433832795;
#define PI (3.1415926535897932384626433832795)
#define X0 (5.9149671727956128778234784350536)//sqrt(2*PI^2.5)


// Energy Scale
static const QUICKDouble OSCALE                  = (QUICKDouble) 1E16;
static const QUICKDouble ONEOVEROSCALE           = (QUICKDouble)1.0 / OSCALE;
static const QUICKDouble ONEOVEROSCALESQUARED    = (QUICKDouble)1.0 / (OSCALE * OSCALE);


static const QUICKDouble GRADSCALE                  = (QUICKDouble)1E16;
static const QUICKDouble ONEOVERGRADSCALE           = (QUICKDouble)1.0 / OSCALE;
static const QUICKDouble ONEOVERGRADSCALESQUARED    = (QUICKDouble)1.0 / (OSCALE * OSCALE);

// SM Version enum
enum SM_VERSION
{
    SM_10,
    SM_11,
    SM_12,
    SM_13,
    SM_2X
};

enum QUICK_METHOD
{
    HF    = 0,
    B3LYP = 1,
    DFT   = 2,
    LIBXC = 3
};

struct ERI_entry{
    int IJ;
    int KL;
    QUICKDouble value;
};

#endif
