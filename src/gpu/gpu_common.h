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

#include <stdio.h>

#include "../octree/gpack_common.h"

#if defined(DEBUG) || defined(DEBUGTIME)
static FILE *debugFile = NULL;
#endif

/* test code on host (CPU) */
//#define TEST

#define VDIM1 (1)
#define VDIM2 (1)
#define VDIM3_T (4)
#define VDIM3_S (9)
#define VDIM3_L (16)
#define VDIM3 (16)
#define VDIM3_GRAD_T (5)
#define STOREDIM_T (10)
#define STOREDIM_S (35)
#define STOREDIM_GRAD_T (20)
#define STOREDIM_GRAD_S (56)
#if defined(GPU_SPDF)
  #define STOREDIM_L (84)
  #define STOREDIM_XL (120)
  #define MAXPRIM (20)
#else
  #define MAXPRIM (14)
  #define STOREDIM_L (84)
  #define STOREDIM_XL (84)
#endif
#define STORE_OPERATOR +=
#define TRANSDIM (8)
#define BUFFERSIZE (150000)
/* array indexing macros, where d1/d2/d3/d4 are array lengths per dimension
 * and i1/i2/i3/i4 are indexed positions per dimension */
#define LOC2(A,i1,i2,d1,d2) (A[(i2) * (d1) + (i1)])
#define LOC3(A,i1,i2,i3,d1,d2,d3) (A[((i1) * (d2) + (i2)) * (d3) + (i3)])
#define LOC4(A,i1,i2,i3,i4,d1,d2,d3,d4) (A[(((i1) * (d2) + (i2)) * (d3) + i3) * (d4) + (i4)])
// used in OEI and ERI code, auto-generated code
#define LOCSTORE(A,i1,i2,d1,d2) (A[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
// used in ERI code
#define LOCVY(A,i1,i2,i3,d1,d2,d3) (A[(((i1) * (d2) + (i2)) * (d3) + (i3)) * gridDim.x * blockDim.x])
// used in gpu_oei_definitions.h
#define LOCSTOREFULL(A,i1,i2,d1,d2,m) (A[(((m) * (d2) + (i2)) * (d1) + (i1)) * gridDim.x * blockDim.x])
// used in OEI and ERI code, auto-generated code
#define VY(a,b,c) LOCVY(YVerticalTemp, (a), (b), (c), VDIM1, VDIM2, VDIM3)

#define SQR(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))
#define MAX(A,B) (((A) > (B)) ? (A) : (B))
#define MIN(A,B) (((A) < (B)) ? (A) : (B))

/* GPU kernel resource constants */
#if defined(CUDA) || defined(CUDA_MPIV)
  /* constant for general purpose */
  #define SM_13_THREADS_PER_BLOCK (256)
  #define SM_2X_THREADS_PER_BLOCK (256)
  /* constant for 2e-integral */
  #define SM_13_2E_THREADS_PER_BLOCK (256)
  #define SM_2X_2E_THREADS_PER_BLOCK (256)
  /* constant for DFT Exchange-Correlation part */
  #define MAX_GRID (194)
  #define SM_13_XC_THREADS_PER_BLOCK (256)
  #define SM_2X_XC_THREADS_PER_BLOCK (256)
  /* constant for grad */
  #define SM_13_GRAD_THREADS_PER_BLOCK (256)
  #define SM_2X_GRAD_THREADS_PER_BLOCK (256)
  /* Launch parameters for octree based Exchange-Correlation part */
  #define SM_2X_XCGRAD_THREADS_PER_BLOCK MAX_POINTS_PER_CLUSTER
  #define SM_2X_SSW_GRAD_THREADS_PER_BLOCK (320)
#elif defined(HIP) || defined(HIP_MPIV)
#if defined(AMD_ARCH_GFX90a)
    /* constant for general purpose */
    #define SM_13_THREADS_PER_BLOCK (256)
    #define SM_2X_THREADS_PER_BLOCK (256)
    /* constant for 1e-integral */
    #define HIP_1E_THREADS_PER_BLOCK (512)
    #define HIP_1E_GRAD_THREADS_PER_BLOCK (512)
    /* constant for 2e-integral */
    #define SM_13_2E_THREADS_PER_BLOCK (256)
    #define SM_2X_2E_THREADS_PER_BLOCK (256)
    #define HIP_SP_2E_THREADS_PER_BLOCK (256)
    #define HIP_SPD_2E_THREADS_PER_BLOCK (256)
    #define HIP_SP_2E_WAVES_PER_CU (1)
    #define HIP_SPD_2E_WAVES_PER_CU (1)
    /* constant for DFT Exchange-Correlation part */
    #define MAX_GRID (194)
    #define SM_13_XC_THREADS_PER_BLOCK (256)
    #define SM_2X_XC_THREADS_PER_BLOCK (512)
    #define HIP_XC_WAVES_PER_CU (1)
    #define HIP_XC_DENSE_WAVES_PER_CU (1)
    #define HIP_XC_THREADS_PER_BLOCK (512)
    #define HIP_XC_DENSE_THREADS_PER_BLOCK (512)
    //static const int HIP_XC_GRAD_THREADS_PER_BLOCK (384)
    /* constant for grad */
    #define SM_13_GRAD_THREADS_PER_BLOCK (256)
    #define SM_2X_GRAD_THREADS_PER_BLOCK (256)
    #define HIP_SP_2E_GRAD_THREADS_PER_BLOCK (256)
    #define HIP_SPD_2E_GRAD_THREADS_PER_BLOCK (512)
    #define HIP_SPDF_2E_GRAD_THREADS_PER_BLOCK (256)
    #define HIP_SPDF2_2E_GRAD_THREADS_PER_BLOCK (256)
    #define HIP_SP_2E_GRAD_WAVES_PER_CU (1)
    #define HIP_SPD_2E_GRAD_WAVES_PER_CU (1)
    #define HIP_SPDF_2E_GRAD_WAVES_PER_CU (1)
    #define HIP_SPDF2_2E_GRAD_WAVES_PER_CU (1)
    /* constants for LRI */
    #define HIP_LRI_THREADS_PER_BLOCK (512)
    #define HIP_LRI_SPDF2_THREADS_PER_BLOCK (256)
    #define HIP_LRI_GRAD_THREADS_PER_BLOCK (512)
    #define HIP_LRI_GRAD_SPDF2_THREADS_PER_BLOCK (512)
    #define HIP_LRI_WAVES_PER_CU (1)
    #define HIP_LRI_SPDF2_WAVES_PER_CU (1)
    #define HIP_LRI_GRAD_WAVES_PER_CU (1)
    #define HIP_LRI_GRAD_SPDF2_WAVES_PER_CU (1)
    /* constants for cew quad kernels */
    #define HIP_CEW_QUAD_THREADS_PER_BLOCK (384)
    #define HIP_CEW_QUAD_GRAD_THREADS_PER_BLOCK (384)
    #define HIP_CEW_QUAD_WAVES_PER_CU (1)
    #define HIP_CEW_QUAD_GRAD_WAVES_PER_CU (1)
  #else
    /* constant for general purpose */
    #define SM_13_THREADS_PER_BLOCK (256)
    #define SM_2X_THREADS_PER_BLOCK (256)
    /* constant for 1e-integral */
    #define HIP_1E_THREADS_PER_BLOCK (512)
    #define HIP_1E_GRAD_THREADS_PER_BLOCK (512)
    /* constant for 2e-integral */
    #define SM_13_2E_THREADS_PER_BLOCK (256)
    #define SM_2X_2E_THREADS_PER_BLOCK (256)
    #define HIP_SP_2E_THREADS_PER_BLOCK (512)
    #define HIP_SPD_2E_THREADS_PER_BLOCK (768)
    #define HIP_SP_2E_WAVES_PER_CU (1)
    #define HIP_SPD_2E_WAVES_PER_CU (1)
    /* constant for DFT Exchange-Correlation part */
    #define MAX_GRID (194)
    #define SM_13_XC_THREADS_PER_BLOCK (256)
    #define SM_2X_XC_THREADS_PER_BLOCK (512)
    #define HIP_XC_WAVES_PER_CU (1)
    #define HIP_XC_DENSE_WAVES_PER_CU (1)
    #define HIP_XC_THREADS_PER_BLOCK (512)
    #define HIP_XC_DENSE_THREADS_PER_BLOCK (512)
    /* constant for grad */
    #define SM_13_GRAD_THREADS_PER_BLOCK (256)
    #define SM_2X_GRAD_THREADS_PER_BLOCK (256)
    #define HIP_SP_2E_GRAD_THREADS_PER_BLOCK (512)
    #define HIP_SPD_2E_GRAD_THREADS_PER_BLOCK (512)
    #define HIP_SPDF_2E_GRAD_THREADS_PER_BLOCK (768)
    #define HIP_SPDF2_2E_GRAD_THREADS_PER_BLOCK (768)
    #define HIP_SP_2E_GRAD_WAVES_PER_CU (1)
    #define HIP_SPD_2E_GRAD_WAVES_PER_CU (1)
    #define HIP_SPDF_2E_GRAD_WAVES_PER_CU (1)
    #define HIP_SPDF2_2E_GRAD_WAVES_PER_CU (1)
    /* constants for LRI */
    #define HIP_LRI_THREADS_PER_BLOCK (768)
    #define HIP_LRI_SPDF2_THREADS_PER_BLOCK (256)
    #define HIP_LRI_GRAD_THREADS_PER_BLOCK (512)
    #define HIP_LRI_GRAD_SPDF2_THREADS_PER_BLOCK (768)
    #define HIP_LRI_WAVES_PER_CU (1)
    #define HIP_LRI_SPDF2_WAVES_PER_CU (1)
    #define HIP_LRI_GRAD_WAVES_PER_CU (1)
    #define HIP_LRI_GRAD_SPDF2_WAVES_PER_CU (1)
    /* constants for cew quad kernels */
    #define HIP_CEW_QUAD_THREADS_PER_BLOCK (256)
    #define HIP_CEW_QUAD_GRAD_THREADS_PER_BLOCK (256)
    #define HIP_CEW_QUAD_WAVES_PER_CU (1)
    #define HIP_CEW_QUAD_GRAD_WAVES_PER_CU (1)
  #endif

  /* Launch parameters for octree based Exchange-Correlation part */
  #define SM_2X_XCGRAD_THREADS_PER_BLOCK MAX_POINTS_PER_CLUSTER;
  #define SM_2X_SSW_GRAD_THREADS_PER_BLOCK (320)
#endif

/* math constants, same as in quick_constants_module */
#define PI (3.1415926535897932384626433832795)
#define X0 (5.9149671727956128778234784350536) /* sqrt(2 * PI ^ 2.5) */
#define PI_TO_3HALF (5.5683279968317079)
#define PIE4 (0.78539816339744827900)


/* define QUICK floating point types
 * for DPDP:
 *   QUICKDouble = double
 *   QUICKSingle = float
 * for SPSP:
 *   QUICKDouble = float
 *   QUICKSingle = float
 */
//typedef double QUICKDouble;
#define QUICKDouble double
//typedef float  QUICKDouble;
typedef float QUICKSingle;
#define QUICKULL unsigned long long int
#if defined(USE_LEGACY_ATOMICS)
  #define QUICKAtomicType unsigned long long int
#else
  #define QUICKAtomicType double
#endif


/* SM Version enum */
enum SM_VERSION
{
    SM_10,
    SM_11,
    SM_12,
    SM_13,
    SM_2X,
};

enum QUICK_METHOD
{
    HF = 0,
    B3LYP = 1,
    BLYP  = 2,
    LIBXC = 3,
};

struct ERI_entry {
    int IJ;
    int KL;
    QUICKDouble value;
};


#if defined(USE_LEGACY_ATOMICS)
  /* powers in the following constants represent the number of digits
   * to keep in the mantissa (fractional part) when converting floating point values
   * to/from integer values for use in GPU integer atomic operations */

  /* keep 12 digits in the mantissa (fractional part) for energy-related calculations */
  #define OSCALE (1.0e12)
  #define ONEOVEROSCALE (1.0e-12)
  /* keep 16 digits in the mantissa (fractional part) for gradient-related calculations */
  #define GRADSCALE (1.0e16)
  #define ONEOVERGRADSCALE (1.0e-16)

  /* convert QUICKDouble to unsigned long long before accumlating using integer atomics */
  #if defined(CUDA) || defined(CUDA_MPIV)
    #define GPUATOMICADD(address, val, scale) \
{ \
    unsigned long long int temp_ULL; \
    asm("mov.b64    %0, %1;" : "=l"(temp_ULL) : "l"(llrint((val) * (scale)))); \
    atomicAdd((address), temp_ULL); \
}
  #elif defined(HIP) || defined(HIP_MPIV)
    #define GPUATOMICADD(address, val, scale) atomicAdd((address), (unsigned long long int) llrint((val) * (scale)));
  #endif

  /* convert unsigned long long to QUICKDouble in host code */
  #define ULLTODOUBLE(val) (((val) >= 0x8000000000000000ull) ? -(QUICKDouble) ((val) ^ 0xffffffffffffffffull) : (QUICKDouble) (val))

  /* convert QUICKDouble to unsigned long long in host code */
  #define DOUBLETOULL(val) ((unsigned long long int) llrint((val)))
#else
  /* iterative (computationally expensive) approach to emulate full double
   * precision atomics using integer atomics on GPUs lacking hardware support,
   * as given in the CUDA developer documentation */
  #if ((defined(CUDA) || defined(CUDA_MPIV)) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600) \
      || ((defined(HIP) || defined(HIP_MPIV)) && !defined(__HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__))
__device__ static inline double atomicAdd(double * address, double val)
{
    unsigned long long int *address_as_ull, old, assumed;

    address_as_ull = (unsigned long long int *) address;
    old = *address_as_ull;

    do
    {
        assumed = old;
        old = atomicCAS( address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)) );
    }
    while ( assumed != old );

    return __longlong_as_double( old );
}
  #endif

  #define GPUATOMICADD(address, val, scale) atomicAdd((address), (val));
#endif


#endif
