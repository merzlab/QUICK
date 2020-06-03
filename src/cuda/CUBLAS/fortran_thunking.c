/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This file contains example Fortran bindings for the CUBLAS library, These
 * bindings have been tested with Intel Fortran 9.0 on 32-bit and 64-bit 
 * Windows, and with g77 3.4.5 on 32-bit and 64-bit Linux. They will likely
 * have to be adjusted for other Fortran compilers and platforms.
 */
 
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#if defined(__GNUC__)
#include <stdint.h>
#endif /* __GNUC__ */
#include "cublas.h"   /* CUBLAS public header file  */


#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)<(b))?(b):(a))

#include "fortran_common.h"
#include "fortran_thunking.h"


#define CUBLAS_WRAPPER_ERROR_NOERR      0
#define CUBLAS_WRAPPER_ERROR_ALLOC      1
#define CUBLAS_WRAPPER_ERROR_SET        2
#define CUBLAS_WRAPPER_ERROR_GET        3
#define CUBLAS_WRAPPER_ERROR_STUB       4

static char *errMsg[5] = 
{
    "no error",
    "allocation error",
    "setVector/setMatrix error",
    "getVector/getMatrix error",
    "not implemented"
};

static void wrapperError (const char *funcName, int error)
{
    printf ("cublas%s wrapper: %s\n", funcName, errMsg[error]);
    fflush (stdout);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS1 ----------------------------------*/
/*---------------------------------------------------------------------------*/

int CUBLAS_ISAMAX (const int *n, const float *x, const int *incx)
{
    float *devPtrx = 0;
    int retVal = 0;
    cublasStatus stat;

    if (*n <= 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx), sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Isamax", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Isamax", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasIsamax (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
} 

int CUBLAS_ISAMIN (const int *n, const float *x, const int *incx)
{
    float *devPtrx = 0;
    int retVal = 0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx), sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Isamin", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Isamin", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasIsamin (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SASUM (const int *n, const float *x, const int *incx)
#else
float CUBLAS_SASUM (const int *n, const float *x, const int *incx)
#endif
{
    float *devPtrx = 0;
    float retVal = 0.0f;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx), sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sasum", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sasum", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
   
    retVal = cublasSasum (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

void CUBLAS_SAXPY (const int *n, const float *alpha, const float *x, 
                   const int *incx, float *y, const int *incy)
{
    float *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Saxpy", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Saxpy", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasSaxpy (*n, *alpha, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Saxpy", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SCOPY (const int *n, const float *x, const int *incx, float *y,
                   const int *incy)
{
    float *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Scopy", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Scopy", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasScopy (*n, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Scopy", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SDOT (const int *n, const float *x, const int *incx, float *y,
                    const int *incy)
#else
float CUBLAS_SDOT (const int *n, const float *x, const int *incx, float *y,
                   const int *incy)
#endif
{
    float *devPtrx = 0, *devPtry = 0, retVal = 0.0f;
    cublasStatus stat1, stat2;

    if (*n == 0) return retVal;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sdot", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sdot", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
retVal = cublasSdot (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
    return retVal;
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SNRM2 (const int *n, const float *x, const int *incx)
#else
float CUBLAS_SNRM2 (const int *n, const float *x, const int *incx)
#endif
{
    float *devPtrx = 0;
    float retVal = 0.0f;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Snrm2", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Snrm2", CUBLAS_WRAPPER_ERROR_SET);
        return retVal;
    }
    retVal = cublasSnrm2 (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

void CUBLAS_SROT (const int *n, float *x, const int *incx, float *y, 
                  const int *incy, const float *sc, const float *ss)
{
    float *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Srot", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx); 
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Srot", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasSrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *ss);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Srot", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SROTG (float *sa, float *sb, float *sc, float *ss)
{
    cublasSrotg (sa, sb, sc, ss);
}

void CUBLAS_SROTM (const int *n, float *x, const int *incx, float *y, 
                   const int *incy, const float* sparam)
{
    float *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Srotm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Srotm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasSrotm (*n, devPtrx, *incx, devPtry, *incy, sparam);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Srotm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SROTMG (float *sd1, float *sd2, float *sx1, const float *sy1,
                    float* sparam)
{
    cublasSrotmg (sd1, sd2, sx1, sy1, sparam);
}

void CUBLAS_SSCAL (const int *n, const float *alpha, float *x, const int *incx)
{
    float *devPtrx = 0;
    cublasStatus stat;

    if (*n == 0) return;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sscal", CUBLAS_WRAPPER_ERROR_ALLOC);
        return;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sscal", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return;
    }
    cublasSscal (*n, *alpha, devPtrx, *incx);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, *incx, x, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sscal", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx); 
}

void CUBLAS_SSWAP (const int *n, float *x, const int *incx, float *y, 
                   const int *incy)
{
    float *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sswap", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sswap", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasSswap (*n, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sswap", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CAXPY (const int *n, const cuComplex *alpha, const cuComplex *x, 
                   const int *incx, cuComplex *y, const int *incy)
{
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Caxpy", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Caxpy", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasCaxpy (*n, *alpha, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Caxpy", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_ZAXPY (const int *n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, 
                   const int *incx, cuDoubleComplex *y, const int *incy)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zaxpy", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zaxpy", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasZaxpy (*n, *alpha, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zaxpy", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CCOPY (const int *n, const cuComplex *x, const int *incx, 
                   cuComplex *y, const int *incy)
{
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ccopy", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ccopy", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasCcopy (*n, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ccopy", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_ZCOPY (const int *n, const cuDoubleComplex *x, const int *incx, 
                   cuDoubleComplex *y, const int *incy)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zcopy", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zcopy", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasZcopy (*n, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zcopy", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CROT (const int *n, cuComplex *x, const int *incx, cuComplex *y, 
                  const int *incy, const float *sc, const cuComplex *cs)
{
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Crot", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Crot", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasCrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *cs);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Crot", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_ZROT (const int *n, cuDoubleComplex *x, const int *incx, cuDoubleComplex *y, 
                  const int *incy, const double *sc, const cuDoubleComplex *cs)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zrot", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zrot", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasZrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *cs);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zrot", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CROTG (cuComplex *ca, const cuComplex *cb, float *sc,
                   cuComplex *cs)
{
    cublasCrotg (ca, *cb, sc, cs);
}

void CUBLAS_ZROTG (cuDoubleComplex *ca, const cuDoubleComplex *cb, double *sc,
                   cuDoubleComplex *cs)
{
    cublasZrotg (ca, *cb, sc, cs);
}

void CUBLAS_CSCAL (const int *n, const cuComplex *alpha, cuComplex *x, 
                   const int *incx)
{
    cuComplex *devPtrx = 0;
    cublasStatus stat;
    
    if (*n == 0) return;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cscal", CUBLAS_WRAPPER_ERROR_ALLOC);
        return;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cscal", CUBLAS_WRAPPER_ERROR_SET);
        return;
    }
    cublasCscal (*n, *alpha, devPtrx, *incx);
    stat = cublasGetVector (*n, sizeof(x[0]), devPtrx, *incx, x, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cscal", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx); 
}

void CUBLAS_CSROT (const int *n, cuComplex *x, const int *incx, cuComplex *y, 
                   const int *incy, const float *sc, const float *ss)
{
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csrot", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csrot", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasCsrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *ss);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csrot", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_ZDROT (const int *n, cuDoubleComplex *x, const int *incx, cuDoubleComplex *y, 
                   const int *incy, const double *sc, const double *ss)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdrot", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdrot", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasZdrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *ss);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdrot", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CSSCAL (const int *n, const float *alpha, cuComplex *x, 
                    const int *incx)
{
    cuComplex *devPtrx = 0;
    cublasStatus stat;

    if (*n == 0) return;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Csscal", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        return;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Csscal", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return;
    }
    cublasCsscal (*n, *alpha, devPtrx, *incx);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, *incx, x, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Csscal", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx); 
}

                   
void CUBLAS_CSWAP (const int *n, cuComplex *x, const int *incx, cuComplex *y,
                   const int *incy)
{
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (imax(1,*n *abs(*incy)),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cswap", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cswap", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasCswap (*n, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cswap", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CTRMV (const char *uplo, const char *trans,
                   const char *diag, const int *n, const cuComplex *A,
                   const int *lda, cuComplex *x, const int *incx)
{
    cuComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;
    
    if (*n == 0) return;
    
    /*  X      - COMPLEX           array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - COMPLEX            array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */    
    
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctrmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctrmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasCtrmv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    stat1=cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ctrmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

void CUBLAS_ZTRMV (const char *uplo, const char *trans,
                   const char *diag, const int *n, const cuDoubleComplex *A,
                   const int *lda, cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;
    
    if (*n == 0) return;
    
    /*  X      - COMPLEX           array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - COMPLEX            array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztrmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztrmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasZtrmv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    stat1=cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ztrmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

void CUBLAS_ZSWAP (const int *n, cuDoubleComplex *x, const int *incx, cuDoubleComplex *y,
                   const int *incy)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (imax(1,*n *abs(*incy)),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zswap", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zswap", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasZswap (*n, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zswap", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}


#ifdef RETURN_COMPLEX
cuComplex CUBLAS_CDOTU ( const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy)
{
  
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    cuComplex retVal = make_cuComplex (0.0f, 0.0f);
    if (*n == 0) return retVal;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cdotu", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cdotu", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
  
    retVal = cublasCdotu (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
    return( retVal );
    
}
#else
void CUBLAS_CDOTU (cuComplex *retVal, const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy)

               
{ 
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    *retVal = make_cuComplex (0.0f, 0.0f);
    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cdotu", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cdotu", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    *retVal = cublasCdotu (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
  
}
#endif

#ifdef RETURN_COMPLEX
cuComplex CUBLAS_CDOTC ( const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy)
{
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    cuComplex retVal = make_cuComplex (0.0f, 0.0f);
    if (*n == 0) return retVal;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cdotc", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cdotc", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    retVal = cublasCdotc (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
    return retVal;
}
#else
void CUBLAS_CDOTC (cuComplex *retVal, const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy)
                  
{
    cuComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    *retVal = make_cuComplex (0.0f, 0.0f);
    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cdotc", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cdotc", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    *retVal = cublasCdotc (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}
#endif

int CUBLAS_ICAMAX (const int *n, const cuComplex *x, const int *incx)
{
    cuComplex *devPtrx = 0;
    int retVal = 0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Icamax", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Icamax", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasIcamax (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

int CUBLAS_ICAMIN (const int *n, const cuComplex *x, const int *incx)
{
    cuComplex *devPtrx = 0;
    int retVal = 0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Icamin", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Icamin", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasIcamin (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

int CUBLAS_IZAMAX (const int *n, const cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrx = 0;
    int retVal = 0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Izamax", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Izamax", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasIzamax (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

int CUBLAS_IZAMIN (const int *n, const cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrx = 0;
    int retVal = 0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Izamin", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Izamin", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasIzamin (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SCASUM (const int *n, const cuComplex *x, const int *incx)
#else
float CUBLAS_SCASUM (const int *n, const cuComplex *x, const int *incx)
#endif
{
    cuComplex *devPtrx = 0;
    float retVal = 0.0f;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Scasum", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Scasum", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasScasum (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

double CUBLAS_DZASUM (const int *n, const cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrx = 0;
    double retVal = 0.0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dzasum", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dzasum", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasDzasum (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SCNRM2 (const int *n, const cuComplex *x, const int *incx)
#else
float CUBLAS_SCNRM2 (const int *n, const cuComplex *x, const int *incx)
#endif
{
    cuComplex *devPtrx = 0;
    float retVal = 0.0f;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Scnrm2", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Scnrm2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasScnrm2 (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}
 
double CUBLAS_DZNRM2 (const int *n, const cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrx = 0;
    double retVal = 0.0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dznrm2", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dznrm2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasDznrm2 (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

int CUBLAS_IDAMAX (const int *n, const double *x, const int *incx)
{
    double *devPtrx = 0;
    int retVal = 0;
    cublasStatus stat;

    if (*n == 0) return retVal;;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Idamax", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Idamax", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasIdamax (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

int CUBLAS_IDAMIN (const int *n, const double *x, const int *incx)
{
    double *devPtrx = 0;
    int retVal = 0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Idamin", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Idamin", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasIdamin (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

double CUBLAS_DASUM (const int *n, const double *x, const int *incx)
{
    double *devPtrx = 0;
    double retVal = 0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dasum", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dasum", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        return retVal;
    }
    retVal = cublasDasum (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

void CUBLAS_DAXPY (const int *n, const double *alpha, const double *x, 
                   const int *incx, double *y, const int *incy)
{
    double *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Daxpy", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Daxpy", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasDaxpy (*n, *alpha, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Daxpy", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_DCOPY (const int *n, const double *x, const int *incx, double *y,
                   const int *incy)
{
    double *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dcopy", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dcopy", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasDcopy (*n, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dcopy", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

double CUBLAS_DDOT (const int *n, const double *x, const int *incx, double *y,
                    const int *incy)
{
    double *devPtrx = 0, *devPtry = 0;
    double retVal = 0.0;
    cublasStatus stat1, stat2;

    if (*n == 0) return retVal;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ddot", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ddot", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    retVal = cublasDdot (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
    return retVal;
}

double CUBLAS_DNRM2 (const int *n, const double *x, const int *incx)
{
    double *devPtrx = 0;
    double retVal = 0.0;
    cublasStatus stat;

    if (*n == 0) return retVal;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dnrm2", CUBLAS_WRAPPER_ERROR_ALLOC);
        return retVal;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dnrm2", CUBLAS_WRAPPER_ERROR_SET);
        return retVal;
    }
    retVal = cublasDnrm2 (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

void CUBLAS_DROT (const int *n, double *x, const int *incx, double *y, 
                  const int *incy, const double *sc, const double *ss)
{
    double *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Drot", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat1 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Drot", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasDrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *ss);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Drot", CUBLAS_WRAPPER_ERROR_GET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_DROTG (double *sa, double *sb, double *sc, double *ss)
{
    cublasDrotg (sa, sb, sc, ss);
}

void CUBLAS_DROTM (const int *n, double *x, const int *incx, double *y, 
                   const int *incy, const double* sparam)
{
    double *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Drotm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Drotm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasDrotm (*n, devPtrx, *incx, devPtry, *incy, sparam);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Drotm", CUBLAS_WRAPPER_ERROR_GET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_DROTMG (double *sd1, double *sd2, double *sx1, const double *sy1,
                    double* sparam)
{
    cublasDrotmg (sd1, sd2, sx1, sy1, sparam);
}

void CUBLAS_DSCAL (const int *n, const double *alpha, double *x, 
                   const int *incx)
{
    double *devPtrx = 0;
    cublasStatus stat;

    if (*n == 0) return;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dscal", CUBLAS_WRAPPER_ERROR_ALLOC);
        return;
    }
    stat = cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dscal", CUBLAS_WRAPPER_ERROR_SET);
        return;
    }
    cublasDscal (*n, *alpha, devPtrx, *incx);
    stat = cublasGetVector (*n, sizeof(x[0]), devPtrx, *incx, x, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dscal", CUBLAS_WRAPPER_ERROR_GET);
        return;
    }
    cublasFree (devPtrx); 
}

void CUBLAS_DSWAP (const int *n, double *x, const int *incx, double *y, 
                   const int *incy)
{
    double *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dswap", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dswap", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    cublasDswap (*n, devPtrx, *incx, devPtry, *incy);
    stat1 = cublasGetVector (*n, sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    stat2 = cublasGetVector (*n, sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dswap", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

#ifdef RETURN_COMPLEX
cuDoubleComplex CUBLAS_ZDOTU ( const int *n, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *y, const int *incy)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    cuDoubleComplex retVal = make_cuDoubleComplex (0.0f, 0.0f);
    if (*n == 0) return retVal;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(*x),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(*y),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdotu", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    stat1 = cublasSetVector (*n, sizeof(*x),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(*y),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdotu", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    retVal = cublasZdotu (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
    return retVal;
}
#else
void CUBLAS_ZDOTU (cuDoubleComplex *retVal, const int *n, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *y, const int *incy)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    *retVal = make_cuDoubleComplex (0.0f, 0.0f);
    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(*x),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(*y),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdotu", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(*x),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(*y),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdotu", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    *retVal = cublasZdotu (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}
#endif

#ifdef RETURN_COMPLEX
cuDoubleComplex CUBLAS_ZDOTC ( const int *n, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *y, const int *incy)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    cuDoubleComplex retVal = make_cuDoubleComplex (0.0f, 0.0f);
    if (*n == 0) return retVal;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(*x),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(*y),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdotc", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    stat1 = cublasSetVector (*n, sizeof(*x),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(*y),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdotc", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return retVal;
    }
    retVal = cublasZdotc (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
    return retVal;
}
#else
void CUBLAS_ZDOTC (cuDoubleComplex *retVal, const int *n, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *y, const int *incy)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2;

    *retVal = make_cuDoubleComplex (0.0f, 0.0f);
    if (*n == 0) return;
    stat1 = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(*x),(void**)&devPtrx);
    stat2 = cublasAlloc (1+(*n-1)*abs(*incy),sizeof(*y),(void**)&devPtry);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdotc", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    stat1 = cublasSetVector (*n, sizeof(*x),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n, sizeof(*y),y,abs(*incy),devPtry,abs(*incy));
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zdotc", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        return;
    }
    *retVal = cublasZdotc (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}
#endif


void CUBLAS_ZSCAL (const int *n, const cuDoubleComplex *alpha, cuDoubleComplex *x, 
                   const int *incx)
{
    cuDoubleComplex *devPtrx = 0;
    cublasStatus stat;
    
    if (*n == 0) return;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(*x),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zscal", CUBLAS_WRAPPER_ERROR_ALLOC);
        return;
    }
    stat = cublasSetVector (*n, sizeof(*x), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zscal", CUBLAS_WRAPPER_ERROR_SET);
        return;
    }
    cublasZscal (*n, *alpha, devPtrx, *incx);
    stat = cublasGetVector (*n, sizeof(*x), devPtrx, *incx, x, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zscal", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx); 
}

void CUBLAS_ZDSCAL (const int *n, const double *alpha, cuDoubleComplex *x, 
                   const int *incx)
{
    cuDoubleComplex *devPtrx = 0;
    cublasStatus stat;
    
    if (*n == 0) return;
    stat = cublasAlloc (1+(*n-1)*abs(*incx),sizeof(*x),(void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zdscal", CUBLAS_WRAPPER_ERROR_ALLOC);
        return;
    }
    stat = cublasSetVector (*n, sizeof(*x), x, *incx, devPtrx, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zdscal", CUBLAS_WRAPPER_ERROR_SET);
        return;
    }
    cublasZdscal (*n, *alpha, devPtrx, *incx);
    stat = cublasGetVector (*n, sizeof(*x), devPtrx, *incx, x, *incx);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zdscal", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx); 
}


/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS2 ----------------------------------*/
/*---------------------------------------------------------------------------*/

void CUBLAS_SGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const float *alpha, 
                   const float *A, const int *lda, const float *x, 
                   const int *incx, const float *beta, float *y, 
                   const int *incy)
{
    float *devPtrx = 0, *devPtry = 0, *devPtrA = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *  A      - REAL             array of DIMENSION ( LDA, n ). 
     * Before entry, the leading ( kl + ku + 1 ) by n part of the
     * array A must contain the matrix of coefficients
     */
    if (toupper(trans[0]) == 'N') {
        stat1 = cublasAlloc(1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*m-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    } else {
        stat1 = cublasAlloc(1+(*m-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    }
    stat3 = cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sgbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    if (toupper(trans[0]) == 'N') {
        stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*m,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    } else {
        stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    }
    stat3 = cublasSetMatrix (imin(*kl+*ku+1,*lda), *n, sizeof(A[0]), A, *lda, 
                             devPtrA, *lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sgbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasSgbmv (trans[0], *m, *n, *kl, *ku, *alpha, devPtrA, *lda, devPtrx, 
                 *incx, *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        stat1=cublasGetVector(*m,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    } else {
        stat1=cublasGetVector(*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    }
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sgbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_DGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const double *alpha, 
                   const double *A, const int *lda, const double *x,
                   const int *incx, const double *beta, double *y, 
                   const int *incy)
{
    double *devPtrx = 0, *devPtry = 0, *devPtrA = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *  A      - REAL             array of DIMENSION ( LDA, n ). 
     * Before entry, the leading ( kl + ku + 1 ) by n part of the
     * array A must contain the matrix of coefficients
     */
    if (toupper(trans[0]) == 'N') {
        stat1 = cublasAlloc(1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*m-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    } else {
        stat1 = cublasAlloc(1+(*m-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    }
    stat3 = cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dgbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    if (toupper(trans[0]) == 'N') {
        stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*m,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    } else {
        stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    }
    stat3 = cublasSetMatrix (imin(*kl+*ku+1,*lda), *n, sizeof(A[0]), A, *lda, 
                             devPtrA, *lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dgbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasDgbmv (trans[0], *m, *n, *kl, *ku, *alpha, devPtrA, *lda, devPtrx, 
                 *incx, *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        stat1=cublasGetVector(*m,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    } else {
        stat1=cublasGetVector(*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    }
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dgbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}                   
void CUBLAS_CGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const cuComplex *alpha, 
                   const cuComplex *A, const int *lda, const cuComplex *x,
                   const int *incx, const cuComplex *beta, cuComplex *y, 
                   const int *incy)
{
    cuComplex *devPtrx = 0, *devPtry = 0, *devPtrA = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *  A      - REAL             array of DIMENSION ( LDA, n ). 
     * Before entry, the leading ( kl + ku + 1 ) by n part of the
     * array A must contain the matrix of coefficients
     */
    if (toupper(trans[0]) == 'N') {
        stat1 = cublasAlloc(1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*m-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    } else {
        stat1 = cublasAlloc(1+(*m-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    }
    stat3 = cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    if (toupper(trans[0]) == 'N') {
        stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*m,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    } else {
        stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    }
    stat3 = cublasSetMatrix (imin(*kl+*ku+1,*lda), *n, sizeof(A[0]), A, *lda, 
                             devPtrA, *lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasCgbmv (trans[0], *m, *n, *kl, *ku, *alpha, devPtrA, *lda, devPtrx, 
                 *incx, *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        stat1=cublasGetVector(*m,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    } else {
        stat1=cublasGetVector(*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    }
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cgbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}                   
void CUBLAS_ZGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const cuDoubleComplex *alpha, 
                   const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *x,
                   const int *incx, const cuDoubleComplex *beta, cuDoubleComplex *y, 
                   const int *incy)
{
    cuDoubleComplex *devPtrx = 0, *devPtry = 0, *devPtrA = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *  A      - REAL             array of DIMENSION ( LDA, n ). 
     * Before entry, the leading ( kl + ku + 1 ) by n part of the
     * array A must contain the matrix of coefficients
     */
    if (toupper(trans[0]) == 'N') {
        stat1 = cublasAlloc(1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*m-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    } else {
        stat1 = cublasAlloc(1+(*m-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    }
    stat3 = cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    if (toupper(trans[0]) == 'N') {
        stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*m,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    } else {
        stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    }
    stat3 = cublasSetMatrix (imin(*kl+*ku+1,*lda), *n, sizeof(A[0]), A, *lda, 
                             devPtrA, *lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasZgbmv (trans[0], *m, *n, *kl, *ku, *alpha, devPtrA, *lda, devPtrx, 
                 *incx, *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        stat1=cublasGetVector(*m,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    } else {
        stat1=cublasGetVector(*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    }
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zgbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}                   

void CUBLAS_SGEMV (const char *trans, const int *m, const int *n,
                   const float *alpha, const float *A, const int *lda,
                   const float *x, const int *incx, const float *beta,
                   float *y, const int *incy)
{
    float *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;
    
    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry, the leading m by n part of the array A must
     *           contain the matrix of coefficients.
     */
    if (toupper(trans[0]) == 'N') {
        stat1=cublasAlloc (1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2=cublasAlloc (1+(*m-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    } else {
        stat1=cublasAlloc (1+(*m-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2=cublasAlloc (1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    }
    stat3=cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sgemv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    if (toupper(trans[0]) == 'N') {
        stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*m,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    } else {
        stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    }       
    stat3=cublasSetMatrix (imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sgemv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasSgemv (trans[0], *m, *n, *alpha, devPtrA, *lda, devPtrx, *incx,
                 *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        stat1=cublasGetVector(*m,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    } else {
        stat1=cublasGetVector(*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    }       
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sgemv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_SGER (const int *m, const int *n, const float *alpha, 
                  const float *x, const int *incx, const float *y,
                  const int *incy, float *A, const int *lda)
{
    float *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ).
     *  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     *  A      - REAL array of DIMENSION ( LDA, n ).
     *           Before entry, the leading m by n part of the array A must
     *           contain the matrix of coefficients. On exit, A is
     */
    stat1=cublasAlloc(1+(*m-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sger", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix(imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sger", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasSger (*m, *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*m,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sger", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_SSBMV (const char *uplo, const int *n, const int *k, 
                   const float *alpha, const float *A, const int *lda,
                   const float *x, const int *incx, const float *beta, 
                   float *y, const int *incy)
{
    float *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;
    
    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
     *           by n part of the array A must contain the upper triangular
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetMatrix (imin(*k+1,*lda), *n, sizeof(A[0]), A, *lda, 
                             devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasSsbmv (uplo[0], *n, *k, *alpha, devPtrA, *lda, devPtrx, *incx, *beta,
                 devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ssbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_DSBMV (const char *uplo, const int *n, const int *k, 
                   const double *alpha, const double *A, const int *lda,
                   const double *x, const int *incx, const double *beta, 
                   double *y, const int *incy)
{
    double *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;
    
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetMatrix (imin(*k+1,*lda), *n, sizeof(A[0]), A, *lda, 
                             devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasDsbmv (uplo[0], *n, *k, *alpha, devPtrA, *lda, devPtrx, *incx, *beta,
                 devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dsbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_CHBMV (const char *uplo, const int *n, const int *k, 
                   const cuComplex *alpha, const cuComplex *A, const int *lda,
                   const cuComplex *x, const int *incx, const cuComplex *beta, 
                   cuComplex *y, const int *incy)
{
    cuComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;
    
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetMatrix (imin(*k+1,*lda), *n, sizeof(A[0]), A, *lda, 
                             devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasChbmv (uplo[0], *n, *k, *alpha, devPtrA, *lda, devPtrx, *incx, *beta,
                 devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Chbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}


void CUBLAS_ZHBMV (const char *uplo, const int *n, const int *k, 
                   const cuDoubleComplex *alpha, const cuDoubleComplex *A, const int *lda,
                   const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *beta, 
                   cuDoubleComplex *y, const int *incy)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;
    
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetMatrix (imin(*k+1,*lda), *n, sizeof(A[0]), A, *lda, 
                             devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasZhbmv (uplo[0], *n, *k, *alpha, devPtrA, *lda, devPtrx, *incx, *beta,
                 devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zhbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_SSPMV (const char *uplo, const int *n, const float *alpha,
                   const float *AP, const float *x, const int *incx, 
                   const float *beta, float *y, const int *incy)
{
    float *devPtrAP = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     *  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     *           Before entry with UPLO = 'U' or 'u', the array AP must
     *           contain the upper triangular part of the symmetric matrix
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sspmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sspmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    cublasSspmv (*uplo, *n, *alpha, devPtrAP, devPtrx, *incx, *beta, devPtry,
                 *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sspmv", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrAP);
}

void CUBLAS_DSPMV (const char *uplo, const int *n, const double *alpha,
                   const double *AP, const double *x, const int *incx, 
                   const double *beta, double *y, const int *incy)
{
    double *devPtrAP = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dspmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dspmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    cublasDspmv (*uplo, *n, *alpha, devPtrAP, devPtrx, *incx, *beta, devPtry,
                 *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dspmv", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrAP);
}

void CUBLAS_CHPMV (const char *uplo, const int *n, const cuComplex *alpha,
                   const cuComplex *AP, const cuComplex *x, const int *incx, 
                   const cuComplex *beta, cuComplex *y, const int *incy)
{
    cuComplex *devPtrAP = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chpmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chpmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    cublasChpmv (*uplo, *n, *alpha, devPtrAP, devPtrx, *incx, *beta, devPtry,
                 *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Chpmv", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrAP);
}

void CUBLAS_ZHPMV (const char *uplo, const int *n, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *AP, const cuDoubleComplex *x, const int *incx, 
                   const cuDoubleComplex *beta, cuDoubleComplex *y, const int *incy)
{
    cuDoubleComplex *devPtrAP = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhpmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhpmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    cublasZhpmv (*uplo, *n, *alpha, devPtrAP, devPtrx, *incx, *beta, devPtry,
                 *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zhpmv", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrAP);
}

void CUBLAS_SSPR (const char *uplo, const int *n, const float *alpha, 
                  const float *x, const int *incx, float *AP)
{
    float *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sspr", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sspr", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasSspr (uplo[0], *n, *alpha, devPtrx, *incx, devPtrAP);
    stat1=cublasGetVector (((*n)*(*n+1))/2,sizeof(AP[0]),devPtrAP,1,AP,1);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sspr", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_DSPR (const char *uplo, const int *n, const double *alpha, 
                  const double *x, const int *incx, double *AP)
{
    double *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dspr", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dspr", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasDspr (uplo[0], *n, *alpha, devPtrx, *incx, devPtrAP);
    stat1=cublasGetVector (((*n)*(*n+1))/2,sizeof(AP[0]),devPtrAP,1,AP,1);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dspr", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_CHPR (const char *uplo, const int *n, const float *alpha, 
                  const cuComplex *x, const int *incx, cuComplex *AP)
{
    cuComplex *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chpr", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chpr", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasChpr (uplo[0], *n, *alpha, devPtrx, *incx, devPtrAP);
    stat1=cublasGetVector (((*n)*(*n+1))/2,sizeof(AP[0]),devPtrAP,1,AP,1);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Chpr", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_ZHPR (const char *uplo, const int *n, const double *alpha, 
                  const cuDoubleComplex *x, const int *incx, cuDoubleComplex *AP)
{
    cuDoubleComplex *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhpr", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhpr", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasZhpr (uplo[0], *n, *alpha, devPtrx, *incx, devPtrAP);
    stat1=cublasGetVector (((*n)*(*n+1))/2,sizeof(AP[0]),devPtrAP,1,AP,1);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zhpr", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_SSPR2 (const char *uplo, const int *n, const float *alpha,
                   const float *x, const int *incx, const float *y, 
                   const int *incy, float *AP)
{
    float *devPtrAP = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     *  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     *           Before entry with  UPLO = 'U' or 'u', the array AP must
     *           contain the upper triangular part of the symmetric matrix
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sspr2", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (1+(*n-1)*abs(*incx),sizeof(x[0]),x,1,devPtrx,1);
    stat2 = cublasSetVector (1+(*n-1)*abs(*incy),sizeof(y[0]),y,1,devPtry,1);
    stat3 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sspr2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    cublasSspr2 (uplo[0], *n, *alpha, devPtrx, *incx, devPtry, *incy,devPtrAP);
    stat1 = cublasGetVector (((*n)*(*n+1))/2,sizeof(AP[0]),devPtrAP,1,AP,1);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sspr2", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrAP);
}

void CUBLAS_DSPR2 (const char *uplo, const int *n, const double *alpha,
                   const double *x, const int *incx, const double *y, 
                   const int *incy, double *AP)
{
    double *devPtrAP = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dspr2", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (1+(*n-1)*abs(*incx),sizeof(x[0]),x,1,devPtrx,1);
    stat2 = cublasSetVector (1+(*n-1)*abs(*incy),sizeof(y[0]),y,1,devPtry,1);
    stat3 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dspr2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    cublasDspr2 (uplo[0], *n, *alpha, devPtrx, *incx, devPtry, *incy,devPtrAP);
    stat1 = cublasGetVector (((*n)*(*n+1))/2,sizeof(AP[0]),devPtrAP,1,AP,1);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dspr2", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrAP);
}

void CUBLAS_CHPR2 (const char *uplo, const int *n, const cuComplex *alpha,
                   const cuComplex *x, const int *incx, const cuComplex *y, 
                   const int *incy, cuComplex *AP)
{
    cuComplex *devPtrAP = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chpr2", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (1+(*n-1)*abs(*incx),sizeof(x[0]),x,1,devPtrx,1);
    stat2 = cublasSetVector (1+(*n-1)*abs(*incy),sizeof(y[0]),y,1,devPtry,1);
    stat3 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chpr2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    cublasChpr2 (uplo[0], *n, *alpha, devPtrx, *incx, devPtry, *incy,devPtrAP);
    stat1 = cublasGetVector (((*n)*(*n+1))/2,sizeof(AP[0]),devPtrAP,1,AP,1);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Chpr2", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrAP);
}

void CUBLAS_ZHPR2 (const char *uplo, const int *n, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *y, 
                   const int *incy, cuDoubleComplex *AP)
{
    cuDoubleComplex *devPtrAP = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhpr2", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (1+(*n-1)*abs(*incx),sizeof(x[0]),x,1,devPtrx,1);
    stat2 = cublasSetVector (1+(*n-1)*abs(*incy),sizeof(y[0]),y,1,devPtry,1);
    stat3 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhpr2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrAP);
        return;
    }
    cublasZhpr2 (uplo[0], *n, *alpha, devPtrx, *incx, devPtry, *incy,devPtrAP);
    stat1 = cublasGetVector (((*n)*(*n+1))/2,sizeof(AP[0]),devPtrAP,1,AP,1);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zhpr2", CUBLAS_WRAPPER_ERROR_GET); 
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrAP);
}


void CUBLAS_SSYMV (const char *uplo, const int *n, const float *alpha,
                   const float *A, const int *lda, const float *x, 
                   const int *incx, const float *beta, float *y, 
                   const int *incy)
{
    float *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     *           Before entry with UPLO = 'L' or 'l', the leading n by n
     *           lower triangular part of the array A must contain the lower
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssymv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
                             *lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssymv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasSsymv (uplo[0], *n, *alpha, devPtrA, *lda, devPtrx, *incx, *beta,
                 devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ssymv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_DSYMV (const char *uplo, const int *n, const double *alpha,
                   const double *A, const int *lda, const double *x, 
                   const int *incx, const double *beta, double *y, 
                   const int *incy)
{
    double *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsymv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
                             *lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsymv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasDsymv (uplo[0], *n, *alpha, devPtrA, *lda, devPtrx, *incx, *beta,
                 devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dsymv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_CHEMV (const char *uplo, const int *n, const cuComplex *alpha,
                   const cuComplex *A, const int *lda, const cuComplex *x, 
                   const int *incx, const cuComplex *beta, cuComplex *y, 
                   const int *incy)
{
    cuComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chemv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
                             *lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chemv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasChemv (uplo[0], *n, *alpha, devPtrA, *lda, devPtrx, *incx, *beta,
                 devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Chemv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_ZHEMV (const char *uplo, const int *n, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *beta, cuDoubleComplex *y, 
                   const int *incy)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhemv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3 = cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
                             *lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhemv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasZhemv (uplo[0], *n, *alpha, devPtrA, *lda, devPtrx, *incx, *beta,
                 devPtry, *incy);
    stat1 = cublasGetVector (*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zhemv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_SSYR (const char *uplo, const int *n, const float *alpha, 
                  const float *x, const int *incx, float *A, const int *lda)
{
    float *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     *           Before entry with UPLO = 'L' or 'l', the leading n by n
     *           lower triangular part of the array A must contain the lower
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda)*(*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssyr", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetMatrix(imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssyr", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasSsyr (uplo[0], *n, *alpha, devPtrx, *incx, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*n,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ssyr", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_SSYR2 (const char *uplo, const int *n, const float *alpha,
                   const float *x, const int *incx, const float *y,
                   const int *incy, float *A, const int *lda)
{
    float *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     *           Before entry with UPLO = 'L' or 'l', the leading n by n
     *           lower triangular part of the array A must contain the lower
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda)*(*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssyr2", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssyr2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasSsyr2 (uplo[0], *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA,
                 *lda);
    stat1=cublasGetMatrix (imin(*n,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ssyr2", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_DSYR2 (const char *uplo, const int *n, const double *alpha,
                   const double *x, const int *incx, const double *y,
                   const int *incy, double *A, const int *lda)
{
    double *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda)*(*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsyr2", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsyr2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasDsyr2 (uplo[0], *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA,
                 *lda);
    stat1=cublasGetMatrix (imin(*n,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dsyr2", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_CHER2 (const char *uplo, const int *n, const cuComplex *alpha,
                   const cuComplex *x, const int *incx, const cuComplex *y,
                   const int *incy, cuComplex *A, const int *lda)
{
    cuComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda)*(*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cher2", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cher2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasCher2 (uplo[0], *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA,
                 *lda);
    stat1=cublasGetMatrix (imin(*n,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cher2", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_ZHER2 (const char *uplo, const int *n, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *y,
                   const int *incy, cuDoubleComplex *A, const int *lda)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda)*(*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zher2", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector (*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zher2", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasZher2 (uplo[0], *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA,
                 *lda);
    stat1=cublasGetMatrix (imin(*n,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zher2", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_STBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const float *A, const int *lda,
                   float *x, const int *incx)
{
    float *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    
    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
     *           by n part of the array A must contain the upper triangular
     *           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 )
     *           by n part of the array A must contain the lower triangular
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix(imin(*k+1,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasStbmv (uplo[0],trans[0],diag[0],*n,*k,devPtrA,*lda,devPtrx,*incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Stbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}


void CUBLAS_DTBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const double *A, const int *lda,
                   double *x, const int *incx)
{
    double *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix(imin(*k+1,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasDtbmv (uplo[0],trans[0],diag[0],*n,*k,devPtrA,*lda,devPtrx,*incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dtbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_CTBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const cuComplex *A, const int *lda,
                   cuComplex *x, const int *incx)
{
    cuComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix(imin(*k+1,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasCtbmv (uplo[0],trans[0],diag[0],*n,*k,devPtrA,*lda,devPtrx,*incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ctbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_ZTBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const cuDoubleComplex *A, const int *lda,
                   cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    
    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
     *           by n part of the array A must contain the upper triangular
     *           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 )
     *           by n part of the array A must contain the lower triangular
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztbmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix(imin(*k+1,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztbmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasZtbmv (uplo[0],trans[0],diag[0],*n,*k,devPtrA,*lda,devPtrx,*incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ztbmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_STBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const float *A, const int *lda,
                   float *x, const int *incx)
{
    float *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
     *           by n part of the array A must contain the upper triangular
     *           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 )
     *           by n part of the array A must contain the lower triangular
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stbsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix(imin(*k+1,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stbsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasStbsv (uplo[0],trans[0],diag[0],*n,*k,devPtrA,*lda,devPtrx,*incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Stbsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_DTBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const double *A, const int *lda,
                   double *x, const int *incx)
{
    double *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtbsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix(imin(*k+1,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtbsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasDtbsv (uplo[0],trans[0],diag[0],*n,*k,devPtrA,*lda,devPtrx,*incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dtbsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_CTBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const cuComplex *A, const int *lda,
                   cuComplex *x, const int *incx)
{
    cuComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctbsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix(imin(*k+1,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctbsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasCtbsv (uplo[0],trans[0],diag[0],*n,*k,devPtrA,*lda,devPtrx,*incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ctbsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_ZTBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const cuDoubleComplex *A, const int *lda,
                   cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztbsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix(imin(*k+1,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztbsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasZtbsv (uplo[0],trans[0],diag[0],*n,*k,devPtrA,*lda,devPtrx,*incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ztbsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_STPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const float *AP, float *x, const int *incx)
{
    float *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stpmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stpmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasStpmv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Stpmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_DTPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const double *AP, double *x, const int *incx)
{
    double *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtpmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtpmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasDtpmv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dtpmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_CTPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const cuComplex *AP, cuComplex *x, const int *incx)
{
    cuComplex *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctpmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctpmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasCtpmv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ctpmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_ZTPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const cuDoubleComplex *AP, cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztpmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctpmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasZtpmv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ztpmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_STPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const float *AP, float *x, const int *incx)
{
    float *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stpsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Stpsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasStpsv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Stpsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_DTPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const double *AP, double *x, const int *incx)
{
    double *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtpsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtpsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasDtpsv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dtpsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_CTPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const cuComplex *AP, cuComplex *x, const int *incx)
{
    cuComplex *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctpsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctpsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasCtpsv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ctpsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_ZTPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const cuDoubleComplex *AP, cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrAP = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztpsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    stat1 = cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetVector (((*n)*(*n+1))/2,sizeof(AP[0]),AP,1,devPtrAP,1);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztpsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrAP);
        return;
    }
    cublasZtpsv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    stat1 = cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ztpsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrAP);
}

void CUBLAS_STRMV (const char *uplo, const char *trans,
                            const char *diag, const int *n, const float *A,
                            const int *lda, float *x, const int *incx)
{
    float *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    
    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Strmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Strmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasStrmv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    stat1=cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Strmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

void CUBLAS_STRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const float *A, const int *lda, float *x, 
                   const int *incx)
{
    float *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     *           Before entry with UPLO = 'L' or 'l', the leading n by n
     *           lower triangular part of the array A must contain the lower
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Strsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Strsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasStrsv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    stat1=cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Strsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_DGEMV (const char *trans, const int *m, const int *n,
                   const double *alpha, const double *A, const int *lda,
                   const double *x, const int *incx, const double *beta,
                   double *y, const int *incy)
{
    double *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;
    
    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry, the leading m by n part of the array A must
     *           contain the matrix of coefficients.
     */
    if (toupper(trans[0]) == 'N') {
        stat1 = cublasAlloc(1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*m-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    } else {
        stat1 = cublasAlloc(1+(*m-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    }
    stat3 = cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dgemv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    if (toupper(trans[0]) == 'N') {
        stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*m,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    } else {
        stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    }       
    stat3=cublasSetMatrix (imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat3=cublasSetMatrix (imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dgemv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasDgemv (trans[0], *m, *n, *alpha, devPtrA, *lda, devPtrx, *incx,
                 *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        stat1=cublasGetVector(*m,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    } else {
        stat1=cublasGetVector(*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    }       
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dgemv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_ZGEMV (const char *trans, const int *m, const int *n,
                   const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                   const int *lda, const cuDoubleComplex *x, const int *incx,
                   const cuDoubleComplex *beta, cuDoubleComplex *y,
                   const int *incy)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;
    
    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry, the leading m by n part of the array A must
     *           contain the matrix of coefficients.
     */
    if (toupper(trans[0]) == 'N') {
        stat1 = cublasAlloc(1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*m-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    } else {
        stat1 = cublasAlloc(1+(*m-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    }
    stat3 = cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgemv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    if (toupper(trans[0]) == 'N') {
        stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*m,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    } else {
        stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    }       
    stat3=cublasSetMatrix (imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat3=cublasSetMatrix (imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgemv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasZgemv (trans[0], *m, *n, *alpha, devPtrA, *lda, devPtrx, *incx,
                 *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        stat1=cublasGetVector(*m,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    } else {
        stat1=cublasGetVector(*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    }       
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zgemv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_DGER (const int *m, const int *n, const double *alpha, 
                  const double *x, const int *incx, const double *y,
                  const int *incy, double *A, const int *lda)
{
    double *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ).
     *
     *  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     *
     * A       - REAL array of DIMENSION ( LDA, n ).
     *           Before entry, the leading m by n part of the array A must
     *           contain the matrix of coefficients. On exit, A is
     */
    stat1=cublasAlloc(1+(*m-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dger", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix(imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dger", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasDger (*m, *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*m,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dger", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_CGERU (const int *m, const int *n, const cuComplex *alpha, 
                  const cuComplex *x, const int *incx, const cuComplex *y,
                  const int *incy, cuComplex *A, const int *lda)
{
    cuComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    stat1=cublasAlloc(1+(*m-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgeru", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix(imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgeru", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasCgeru (*m, *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*m,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cgeru", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_CGERC (const int *m, const int *n, const cuComplex *alpha, 
                  const cuComplex *x, const int *incx, const cuComplex *y,
                  const int *incy, cuComplex *A, const int *lda)
{
    cuComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    stat1=cublasAlloc(1+(*m-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgerc", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix(imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgerc", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasCgerc (*m, *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*m,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cgerc", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_ZGERU (const int *m, const int *n, const cuDoubleComplex *alpha, 
                  const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *y,
                  const int *incy, cuDoubleComplex *A, const int *lda)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    stat1=cublasAlloc(1+(*m-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgeru", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix(imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgeru", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasZgeru (*m, *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*m,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zgeru", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_ZGERC (const int *m, const int *n, const cuDoubleComplex *alpha, 
                  const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *y,
                  const int *incy, cuDoubleComplex *A, const int *lda)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    stat1=cublasAlloc(1+(*m-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc(1+(*n-1)*abs(*incy),sizeof(devPtry[0]),(void**)&devPtry);
    stat3=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgerc", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    stat3=cublasSetMatrix(imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgerc", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasZgerc (*m, *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*m,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zgerc" , CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtry);
    cublasFree (devPtrA);
}

void CUBLAS_DSYR (const char *uplo, const int *n, const double *alpha, 
                  const double *x, const int *incx, double *A, const int *lda)
{
    double *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     *           Before entry with UPLO = 'L' or 'l', the leading n by n
     *           lower triangular part of the array A must contain the lower
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsyr", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetMatrix(imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsyr", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasDsyr (uplo[0], *n, *alpha, devPtrx, *incx, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*n,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dsyr", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_CHER (const char *uplo, const int *n, const float *alpha, 
                  const cuComplex *x, const int *incx, cuComplex *A, const int *lda)
{
    cuComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cher", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetMatrix(imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cher", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasCher (uplo[0], *n, *alpha, devPtrx, *incx, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*n,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cher", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_ZHER (const char *uplo, const int *n, const double *alpha, 
                  const cuDoubleComplex *x, const int *incx, cuDoubleComplex *A, const int *lda)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zher", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1 = cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2 = cublasSetMatrix(imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zher", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasZher (uplo[0], *n, *alpha, devPtrx, *incx, devPtrA, *lda);
    stat1 = cublasGetMatrix(imin(*n,*lda),*n,sizeof(A[0]),devPtrA,*lda,A,*lda);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zher", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_DTRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const double *A, const int *lda, double *x, 
                   const int *incx)
{
    double *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     *           Before entry with UPLO = 'L' or 'l', the leading n by n
     *           lower triangular part of the array A must contain the lower
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtrsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtrsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasDtrsv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    stat1=cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dtrsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_CTRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const cuComplex *A, const int *lda, cuComplex *x, 
                   const int *incx)
{
    cuComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctrsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctrsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasCtrsv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    stat1=cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ctrsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_ZTRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const cuDoubleComplex *A, const int *lda, 
                   cuDoubleComplex *x, const int *incx)
{
    cuDoubleComplex *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztrsv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztrsv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasZtrsv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    stat1=cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ztrsv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrx);
    cublasFree (devPtrA);
}

void CUBLAS_DTRMV (const char *uplo, const char *trans,
                            const char *diag, const int *n, const double *A,
                            const int *lda, double *x, const int *incx)
{
    double *devPtrA = 0, *devPtrx = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;
    
    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */
    stat1=cublasAlloc(1+(*n-1)*abs(*incx),sizeof(devPtrx[0]),(void**)&devPtrx);
    stat2=cublasAlloc((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtrmv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    stat1=cublasSetVector (*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
    stat2=cublasSetMatrix (imin(*n,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtrmv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtrA);
        return;
    }
    cublasDtrmv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    stat1=cublasGetVector (*n,sizeof(x[0]),devPtrx,abs(*incx),x,abs(*incx));
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dtrmv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS3 ----------------------------------*/
/*---------------------------------------------------------------------------*/

void CUBLAS_SGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const float *alpha,
                   const float *A, const int *lda, const float *B,
                   const int *ldb, const float *beta, float *C, const int *ldc)
{
    int ka, kb;
    float *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return; 

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
     *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by m  part of the array  A  must contain  the
     *           matrix A.
     *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
     *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  n by k  part of the array  B  must contain  the
     *           matrix B.
     *  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     *           On exit, the array  C  is overwritten by the  m by n  matrix
     */
    ka = (toupper(transa[0]) == 'N') ? *k : *m;
    kb = (toupper(transb[0]) == 'N') ? *n : *k;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]), (void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sgemm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(transa[0]) == 'N') {
        stat1=cublasSetMatrix(imin(*m,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
        stat1=cublasSetMatrix(imin(*k,*lda),*m,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    if (toupper(transb[0]) == 'N') {
        stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
        stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Sgemm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasSgemm (transa[0], transb[0], *m, *n, *k, *alpha, devPtrA, *lda, 
                 devPtrB, *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Sgemm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_SSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const float *alpha, const float *A, 
                   const int *lda, const float *B, const int *ldb, 
                   const float *beta, float *C, const int *ldc)
{
    int ka;
    float *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;
    
    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
     *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry, the leading  m by n part of the array  B  must
     *           contain the matrix B.
     *  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     */
    ka = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc ((*lda) * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssymm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    stat1 = cublasSetMatrix(imin(ka,*lda),ka,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    stat3 = cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssymm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasSsymm (side[0], uplo[0], *m, *n, *alpha, devPtrA, *lda, devPtrB,
                 *ldb, *beta, devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ssymm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_SSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const float *alpha, const float *A, 
                    const int *lda, const float *B, const int *ldb, 
                    const float *beta, float *C, const int *ldc)
{
    int ka, kb;
    float *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by n  part of the array  A  must contain  the
     *           matrix A.
     *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  k by n  part of the array  B  must contain  the
     *           matrix B.
     * C       - single precision array of dimensions (ldc, n). If uplo == 'U' 
     *           or 'u', the leading n x n triangular part of the array C must 
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    kb = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]),(void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssyr2k", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3 = cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssyr2k", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasSsyr2k (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, devPtrB, 
                  *ldb, *beta, devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ssyr2k", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_SSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const float *alpha, const float *A, 
                   const int *lda, const float *beta, float *C, const int *ldc)
{
    int ka;
    float *devPtrA = 0, *devPtrC = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /* A      single precision array of dimensions (lda, ka), where ka is k 
     *        when trans == 'N' or 'n', and is n otherwise. When trans == 'N' 
     *        or 'n', the leading n x k part of array A must contain the matrix
     *        A, otherwise the leading k x n part of the array must contain the
     *        matrix A.
     * C      single precision array of dimensions (ldc, n). If uplo='U'or'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly 
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc ((*ldc)*(*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssyrk", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    stat2 = cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ssyrk", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    cublasSsyrk (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, *beta,
                 devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ssyrk", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrC);
}

void CUBLAS_STRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const float *alpha, const float *A, const int *lda,
                   float *B, const int *ldb)
{
    int k;
    float *devPtrA = 0, *devPtrB = 0;
    cublasStatus stat1, stat2;
    
    if ((*m == 0) || (*n == 0)) return;

    /* A      single precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     * B      single precision array of dimensions (ldb, n). On entry, the 
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (*ldb * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Strmm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    stat1 = cublasSetMatrix(imin(k,*lda),k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Strmm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    cublasStrmm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    stat1 = cublasGetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),devPtrB,*ldb,B,*ldb);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Strmm", CUBLAS_WRAPPER_ERROR_GET);
    }        
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_STRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n, 
                   const float *alpha, const float *A, const int *lda,
                   float *B, const int *ldb)
{
    float *devPtrA = 0, *devPtrB = 0;
    int k;
    cublasStatus stat1, stat2;

    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, k ), where k is m
     *           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
     *           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
     *           upper triangular part of the array  A must contain the upper
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry,  the leading  m by n part of the array  B must
     *           contain  the  right-hand  side  matrix  B,  and  on exit  is
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (*ldb * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Strsm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    stat1 = cublasSetMatrix(imin(k,*lda),k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Strsm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    cublasStrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    stat1 = cublasGetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),devPtrB,*ldb,B,*ldb);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Strsm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_CGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const cuComplex *alpha,
                   const cuComplex *A, const int *lda, const cuComplex *B,
                   const int *ldb, const cuComplex *beta, cuComplex *C, 
                   const int *ldc)
{
    int ka, kb;
    cuComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return; 

    /*  A      - COMPLEX          array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
     *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by m  part of the array  A  must contain  the
     *           matrix A.
     *  B      - COMPLEX          array of DIMENSION ( LDB, kb ), where kb is
     *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
     *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  n by k  part of the array  B  must contain  the
     *           matrix B.
     *  C      - COMPLEX          array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     *           On exit, the array  C  is overwritten by the  m by n  matrix
     */
    ka = (toupper(transa[0]) == 'N') ? *k : *m;
    kb = (toupper(transb[0]) == 'N') ? *n : *k;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]),(void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgemm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(transa[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*m,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*m,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    if (toupper(transb[0]) == 'N') {
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgemm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasCgemm (transa[0], transb[0], *m, *n, *k, *alpha, devPtrA, *lda, 
                 devPtrB, *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cgemm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_CSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const cuComplex *alpha, const cuComplex *A, 
                   const int *lda, const cuComplex *B, const int *ldb, 
                   const cuComplex *beta, cuComplex *C, const int *ldc)
{
    int ka;
    cuComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;
    
    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
     *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry, the leading  m by n part of the array  B  must
     *           contain the matrix B.
     *  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     */
    ka = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc ((*lda) * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csymm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    stat1 = cublasSetMatrix(imin(ka,*lda),ka,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    stat3 = cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csymm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasCsymm (side[0], uplo[0], *m, *n, *alpha, devPtrA, *lda, devPtrB,
                 *ldb, *beta, devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Csymm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}


void CUBLAS_CHEMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const cuComplex *alpha, const cuComplex *A, 
                   const int *lda, const cuComplex *B, const int *ldb, 
                   const cuComplex *beta, cuComplex *C, const int *ldc)
{
    int ka;
    cuComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;
    
    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
     *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry, the leading  m by n part of the array  B  must
     *           contain the matrix B.
     *  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     */
    ka = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc ((*lda) * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chemm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    stat1 = cublasSetMatrix(imin(ka,*lda),ka,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    stat3 = cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Chemm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasChemm (side[0], uplo[0], *m, *n, *alpha, devPtrA, *lda, devPtrB,
                 *ldb, *beta, devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Chemm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}


void CUBLAS_CTRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const cuComplex *alpha, const cuComplex *A, const int *lda,
                   cuComplex *B, const int *ldb)
{
    int k;
    cuComplex *devPtrA = 0, *devPtrB = 0;
    cublasStatus stat1, stat2;

    if ((*m == 0) || (*n == 0)) return;

    /* A      single precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     * B      single precision array of dimensions (ldb, n). On entry, the 
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (*ldb * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctrmm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    stat1 = cublasSetMatrix(imin(k,*lda),k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctrmm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    cublasCtrmm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    stat1 = cublasGetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),devPtrB,*ldb,B,*ldb);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ctrmm", CUBLAS_WRAPPER_ERROR_GET);
    }        
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_CTRSM ( const char *side, const char *uplo, const char *transa,
                    const char *diag, const int *m, const int *n,
                    const cuComplex *alpha, const cuComplex *A, const int *lda,
                    cuComplex *B, const int *ldb)
{
    cuComplex *devPtrA = 0, *devPtrB = 0;
    int k;
    cublasStatus stat1, stat2;

    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, k ), where k is m
     *           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
     *           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
     *           upper triangular part of the array  A must contain the upper
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry,  the leading  m by n part of the array  B must
     *           contain  the  right-hand  side  matrix  B,  and  on exit  is
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (*ldb * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctrsm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    stat1 = cublasSetMatrix(imin(k,*lda),k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ctrsm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    cublasCtrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    stat1=cublasGetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),devPtrB,*ldb,B,*ldb);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ctrsm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_CHERK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const float *alpha,
                   const cuComplex *A, const int *lda,
                   const float *beta, cuComplex *C, 
                   const int *ldc)
{
    int ka;
    cuComplex *devPtrA = 0, *devPtrC = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /* A      double complex precision array of dimensions (lda, ka), where ka is k 
     *        when trans == 'N' or 'n', and is n otherwise. When trans == 'N' 
     *        or 'n', the leading n x k part of array A must contain the matrix
     *        A, otherwise the leading k x n part of the array must contain the
     *        matrix A.
     * C      double complex precision array of dimensions (ldc, n). If uplo='U'or'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly 
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc(imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc(imax(1,*ldc*(*n)),sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cherk", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    stat2 = cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cherk", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    cublasCherk (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, *beta,
                 devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cherk", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrC);
}

void CUBLAS_CHER2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const cuComplex *alpha, const cuComplex *A, 
                    const int *lda, const cuComplex *B, const int *ldb,
                    const float *beta, cuComplex *C, const int *ldc)
{
    int ka, kb;
    cuComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by n  part of the array  A  must contain  the
     *           matrix A.
     *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  k by n  part of the array  B  must contain  the
     *           matrix B.
     *  C      - single precision array of dimensions (ldc, n). If uplo == 'U' 
     *           or  'u', the leading n x n triangular part of the array C must
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    kb = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]),(void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cher2k", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csyr2k", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasCher2k (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, devPtrB, 
                  *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cher2k", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_CSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const cuComplex *alpha,
                   const cuComplex *A, const int *lda,
                   const cuComplex *beta, cuComplex *C, 
                   const int *ldc)
{
    int ka;
    cuComplex *devPtrA = 0, *devPtrC = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /* A      double complex precision array of dimensions (lda, ka), where ka is k 
     *        when trans == 'N' or 'n', and is n otherwise. When trans == 'N' 
     *        or 'n', the leading n x k part of array A must contain the matrix
     *        A, otherwise the leading k x n part of the array must contain the
     *        matrix A.
     * C      double complex precision array of dimensions (ldc, n). If uplo='U'or'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly 
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc(imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc(imax(1,*ldc*(*n)),sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csyrk", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    stat2 = cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csyrk", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    cublasCsyrk (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, *beta,
                 devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Csyrk", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrC);
}

void CUBLAS_CSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const cuComplex *alpha, const cuComplex *A, 
                    const int *lda, const cuComplex *B, const int *ldb,
                    const cuComplex *beta, cuComplex *C, const int *ldc)
{
    int ka, kb;
    cuComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by n  part of the array  A  must contain  the
     *           matrix A.
     *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  k by n  part of the array  B  must contain  the
     *           matrix B.
     *  C      - single precision array of dimensions (ldc, n). If uplo == 'U' 
     *           or  'u', the leading n x n triangular part of the array C must
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    kb = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]),(void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csyr2k", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Csyr2k", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasCsyr2k (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, devPtrB, 
                  *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Csyr2k", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_DGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const double *alpha,
                   const double *A, const int *lda, const double *B,
                   const int *ldb, const double *beta, double *C,
                   const int *ldc)
{
    int ka, kb;
    double *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
     *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by m  part of the array  A  must contain  the
     *           matrix A.
     *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
     *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  n by k  part of the array  B  must contain  the
     *           matrix B. 
     *  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     *           On exit, the array  C  is overwritten by the  m by n  matrix
     */
    ka = (toupper(transa[0]) == 'N') ? *k : *m;
    kb = (toupper(transb[0]) == 'N') ? *n : *k;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]), (void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dgemm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(transa[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*m,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*m,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    if (toupper(transb[0]) == 'N') {
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dgemm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasDgemm (transa[0], transb[0], *m, *n, *k, *alpha, devPtrA, *lda, 
                 devPtrB, *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dgemm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_DSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const double *alpha, const double *A, 
                   const int *lda, const double *B, const int *ldb, 
                   const double *beta, double *C, const int *ldc)
{
    int ka;
    double *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;
    
    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
     *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry, the leading  m by n part of the array  B  must
     *           contain the matrix B.
     *  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     */
    ka = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc ((*lda) * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsymm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    stat1 = cublasSetMatrix(imin(ka,*lda),ka,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    stat3 = cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsymm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasDsymm (side[0], uplo[0], *m, *n, *alpha, devPtrA, *lda, devPtrB,
                 *ldb, *beta, devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dsymm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_DSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const double *alpha, const double *A, 
                    const int *lda, const double *B, const int *ldb, 
                    const double *beta, double *C, const int *ldc)
{
    int ka, kb;
    double *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by n  part of the array  A  must contain  the
     *           matrix A.
     *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  k by n  part of the array  B  must contain  the
     *           matrix B.
     *  C      - single precision array of dimensions (ldc, n). If uplo == 'U' 
     *           or  'u', the leading n x n triangular part of the array C must
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    kb = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]),(void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsyr2k", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsyr2k", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasDsyr2k (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, devPtrB, 
                  *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dsyr2k", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_DSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const double *alpha, const double *A, 
                   const int *lda, const double *beta, double *C, 
                   const int *ldc)
{
    int ka;
    double *devPtrA = 0, *devPtrC = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /* A      double precision array of dimensions (lda, ka), where ka is k 
     *        when trans == 'N' or 'n', and is n otherwise. When trans == 'N' 
     *        or 'n', the leading n x k part of array A must contain the matrix
     *        A, otherwise the leading k x n part of the array must contain the
     *        matrix A.
     * C      double precision array of dimensions (ldc, n). If uplo='U'or'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly 
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc(imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc(imax(1,*ldc*(*n)),sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsyrk", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    stat2 = cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dsyrk", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    cublasDsyrk (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, *beta,
                 devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dsyrk", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrC);
}

void CUBLAS_ZSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda,
                   const cuDoubleComplex *beta, cuDoubleComplex *C, 
                   const int *ldc)
{
    int ka;
    cuDoubleComplex *devPtrA = 0, *devPtrC = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /* A      double complex precision array of dimensions (lda, ka), where ka is k 
     *        when trans == 'N' or 'n', and is n otherwise. When trans == 'N' 
     *        or 'n', the leading n x k part of array A must contain the matrix
     *        A, otherwise the leading k x n part of the array must contain the
     *        matrix A.
     * C      double complex precision array of dimensions (ldc, n). If uplo='U'or'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly 
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc(imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc(imax(1,*ldc*(*n)),sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zsyrk", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    stat2 = cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zsyrk", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    cublasZsyrk (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, *beta,
                 devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zsyrk", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrC);
}

void CUBLAS_ZSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, 
                    const int *lda, const cuDoubleComplex *B, const int *ldb, 
                    const cuDoubleComplex *beta, cuDoubleComplex *C, const int *ldc)
{
    int ka, kb;
    cuDoubleComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by n  part of the array  A  must contain  the
     *           matrix A.
     *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  k by n  part of the array  B  must contain  the
     *           matrix B.
     *  C      - single precision array of dimensions (ldc, n). If uplo == 'U' 
     *           or  'u', the leading n x n triangular part of the array C must
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    kb = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]),(void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zsyr2k", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zsyr2k", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasZsyr2k (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, devPtrB, 
                  *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zsyr2k", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}


void CUBLAS_DTRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const double *alpha, const double *A, const int *lda,
                   double *B, const int *ldb)
{
    int k;
    double *devPtrA = 0, *devPtrB = 0;
    cublasStatus stat1, stat2;

    if ((*m == 0) || (*n == 0)) return;

    /* A      single precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     * B      single precision array of dimensions (ldb, n). On entry, the 
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (*ldb * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtrmm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    stat1 = cublasSetMatrix(imin(k,*lda),k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtrmm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    cublasDtrmm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    stat1 = cublasGetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),devPtrB,*ldb,B,*ldb);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dtrmm", CUBLAS_WRAPPER_ERROR_GET);
    }        
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_DTRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n, 
                   const double *alpha, const double *A, const int *lda,
                   double *B, const int *ldb)
{
    double *devPtrA = 0, *devPtrB = 0;
    int k;
    cublasStatus stat1, stat2;

    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, k ), where k is m
     *           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
     *           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
     *           upper triangular part of the array  A must contain the upper
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry,  the leading  m by n part of the array  B must
     *           contain  the  right-hand  side  matrix  B,  and  on exit  is
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (*ldb * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtrsm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    stat1 = cublasSetMatrix(imin(k,*lda),k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dtrsm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    cublasDtrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    stat1=cublasGetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),devPtrB,*ldb,B,*ldb);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Dtrsm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_ZTRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n, 
                   const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda,
                   cuDoubleComplex *B, const int *ldb)
{
    cuDoubleComplex *devPtrA = 0, *devPtrB = 0;
    int k;
    cublasStatus stat1, stat2;

    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, k ), where k is m
     *           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
     *           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
     *           upper triangular part of the array  A must contain the upper
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry,  the leading  m by n part of the array  B must
     *           contain  the  right-hand  side  matrix  B,  and  on exit  is
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (*ldb * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztrsm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    stat1 = cublasSetMatrix(imin(k,*lda),k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztrsm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    cublasZtrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    stat1=cublasGetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),devPtrB,*ldb,B,*ldb);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ztrsm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_ZGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda, 
                   const cuDoubleComplex *B, const int *ldb, 
                   const cuDoubleComplex *beta, cuDoubleComplex *C, 
                   const int *ldc)
{
    int ka, kb;
    cuDoubleComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if ((*m == 0) || (*n == 0)) return; 
    
    /*  A      - COMPLEX          array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
     *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by m  part of the array  A  must contain  the
     *           matrix A.
     *  B      - COMPLEX          array of DIMENSION ( LDB, kb ), where kb is
     *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
     *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  n by k  part of the array  B  must contain  the
     *           matrix B.
     *  C      - COMPLEX          array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     *           On exit, the array  C  is overwritten by the  m by n  matrix
     */
    ka = (toupper(transa[0]) == 'N') ? *k : *m;
    kb = (toupper(transb[0]) == 'N') ? *n : *k;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]),(void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgemm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(transa[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*m,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*m,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    if (toupper(transb[0]) == 'N') {
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zgemm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasZgemm (transa[0], transb[0], *m, *n, *k, *alpha, devPtrA, *lda, 
                 devPtrB, *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zgemm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}


void CUBLAS_CGEMV (const char *trans, const int *m, const int *n,
                   const cuComplex *alpha, const cuComplex *A,
                   const int *lda, const cuComplex *x, const int *incx,
                   const cuComplex *beta, cuComplex *y,
                   const int *incy)
{
    cuComplex *devPtrA = 0, *devPtrx = 0, *devPtry = 0;
    cublasStatus stat1, stat2, stat3;
    
    if ((*m == 0) || (*n == 0)) return;

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry, the leading m by n part of the array A must
     *           contain the matrix of coefficients.
     */
    if (toupper(trans[0]) == 'N') {
        stat1 = cublasAlloc(1+(*n-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*m-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    } else {
        stat1 = cublasAlloc(1+(*m-1)*abs(*incx),sizeof(x[0]),(void**)&devPtrx);
        stat2 = cublasAlloc(1+(*n-1)*abs(*incy),sizeof(y[0]),(void**)&devPtry);
    }
    stat3 = cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgemv", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    if (toupper(trans[0]) == 'N') {
        stat1=cublasSetVector(*n,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*m,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    } else {
        stat1=cublasSetVector(*m,sizeof(x[0]),x,abs(*incx),devPtrx,abs(*incx));
        stat2=cublasSetVector(*n,sizeof(y[0]),y,abs(*incy),devPtry,abs(*incy));
    }       
    stat3=cublasSetMatrix (imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat3=cublasSetMatrix (imin(*m,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Cgemv", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrx);
        cublasFree (devPtry);
        cublasFree (devPtrA);
        return;
    }
    cublasCgemv (trans[0], *m, *n, *alpha, devPtrA, *lda, devPtrx, *incx,
                 *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        stat1=cublasGetVector(*m,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    } else {
        stat1=cublasGetVector(*n,sizeof(y[0]),devPtry,abs(*incy),y,abs(*incy));
    }       
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Cgemv", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}


void CUBLAS_ZSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, 
                   const int *lda, const cuDoubleComplex *B, const int *ldb, 
                   const cuDoubleComplex *beta, cuDoubleComplex *C, const int *ldc)
{
    int ka;
    cuDoubleComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;
    
    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
     *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry, the leading  m by n part of the array  B  must
     *           contain the matrix B.
     *  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     */
    ka = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc ((*lda) * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zsymm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    stat1 = cublasSetMatrix(imin(ka,*lda),ka,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    stat3 = cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zsymm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasZsymm (side[0], uplo[0], *m, *n, *alpha, devPtrA, *lda, devPtrB,
                 *ldb, *beta, devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zsymm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_ZHEMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, 
                   const int *lda, const cuDoubleComplex *B, const int *ldb, 
                   const cuDoubleComplex *beta, cuDoubleComplex *C, const int *ldc)
{
    int ka;
    cuDoubleComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;
    
    if ((*m == 0) || (*n == 0)) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
     *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry, the leading  m by n part of the array  B  must
     *           contain the matrix B.
     *  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     */
    ka = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc ((*lda) * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhemm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    stat1 = cublasSetMatrix(imin(ka,*lda),ka,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    stat3 = cublasSetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zhemm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasZhemm (side[0], uplo[0], *m, *n, *alpha, devPtrA, *lda, devPtrB,
                 *ldb, *beta, devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*m,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zhemm", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_ZTRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const cuDoubleComplex *alpha, const cuDoubleComplex *A, const int *lda,
                   cuDoubleComplex *B, const int *ldb)
{
    int k;
    cuDoubleComplex *devPtrA = 0, *devPtrB = 0;
    cublasStatus stat1, stat2;

    if ((*m == 0) || (*n == 0)) return;

    /* A      double precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     * B      single precision array of dimensions (ldb, n). On entry, the 
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    stat1 = cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    stat2 = cublasAlloc (*ldb * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztrmm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    stat1 = cublasSetMatrix(imin(k,*lda),k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    stat2 = cublasSetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Ztrmm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        return;
    }
    cublasZtrmm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    stat1 = cublasGetMatrix(imin(*m,*ldb),*n,sizeof(B[0]),devPtrB,*ldb,B,*ldb);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Ztrmm", CUBLAS_WRAPPER_ERROR_GET);
    }        
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_ZHERK (const char *uplo, const char *trans, const int *n,
                   const int *k, const double *alpha,
                   const cuDoubleComplex *A, const int *lda,
                   const double *beta, cuDoubleComplex *C,
                   const int *ldc)
{
    int ka;
    cuDoubleComplex *devPtrA = 0, *devPtrC = 0;
    cublasStatus stat1, stat2;

    if (*n == 0) return;

    /* A      double complex precision array of dimensions (lda, ka), where ka is k 
     *        when trans == 'N' or 'n', and is n otherwise. When trans == 'N' 
     *        or 'n', the leading n x k part of array A must contain the matrix
     *        A, otherwise the leading k x n part of the array must contain the
     *        matrix A.
     * C      double complex precision array of dimensions (ldc, n). If uplo='U'or'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly 
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc(imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc(imax(1,*ldc*(*n)),sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zherk", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
    }
    stat2 = cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zsyrk", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrC);
        return;
    }
    cublasZherk (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, *beta,
                 devPtrC, *ldc);
    stat1 = cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zherk", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrC);
}

void CUBLAS_ZHER2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, 
                    const int *lda, const cuDoubleComplex *B, const int *ldb, 
                    const double *beta, cuDoubleComplex *C, const int *ldc)
{
    int ka, kb;
    cuDoubleComplex *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    cublasStatus stat1, stat2, stat3;

    if (*n == 0) return;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by n  part of the array  A  must contain  the
     *           matrix A.
     *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  k by n  part of the array  B  must contain  the
     *           matrix B.
     *  C      - single precision array of dimensions (ldc, n). If uplo == 'U' 
     *           or  'u', the leading n x n triangular part of the array C must
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    kb = (toupper(trans[0]) == 'N') ? *k : *n;
    stat1 = cublasAlloc (imax(1,*lda*ka),sizeof(devPtrA[0]),(void**)&devPtrA);
    stat2 = cublasAlloc (imax(1,*ldb*kb),sizeof(devPtrB[0]),(void**)&devPtrB);
    stat3 = cublasAlloc ((*ldc) * (*n),  sizeof(devPtrC[0]),(void**)&devPtrC);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zher2k", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    if (toupper(trans[0]) == 'N') {
      stat1=cublasSetMatrix(imin(*n,*lda),*k,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*n,*ldb),*k,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    } else {
      stat1=cublasSetMatrix(imin(*k,*lda),*n,sizeof(A[0]),A,*lda,devPtrA,*lda);
      stat2=cublasSetMatrix(imin(*k,*ldb),*n,sizeof(B[0]),B,*ldb,devPtrB,*ldb);
    }
    stat3=cublasSetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),C,*ldc,devPtrC,*ldc);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) || 
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Zher2k", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtrA);
        cublasFree (devPtrB);
        cublasFree (devPtrC);
        return;
    }
    cublasZher2k (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, devPtrB, 
                  *ldb, *beta, devPtrC, *ldc);
    stat1=cublasGetMatrix(imin(*n,*ldc),*n,sizeof(C[0]),devPtrC,*ldc,C,*ldc);
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        wrapperError ("Zher2k", CUBLAS_WRAPPER_ERROR_GET);
    }
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}
/*
void CUDA_DIAG (double* o, const double* x,double* hold,
		const double* E, const double* idegen,
		const double* vec, const double* co,
		const double* V2, const int* nbasis)
{

  int ka, kb;
  cublasStatus_t stat1, stat2, stat3;
  
  double* devPtr_o=0;
  double* devPtr_x=0;
  double* devPtr_hold=0;

    if (nbasis == 0) return;
  
    int dim = *nbasis;

    printf("nbasis: %i", *nbasis);
    printf("dim: %i", dim);    

    stat1=cublasAlloc (imax(1, dim*dim), sizeof(devPtr_o), (void**)&devPtr_o);

      if(stat1 != CUBLAS_STATUS_SUCCESS)
	{
	  printf("dgemm1\n");
	wrapperError("Dgemm", CUBLAS_WRAPPER_ERROR_ALLOC);
	}        

    stat2=cublasAlloc (imax(1, dim*dim), sizeof(devPtr_x), (void**)&devPtr_x);

      if(stat2 != CUBLAS_STATUS_SUCCESS)
	{
	  printf("dgemm2\n");
	wrapperError("Dgemm", CUBLAS_WRAPPER_ERROR_SET);
	}
      
    stat3=cublasAlloc (dim*dim, sizeof(devPtr_hold), (void**)&devPtr_hold);

      if(stat3 != CUBLAS_STATUS_SUCCESS)
	{
	  printf("dgemm3\n");
	wrapperError("Dgemm", CUBLAS_WRAPPER_ERROR_SET);
	}            



    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dgemm", CUBLAS_WRAPPER_ERROR_ALLOC);
        cublasFree (devPtr_o);
        cublasFree (devPtr_x);
	cublasFree (devPtr_hold);
        return;
    }    
      stat1=cublasSetMatrix(dim,dim,sizeof(devPtr_o[0]),o,dim,devPtr_o,dim);

      stat2=cublasSetMatrix(dim,dim,sizeof(devPtr_x[0]),x,dim,devPtr_x,dim);
      stat3=cublasSetMatrix(dim,dim,sizeof(devPtr_hold[0]),hold,dim,devPtr_hold,dim);
    if ((stat1 != CUBLAS_STATUS_SUCCESS) ||
        (stat2 != CUBLAS_STATUS_SUCCESS) ||
        (stat3 != CUBLAS_STATUS_SUCCESS)) {
        wrapperError ("Dgemm", CUBLAS_WRAPPER_ERROR_SET);
        cublasFree (devPtr_o);
        cublasFree (devPtr_x);
        cublasFree (devPtr_hold);
        return;
    }


    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    
  // hold = o * x

      cublasDgemm_v2 (cublasH, 'n','n', dim, dim, dim, 1.0, devPtr_o, dim,
                 devPtr_x, dim, 0.0, devPtr_hold, dim);


    // o = x * hold
    cublasDgemm_v2 (cublasH, 'n','n', dim, dim, dim, 1.0, devPtr_x, dim,
                 devPtr_hold, dim, 0.0, devPtr_o, dim);    



    //    

	      cusolverDnHandle_t cusolverH = NULL;
	    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	    cudaError_t cudaStat1 = cudaSuccess;
	    cudaError_t cudaStat2 = cudaSuccess;
	    cudaError_t cudaStat3 = cudaSuccess;

	    //Step 1: create cusolver handle
	    cusolver_status = cusolverDnCreate(cusolverH);
	    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);	    



	    //Step 2: Copy arrays to device

  double* devPtr_E=0;
  cudaStat1=cublasAlloc (dim, sizeof(devPtr_E), (void**)&devPtr_E);
  assert(cudaSuccess == cudaStat1);

  int* devPtr_devInfo=0;
  cudaStat2=cublasAlloc (1, sizeof(devPtr_devInfo), (void**)&devPtr_devInfo);
  assert(cudaSuccess == cudaStat2);

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

  int lwork = 0;

  // Query the workspace for work buffer size
  cusolver_status = cusolverDnDsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        dim,
        devPtr_o,
        dim,
        devPtr_E,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    double* devPtr_work = 0;

    // Allocate work space    
    cudaStat3 = cudaMalloc((void**)&devPtr_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat3);
    
    // Compute Spectrum
    cusolver_status = cusolverDnDsyevd(
        cusolverH,
        jobz,
        uplo,
	dim,
        devPtr_o,
	dim,
        devPtr_E,
        devPtr_work,
        lwork,
        devPtr_devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(vec, devPtr_o, sizeof(double)*dim, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);




    
    cublasFree (devPtr_o);
    cublasFree (devPtr_x);
    cublasFree (devPtr_hold);
    cublasFree (devPtr_work);
    cublasFree (devPtr_E);        

}
*/
