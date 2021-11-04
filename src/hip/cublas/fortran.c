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
#include "hipblas.h"   /* CUBLAS public header file  */

#include "fortran_common.h"
#include "fortran.h"


int CUBLAS_INIT (void) 
{
    return (int)cublasInit ();
}

int CUBLAS_SHUTDOWN (void) 
{
    return (int)cublasShutdown ();
}

int CUBLAS_ALLOC (const int *n, const int *elemSize, devptr_t *devicePtr)
{    
    void *tPtr;
    int retVal;
    retVal = (int)cublasAlloc (*n, *elemSize, &tPtr);
    *devicePtr = (devptr_t)tPtr;
    return retVal;
}

int CUBLAS_FREE (const devptr_t *devicePtr)
{
    void *tPtr;
    tPtr = (void *)(*devicePtr);
    return (int)cublasFree (tPtr);
}

int CUBLAS_SET_VECTOR (const int *n, const int *elemSize, const void *x,
                       const int *incx, const devptr_t *y, const int *incy)
{
    void *tPtr = (void *)(*y);
    return (int)hipblasSetVector (*n, *elemSize, x, *incx, tPtr, *incy);
}

int CUBLAS_GET_VECTOR (const int *n, const int *elemSize, const devptr_t *x,
                       const int *incx, void *y, const int *incy)
{
    const void *tPtr = (const void *)(*x);
    return (int)hipblasGetVector (*n, *elemSize, tPtr, *incx, y, *incy);
}

int CUBLAS_SET_MATRIX (const int *rows, const int *cols, const int *elemSize,
                       const void *A, const int *lda, const devptr_t *B, 
                       const int *ldb)
{
    void *tPtr = (void *)(*B);
    return (int)hipblasSetMatrix (*rows, *cols, *elemSize, A, *lda, tPtr,*ldb);
}

int CUBLAS_GET_MATRIX (const int *rows, const int *cols, const int *elemSize,
                       const devptr_t *A, const int *lda, void *B, 
                       const int *ldb)
{
    const void *tPtr = (const void *)(*A);
    return (int)hipblasGetMatrix (*rows, *cols, *elemSize, tPtr, *lda, B, *ldb);
}

int CUBLAS_GET_ERROR (void)
{
    return (int)cublasGetError();
}

void CUBLAS_XERBLA (const char *srName, int *info)
{
    cublasXerbla (srName, *info);
}



/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS1 ----------------------------------*/
/*---------------------------------------------------------------------------*/

int CUBLAS_ISAMAX (const int *n, const devptr_t *devPtrx, const int *incx)
{
    float *x = (float *)(*devPtrx);
    int retVal;
    retVal = hipblasIsamax (*n, x, *incx);
    return retVal;
}

int CUBLAS_ISAMIN (const int *n, const devptr_t *devPtrx, const int *incx)
{
    float *x = (float *)(*devPtrx);
    int retVal;
    retVal = hipblasIsamin (*n, x, *incx);
    return retVal;
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SASUM (const int *n, const devptr_t *devPtrx, const int *incx)
#else
float CUBLAS_SASUM (const int *n, const devptr_t *devPtrx, const int *incx)
#endif
{
    float *x = (float *)(*devPtrx);
    float retVal;
    retVal = hipblasSasum (*n, x, *incx);
    return retVal;
}

void CUBLAS_SAXPY (const int *n, const float *alpha, const devptr_t *devPtrx, 
                   const int *incx, const devptr_t *devPtry, const int *incy)
{
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);
    hipblasSaxpy (*n, *alpha, x, *incx, y, *incy);
}

void CUBLAS_SCOPY (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy)
{
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);
    hipblasScopy (*n, x, *incx, y, *incy);
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SDOT (const int *n, const devptr_t *devPtrx, const int *incx, 
                    const devptr_t *devPtry, const int *incy)
#else
float CUBLAS_SDOT (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy)
#endif
{
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);
    return hipblasSdot (*n, x, *incx, y, *incy);
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SNRM2 (const int *n, const devptr_t *devPtrx, const int *incx)
#else
float CUBLAS_SNRM2 (const int *n, const devptr_t *devPtrx, const int *incx)
#endif
{
    float *x = (float *)(*devPtrx);
    return hipblasSnrm2 (*n, x, *incx);
}

void CUBLAS_SROT (const int *n, const devptr_t *devPtrx, const int *incx, 
                  const devptr_t *devPtry, const int *incy, const float *sc, 
                  const float *ss)
{
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);
    hipblasSrot (*n, x, *incx, y, *incy, *sc, *ss);
}

void CUBLAS_SROTG (float *sa, float *sb, float *sc, float *ss)
{
    hipblasSrotg (sa, sb, sc, ss);
}

void CUBLAS_SROTM (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy, 
                   const float* sparam) 
{
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);
    hipblasSrotm (*n, x, *incx, y, *incy, sparam);
}

void CUBLAS_SROTMG (float *sd1, float *sd2, float *sx1, const float *sy1,
                    float* sparam)
{
    hipblasSrotmg (sd1, sd2, sx1, sy1, sparam);
}

void CUBLAS_SSCAL (const int *n, const float *alpha, const devptr_t *devPtrx,
                   const int *incx)
{
    float *x = (float *)(*devPtrx);
    hipblasSscal (*n, *alpha, x, *incx);
}

void CUBLAS_SSWAP (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy)
{
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);
    hipblasSswap (*n, x, *incx, y, *incy);
}

void CUBLAS_CAXPY (const int *n, const hipComplex *alpha, 
                   const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipblasCaxpy (*n, *alpha, x, *incx, y, *incy);
}

void CUBLAS_ZAXPY (const int *n, const hipDoubleComplex *alpha, 
                   const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    hipblasZaxpy (*n, *alpha, x, *incx, y, *incy);
}

void CUBLAS_CCOPY (const int *n, const devptr_t *devPtrx, const int *incx,
                   const devptr_t *devPtry, const int *incy)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipblasCcopy (*n, x, *incx, y, *incy);
}
void CUBLAS_ZCOPY (const int *n, const devptr_t *devPtrx, const int *incx,
                   const devptr_t *devPtry, const int *incy)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    hipblasZcopy (*n, x, *incx, y, *incy);
}
void CUBLAS_CROT (const int *n, const devptr_t *devPtrx, const int *incx, 
                  const devptr_t *devPtry, const int *incy, const float *sc, 
                  const hipComplex *cs)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipblasCrot (*n, x, *incx, y, *incy, *sc, *cs);
}

void CUBLAS_ZROT (const int *n, const devptr_t *devPtrx, const int *incx, 
                  const devptr_t *devPtry, const int *incy, const double *sc, 
                  const hipDoubleComplex *cs)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    hipblasZrot (*n, x, *incx, y, *incy, *sc, *cs);
}

void CUBLAS_CROTG (hipComplex *ca, const hipComplex *cb, float *sc,
                   hipComplex *cs)
{
    hipblasCrotg (ca, *cb, sc, cs);
}

void CUBLAS_ZROTG (hipDoubleComplex *ca, const hipDoubleComplex *cb, double *sc,
                   hipDoubleComplex *cs)
{
    hipblasZrotg (ca, *cb, sc, cs);
}

void CUBLAS_CSCAL (const int *n, const hipComplex *alpha, 
                   const devptr_t *devPtrx, const int *incx)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipblasCscal (*n, *alpha, x, *incx);
}

void CUBLAS_CSROT (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy, const float *sc, 
                   const float *ss)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipblasCsrot (*n, x, *incx, y, *incy, *sc, *ss);
}

void CUBLAS_ZDROT (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy, const double *sc, 
                   const double *ss)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    hipblasZdrot (*n, x, *incx, y, *incy, *sc, *ss);
}

void CUBLAS_CSSCAL (const int *n, const float *alpha, const devptr_t *devPtrx,
                    const int *incx)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipblasCsscal (*n, *alpha, x, *incx);
}

void CUBLAS_CSWAP (const int *n, const devptr_t *devPtrx, const int *incx,
                   const devptr_t *devPtry, const int *incy)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipblasCswap (*n, x, *incx, y, *incy);
}

void CUBLAS_ZSWAP (const int *n, const devptr_t *devPtrx, const int *incx,
                   const devptr_t *devPtry, const int *incy)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    hipblasZswap (*n, x, *incx, y, *incy);
}

void CUBLAS_CTRMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);       
    hipblasCtrmv (uplo[0], trans[0], diag[0], *n, A, *lda, x, *incx);
}

void CUBLAS_ZTRMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);       
    hipblasZtrmv (uplo[0], trans[0], diag[0], *n, A, *lda, x, *incx);
}
#ifdef RETURN_COMPLEX
hipComplex CUBLAS_CDOTU ( const int *n, const devptr_t *devPtrx,
                   const int *incx, const devptr_t *devPtry,const int *incy)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipComplex retVal = hipblasCdotu (*n, x, *incx, y, *incy);
    return retVal;
}
#else
void CUBLAS_CDOTU (hipComplex *retVal, const int *n, const devptr_t *devPtrx,
                   const int *incx, const devptr_t *devPtry,const int *incy)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    *retVal = hipblasCdotu (*n, x, *incx, y, *incy);
}
#endif
#ifdef RETURN_COMPLEX
hipComplex CUBLAS_CDOTC ( const int *n, const devptr_t *devPtrx,
                   const int *incx, const devptr_t *devPtry, const int *incy)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipComplex retVal = hipblasCdotc (*n, x, *incx, y, *incy);
    return retVal;
}
#else
void CUBLAS_CDOTC (hipComplex *retVal, const int *n, const devptr_t *devPtrx,
                   const int *incx, const devptr_t *devPtry, const int *incy)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    *retVal = hipblasCdotc (*n, x, *incx, y, *incy);
}
#endif
int CUBLAS_ICAMAX (const int *n, const devptr_t *devPtrx, const int *incx)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    return hipblasIcamax (*n, x, *incx);
}

int CUBLAS_ICAMIN (const int *n, const devptr_t *devPtrx, const int *incx)
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    return hipblasIcamin (*n, x, *incx);
}

int CUBLAS_IZAMAX (const int *n, const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    return hipblasIzamax (*n, x, *incx);
}

int CUBLAS_IZAMIN (const int *n, const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    return hipblasIzamin (*n, x, *incx);
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SCASUM (const int *n, const devptr_t *devPtrx, const int *incx)
#else
float CUBLAS_SCASUM (const int *n, const devptr_t *devPtrx, const int *incx)
#endif
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    return hipblasScasum (*n, x, *incx);
}

double CUBLAS_DZASUM (const int *n, const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    return hipblasDzasum (*n, x, *incx);
}

#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SCNRM2 (const int *n, const devptr_t *devPtrx, const int *incx)
#else
float CUBLAS_SCNRM2 (const int *n, const devptr_t *devPtrx, const int *incx)
#endif
{
    hipComplex *x = (hipComplex *)(*devPtrx);
    return hipblasScnrm2 (*n, x, *incx);
}

double CUBLAS_DZNRM2 (const int *n, const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    return hipblasDznrm2 (*n, x, *incx);
}

int CUBLAS_IDAMAX (const int *n, const devptr_t *devPtrx, const int *incx)
{
    double *x = (double *)(*devPtrx);
    int retVal;
    retVal = hipblasIdamax (*n, x, *incx);
    return retVal;
}

int CUBLAS_IDAMIN (const int *n, const devptr_t *devPtrx, const int *incx)
{
    double *x = (double *)(*devPtrx);
    int retVal;
    retVal = hipblasIdamin (*n, x, *incx);
    return retVal;
}

double CUBLAS_DASUM (const int *n, const devptr_t *devPtrx, const int *incx)
{
    double *x = (double *)(*devPtrx);
    double retVal;
    retVal = hipblasDasum (*n, x, *incx);
    return retVal;
}

void CUBLAS_DAXPY (const int *n, const double *alpha, const devptr_t *devPtrx, 
                   const int *incx, const devptr_t *devPtry, const int *incy)
{
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);
    hipblasDaxpy (*n, *alpha, x, *incx, y, *incy);
}

void CUBLAS_DCOPY (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy)
{
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);
    hipblasDcopy (*n, x, *incx, y, *incy);
}

double CUBLAS_DDOT (const int *n, const devptr_t *devPtrx, const int *incx, 
                    const devptr_t *devPtry, const int *incy)
{
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);
    return hipblasDdot (*n, x, *incx, y, *incy);
}

double CUBLAS_DNRM2 (const int *n, const devptr_t *devPtrx, const int *incx)
{
    double *x = (double *)(*devPtrx);
    return hipblasDnrm2 (*n, x, *incx);
}

void CUBLAS_DROT (const int *n, const devptr_t *devPtrx, const int *incx, 
                  const devptr_t *devPtry, const int *incy, const double *sc, 
                  const double *ss)
{
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);
    hipblasDrot (*n, x, *incx, y, *incy, *sc, *ss);
}

void CUBLAS_DROTG (double *sa, double *sb, double *sc, double *ss)
{
    hipblasDrotg (sa, sb, sc, ss);
}

void CUBLAS_DROTM (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy, 
                   const double* sparam) 
{
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);
    hipblasDrotm (*n, x, *incx, y, *incy, sparam);
}

void CUBLAS_DROTMG (double *sd1, double *sd2, double *sx1, const double *sy1,
                    double* sparam)
{
    hipblasDrotmg (sd1, sd2, sx1, sy1, sparam);
}

void CUBLAS_DSCAL (const int *n, const double *alpha, const devptr_t *devPtrx,
                   const int *incx)
{
    double *x = (double *)(*devPtrx);
    hipblasDscal (*n, *alpha, x, *incx);
}

void CUBLAS_DSWAP (const int *n, const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy)
{
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);
    hipblasDswap (*n, x, *incx, y, *incy);
}
#ifdef RETURN_COMPLEX
hipDoubleComplex CUBLAS_ZDOTU ( const int *n, const devptr_t *devPtrx,
                   const int *incx, const devptr_t *devPtry,const int *incy)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    return (hipblasZdotu (*n, x, *incx, y, *incy));
}
#else
void CUBLAS_ZDOTU (hipDoubleComplex *retVal, const int *n, const devptr_t *devPtrx,
                   const int *incx, const devptr_t *devPtry,const int *incy)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    *retVal = hipblasZdotu (*n, x, *incx, y, *incy);
}
#endif
#ifdef RETURN_COMPLEX
hipDoubleComplex CUBLAS_ZDOTC ( const int *n, const devptr_t *devPtrx,
                   const int *incx, const devptr_t *devPtry,const int *incy)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    return (hipblasZdotc (*n, x, *incx, y, *incy));
}
#else
void CUBLAS_ZDOTC (hipDoubleComplex *retVal, const int *n, const devptr_t *devPtrx,
                   const int *incx, const devptr_t *devPtry,const int *incy)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    *retVal = hipblasZdotc (*n, x, *incx, y, *incy);
}
#endif
void CUBLAS_ZSCAL (const int *n, const hipDoubleComplex *alpha, 
                   const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipblasZscal (*n, *alpha, x, *incx);
}

void CUBLAS_ZDSCAL (const int *n, const double *alpha, const devptr_t *devPtrx,
                    const int *incx)
{
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipblasZdscal (*n, *alpha, x, *incx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS2 ----------------------------------*/
/*---------------------------------------------------------------------------*/

void CUBLAS_SGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const float *alpha,
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrx, const int *incx, const float *beta,
                   const devptr_t *devPtry, const int *incy)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);
    hipblasSgbmv (trans[0], *m, *n, *kl, *ku, *alpha, A, *lda, x, *incx, *beta,
                 y, *incy);
}

void CUBLAS_DGBMV (const char *trans, const int *m, const int *n,
                   const int *kl, const int *ku, const double *alpha, 
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrx, const int *incx, const double *beta,
                   const devptr_t *devPtry, const int *incy)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);
    hipblasDgbmv (trans[0], *m, *n, *kl, *ku, *alpha, A, *lda, x, *incx, *beta,
                 y, *incy);
}                   
void CUBLAS_CGBMV (const char *trans, const int *m, const int *n,
                   const int *kl, const int *ku, const hipComplex *alpha, 
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrx, const int *incx, const hipComplex *beta,
                   const devptr_t *devPtry, const int *incy)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipblasCgbmv (trans[0], *m, *n, *kl, *ku, *alpha, A, *lda, x, *incx, *beta,
                 y, *incy);
}                   
void CUBLAS_ZGBMV (const char *trans, const int *m, const int *n,
                   const int *kl, const int *ku, const hipDoubleComplex *alpha, 
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrx, const int *incx, const hipDoubleComplex *beta,
                   const devptr_t *devPtry, const int *incy)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    hipblasZgbmv (trans[0], *m, *n, *kl, *ku, *alpha, A, *lda, x, *incx, *beta,
                 y, *incy);
}                   

void CUBLAS_SGEMV (const char *trans, const int *m, const int *n, 
                   const float *alpha, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx, const float *beta,
                   const devptr_t *devPtry, const int *incy)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);
    hipblasSgemv (trans[0], *m, *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_SGER (const int *m, const int *n, const float *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtry, const int *incy,
                  const devptr_t *devPtrA, const int *lda)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);    
    hipblasSger (*m, *n, *alpha, x, *incx, y, *incy, A, *lda);
}

void CUBLAS_SSBMV (const char *uplo, const int *n, const int *k,
                   const float *alpha, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx, const float *beta,
                   const devptr_t *devPtry, const int *incy)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);    
    hipblasSsbmv (uplo[0], *n, *k, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_DSBMV (const char *uplo, const int *n, const int *k,
                   const double *alpha, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx, const double *beta,
                   const devptr_t *devPtry, const int *incy)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);    
    hipblasDsbmv (uplo[0], *n, *k, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_CHBMV (const char *uplo, const int *n, const int *k,
                   const hipComplex *alpha, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx, const hipComplex *beta,
                   const devptr_t *devPtry, const int *incy)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);    
    hipblasChbmv (uplo[0], *n, *k, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_ZHBMV (const char *uplo, const int *n, const int *k,
                   const hipDoubleComplex *alpha, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx, const hipDoubleComplex *beta,
                   const devptr_t *devPtry, const int *incy)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);    
    hipblasZhbmv (uplo[0], *n, *k, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_SSPMV (const char *uplo, const int *n, const float *alpha,
                   const devptr_t *devPtrAP, const devptr_t *devPtrx,
                   const int *incx, const float *beta, const devptr_t *devPtry,
                   const int *incy)
{
    float *AP = (float *)(*devPtrAP);
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);    
    hipblasSspmv (uplo[0], *n, *alpha, AP, x, *incx, *beta, y, *incy);
}
void CUBLAS_DSPMV (const char *uplo, const int *n, const double *alpha,
                   const devptr_t *devPtrAP, const devptr_t *devPtrx,
                   const int *incx, const double *beta, const devptr_t *devPtry,
                   const int *incy)
{
    double *AP = (double *)(*devPtrAP);
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);    
    hipblasDspmv (uplo[0], *n, *alpha, AP, x, *incx, *beta, y, *incy);
}
void CUBLAS_CHPMV (const char *uplo, const int *n, const hipComplex *alpha,
                   const devptr_t *devPtrAP, const devptr_t *devPtrx,
                   const int *incx, const hipComplex *beta, const devptr_t *devPtry,
                   const int *incy)
{
    hipComplex *AP = (hipComplex *)(*devPtrAP);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);    
    hipblasChpmv (uplo[0], *n, *alpha, AP, x, *incx, *beta, y, *incy);
}
void CUBLAS_ZHPMV (const char *uplo, const int *n, const hipDoubleComplex *alpha,
                   const devptr_t *devPtrAP, const devptr_t *devPtrx,
                   const int *incx, const hipDoubleComplex *beta, const devptr_t *devPtry,
                   const int *incy)
{
    hipDoubleComplex *AP = (hipDoubleComplex *)(*devPtrAP);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);    
    hipblasZhpmv (uplo[0], *n, *alpha, AP, x, *incx, *beta, y, *incy);
}

void CUBLAS_SSPR (const char *uplo, const int *n, const float *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtrAP)
{
    float *AP = (float *)(*devPtrAP);
    float *x = (float *)(*devPtrx);
    hipblasSspr (uplo[0], *n, *alpha, x, *incx, AP);
}

void CUBLAS_DSPR (const char *uplo, const int *n, const double *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtrAP)
{
    double *AP = (double *)(*devPtrAP);
    double *x = (double *)(*devPtrx);
    hipblasDspr (uplo[0], *n, *alpha, x, *incx, AP);
}

void CUBLAS_CHPR (const char *uplo, const int *n, const float *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtrAP)
{
    hipComplex *AP = (hipComplex *)(*devPtrAP);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipblasChpr (uplo[0], *n, *alpha, x, *incx, AP);
}

void CUBLAS_ZHPR (const char *uplo, const int *n, const double *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtrAP)
{
    hipDoubleComplex *AP = (hipDoubleComplex *)(*devPtrAP);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipblasZhpr (uplo[0], *n, *alpha, x, *incx, AP);
}


void CUBLAS_SSPR2 (const char *uplo, const int *n, const float *alpha,
                   const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy,
                   const devptr_t *devPtrAP)
{
    float *AP = (float *)(*devPtrAP);
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);    
    hipblasSspr2 (uplo[0], *n, *alpha, x, *incx, y, *incy, AP);
}

void CUBLAS_DSPR2 (const char *uplo, const int *n, const double *alpha,
                   const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy,
                   const devptr_t *devPtrAP)
{
    double *AP = (double *)(*devPtrAP);
    double *x  = (double *)(*devPtrx);
    double *y  = (double *)(*devPtry);    
    hipblasDspr2 (uplo[0], *n, *alpha, x, *incx, y, *incy, AP);
}

void CUBLAS_CHPR2 (const char *uplo, const int *n, const hipComplex *alpha,
                   const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy,
                   const devptr_t *devPtrAP)
{
    hipComplex *AP = (hipComplex *)(*devPtrAP);
    hipComplex *x  = (hipComplex *)(*devPtrx);
    hipComplex *y  = (hipComplex *)(*devPtry);    
    hipblasChpr2 (uplo[0], *n, *alpha, x, *incx, y, *incy, AP);
}

void CUBLAS_ZHPR2 (const char *uplo, const int *n, const hipDoubleComplex *alpha,
                   const devptr_t *devPtrx, const int *incx, 
                   const devptr_t *devPtry, const int *incy,
                   const devptr_t *devPtrAP)
{
    hipDoubleComplex *AP = (hipDoubleComplex *)(*devPtrAP);
    hipDoubleComplex *x  = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y  = (hipDoubleComplex *)(*devPtry);    
    hipblasZhpr2 (uplo[0], *n, *alpha, x, *incx, y, *incy, AP);
}

void CUBLAS_SSYMV (const char *uplo, const int *n, const float *alpha,
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrx, const int *incx, const float *beta,
                   const devptr_t *devPtry,
                   const int *incy)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);    
    hipblasSsymv (uplo[0], *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_DSYMV (const char *uplo, const int *n, const double *alpha,
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrx, const int *incx, const double *beta,
                   const devptr_t *devPtry,
                   const int *incy)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);    
    hipblasDsymv (uplo[0], *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_CHEMV (const char *uplo, const int *n, const hipComplex *alpha,
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrx, const int *incx, const hipComplex *beta,
                   const devptr_t *devPtry,
                   const int *incy)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);    
    hipblasChemv (uplo[0], *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_ZHEMV (const char *uplo, const int *n, const hipDoubleComplex *alpha,
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrx, const int *incx, const hipDoubleComplex *beta,
                   const devptr_t *devPtry,
                   const int *incy)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);    
    hipblasZhemv (uplo[0], *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_SSYR (const char *uplo, const int *n, const float *alpha,
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtrA, const int *lda)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);    
    hipblasSsyr (uplo[0], *n, *alpha, x, *incx, A, *lda);
}

void CUBLAS_SSYR2 (const char *uplo, const int *n, const float *alpha,
                   const devptr_t *devPtrx, const int *incx,
                   const devptr_t *devPtry, const int *incy, 
                   const devptr_t *devPtrA, const int *lda)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);
    float *y = (float *)(*devPtry);    
    hipblasSsyr2 (uplo[0], *n, *alpha, x, *incx, y, *incy, A, *lda);
}

void CUBLAS_DSYR2 (const char *uplo, const int *n, const double *alpha,
                   const devptr_t *devPtrx, const int *incx,
                   const devptr_t *devPtry, const int *incy, 
                   const devptr_t *devPtrA, const int *lda)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);    
    hipblasDsyr2 (uplo[0], *n, *alpha, x, *incx, y, *incy, A, *lda);
}

void CUBLAS_CHER2 (const char *uplo, const int *n, const hipComplex *alpha,
                   const devptr_t *devPtrx, const int *incx,
                   const devptr_t *devPtry, const int *incy, 
                   const devptr_t *devPtrA, const int *lda)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);    
    hipblasCher2 (uplo[0], *n, *alpha, x, *incx, y, *incy, A, *lda);
}

void CUBLAS_ZHER2 (const char *uplo, const int *n, const hipDoubleComplex *alpha,
                   const devptr_t *devPtrx, const int *incx,
                   const devptr_t *devPtry, const int *incy, 
                   const devptr_t *devPtrA, const int *lda)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);    
    hipblasZher2 (uplo[0], *n, *alpha, x, *incx, y, *incy, A, *lda);
}


void CUBLAS_STBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrx, const int *incx)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);    
    hipblasStbmv (uplo[0], trans[0], diag[0], *n, *k, A, *lda, x, *incx);
}

void CUBLAS_DTBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrx, const int *incx)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);    
    hipblasDtbmv (uplo[0], trans[0], diag[0], *n, *k, A, *lda, x, *incx);
}

void CUBLAS_CTBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrx, const int *incx)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);    
    hipblasCtbmv (uplo[0], trans[0], diag[0], *n, *k, A, *lda, x, *incx);
}

void CUBLAS_ZTBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);    
    hipblasZtbmv (uplo[0], trans[0], diag[0], *n, *k, A, *lda, x, *incx);
}

void CUBLAS_STBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrx, const int *incx)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);       
    hipblasStbsv (uplo[0], trans[0], diag[0], *n, *k, A, *lda, x, *incx);
}

void CUBLAS_DTBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrx, const int *incx)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);       
    hipblasDtbsv (uplo[0], trans[0], diag[0], *n, *k, A, *lda, x, *incx);
}

void CUBLAS_CTBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrx, const int *incx)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);       
    hipblasCtbsv (uplo[0], trans[0], diag[0], *n, *k, A, *lda, x, *incx);
}

void CUBLAS_ZTBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);       
    hipblasZtbsv (uplo[0], trans[0], diag[0], *n, *k, A, *lda, x, *incx);
}

void CUBLAS_STPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n,  const devptr_t *devPtrAP, 
                   const devptr_t *devPtrx, const int *incx)
{
    float *AP = (float *)(*devPtrAP);
    float *x = (float *)(*devPtrx);       
    hipblasStpmv (uplo[0], trans[0], diag[0], *n, AP, x, *incx);
}

void CUBLAS_DTPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n,  const devptr_t *devPtrAP, 
                   const devptr_t *devPtrx, const int *incx)
{
    double *AP = (double *)(*devPtrAP);
    double *x = (double *)(*devPtrx);       
    hipblasDtpmv (uplo[0], trans[0], diag[0], *n, AP, x, *incx);
}

void CUBLAS_CTPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n,  const devptr_t *devPtrAP, 
                   const devptr_t *devPtrx, const int *incx)
{
    hipComplex *AP = (hipComplex *)(*devPtrAP);
    hipComplex *x = (hipComplex *)(*devPtrx);       
    hipblasCtpmv (uplo[0], trans[0], diag[0], *n, AP, x, *incx);
}

void CUBLAS_ZTPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n,  const devptr_t *devPtrAP, 
                   const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *AP = (hipDoubleComplex *)(*devPtrAP);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);       
    hipblasZtpmv (uplo[0], trans[0], diag[0], *n, AP, x, *incx);
}

void CUBLAS_STPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrAP, 
                   const devptr_t *devPtrx, const int *incx)
{
    float *AP = (float *)(*devPtrAP);
    float *x = (float *)(*devPtrx);       
    hipblasStpsv (uplo[0], trans[0], diag[0], *n, AP, x, *incx);
}

void CUBLAS_DTPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrAP, 
                   const devptr_t *devPtrx, const int *incx)
{
    double *AP = (double *)(*devPtrAP);
    double *x = (double *)(*devPtrx);       
    hipblasDtpsv (uplo[0], trans[0], diag[0], *n, AP, x, *incx);
}

void CUBLAS_CTPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrAP, 
                   const devptr_t *devPtrx, const int *incx)
{
    hipComplex *AP = (hipComplex *)(*devPtrAP);
    hipComplex *x = (hipComplex *)(*devPtrx);       
    hipblasCtpsv (uplo[0], trans[0], diag[0], *n, AP, x, *incx);
}

void CUBLAS_ZTPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrAP, 
                   const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *AP = (hipDoubleComplex *)(*devPtrAP);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);       
    hipblasZtpsv (uplo[0], trans[0], diag[0], *n, AP, x, *incx);
}

void CUBLAS_STRMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);       
    hipblasStrmv (uplo[0], trans[0], diag[0], *n, A, *lda, x, *incx);
}

void CUBLAS_DTRMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);       
    hipblasDtrmv (uplo[0], trans[0], diag[0], *n, A, *lda, x, *incx);
}

void CUBLAS_STRSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx)
{
    float *A = (float *)(*devPtrA);
    float *x = (float *)(*devPtrx);       
    hipblasStrsv (uplo[0], trans[0], diag[0], *n, A, *lda, x, *incx);
}

void CUBLAS_DGEMV (const char *trans, const int *m, const int *n, 
                   const double *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrx, const int *incx,
                   const double *beta, const devptr_t *devPtry,
                   const int *incy)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);
    hipblasDgemv (trans[0], *m, *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}
void CUBLAS_CGEMV (const char *trans, const int *m, const int *n,
                   const hipComplex *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrx, const int *incx,
                   const hipComplex *beta, devptr_t *devPtry,
                   const int *incy)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);
    hipblasCgemv (trans[0], *m, *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void CUBLAS_ZGEMV (const char *trans, const int *m, const int *n,
                   const hipDoubleComplex *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrx, const int *incx,
                   const hipDoubleComplex *beta, devptr_t *devPtry,
                   const int *incy)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);
    hipblasZgemv (trans[0], *m, *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}
void CUBLAS_DGER (const int *m, const int *n, const double *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtry, const int *incy,
                  const devptr_t *devPtrA, const int *lda)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);
    double *y = (double *)(*devPtry);    
    hipblasDger (*m, *n, *alpha, x, *incx, y, *incy, A, *lda);
}

void CUBLAS_DSYR (const char *uplo, const int *n, const double *alpha,
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtrA, const int *lda)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);    
    hipblasDsyr (uplo[0], *n, *alpha, x, *incx, A, *lda);
}

void CUBLAS_CHER (const char *uplo, const int *n, const float *alpha,
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtrA, const int *lda)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);    
    hipblasCher (uplo[0], *n, *alpha, x, *incx, A, *lda);
}

void CUBLAS_ZHER (const char *uplo, const int *n, const double *alpha,
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtrA, const int *lda)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);    
    hipblasZher (uplo[0], *n, *alpha, x, *incx, A, *lda);
}

void CUBLAS_DTRSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx)
{
    double *A = (double *)(*devPtrA);
    double *x = (double *)(*devPtrx);       
    hipblasDtrsv (uplo[0], trans[0], diag[0], *n, A, *lda, x, *incx);
}

void CUBLAS_CTRSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);       
    hipblasCtrsv (uplo[0], trans[0], diag[0], *n, A, *lda, x, *incx);
}

void CUBLAS_ZTRSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrx, const int *incx)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);       
    hipblasZtrsv (uplo[0], trans[0], diag[0], *n, A, *lda, x, *incx);
}

void CUBLAS_CGERU (const int *m, const int *n, const hipComplex *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtry, const int *incy,
                  const devptr_t *devPtrA, const int *lda)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);    
    hipblasCgeru (*m, *n, *alpha, x, *incx, y, *incy, A, *lda);
}

void CUBLAS_CGERC (const int *m, const int *n, const hipComplex *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtry, const int *incy,
                  const devptr_t *devPtrA, const int *lda)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *x = (hipComplex *)(*devPtrx);
    hipComplex *y = (hipComplex *)(*devPtry);    
    hipblasCgerc (*m, *n, *alpha, x, *incx, y, *incy, A, *lda);
}

void CUBLAS_ZGERU (const int *m, const int *n, const hipDoubleComplex *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtry, const int *incy,
                  const devptr_t *devPtrA, const int *lda)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);    
    hipblasZgeru (*m, *n, *alpha, x, *incx, y, *incy, A, *lda);
}

void CUBLAS_ZGERC (const int *m, const int *n, const hipDoubleComplex *alpha, 
                  const devptr_t *devPtrx, const int *incx,
                  const devptr_t *devPtry, const int *incy,
                  const devptr_t *devPtrA, const int *lda)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *x = (hipDoubleComplex *)(*devPtrx);
    hipDoubleComplex *y = (hipDoubleComplex *)(*devPtry);    
    hipblasZgerc (*m, *n, *alpha, x, *incx, y, *incy, A, *lda);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS3 ----------------------------------*/
/*---------------------------------------------------------------------------*/

void CUBLAS_SGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const float *alpha,
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrB, const int *ldb, const float *beta,
                   const devptr_t *devPtrC, const int *ldc)
{
    float *A = (float *)(*devPtrA);
    float *B = (float *)(*devPtrB);
    float *C = (float *)(*devPtrC);
    hipblasSgemm (transa[0], transb[0], *m, *n, *k, *alpha, A, *lda, 
                 B, *ldb, *beta, C, *ldc);
}

void CUBLAS_SSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const float *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrB, const int *ldb, 
                   const float *beta, const devptr_t *devPtrC, const int *ldc)
{
    float *A = (float *)(*devPtrA);
    float *B = (float *)(*devPtrB);
    float *C = (float *)(*devPtrC);
    hipblasSsymm (*side, *uplo, *m, *n, *alpha, A, *lda, B, *ldb, *beta, C,
                 *ldc);
}

void CUBLAS_SSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const float *alpha, const devptr_t *devPtrA,
                    const int *lda, const devptr_t *devPtrB, const int *ldb, 
                    const float *beta, const devptr_t *devPtrC, const int *ldc)
{
    float *A = (float *)(*devPtrA);
    float *B = (float *)(*devPtrB);
    float *C = (float *)(*devPtrC);
    hipblasSsyr2k (*uplo, *trans, *n, *k, *alpha, A, *lda, B, *ldb, *beta, 
                  C, *ldc);
}

void CUBLAS_SSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const float *alpha, const devptr_t *devPtrA, 
                   const int *lda, const float *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    float *A = (float *)(*devPtrA);
    float *C = (float *)(*devPtrC);
    hipblasSsyrk (*uplo, *trans, *n, *k, *alpha, A, *lda, *beta, C, *ldc);
}

void CUBLAS_STRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const float *alpha, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrB, const int *ldb)
{
    float *A = (float *)(*devPtrA);
    float *B = (float *)(*devPtrB);
    hipblasStrmm (*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B,
                 *ldb);
}

void CUBLAS_STRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n, 
                   const float *alpha, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrB, const int *ldb)
{
    float *A = (float *)*devPtrA;
    float *B = (float *)*devPtrB;
    hipblasStrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha,
                 A, *lda, B, *ldb);
}

void CUBLAS_CGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const hipComplex *alpha,
                   const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrB, const int *ldb, 
                   const hipComplex *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    hipComplex *A = (hipComplex *)*devPtrA;
    hipComplex *B = (hipComplex *)*devPtrB;
    hipComplex *C = (hipComplex *)*devPtrC;    
    hipblasCgemm (transa[0], transb[0], *m, *n, *k, *alpha, A, *lda, B, *ldb, 
                 *beta, C, *ldc);
}


void CUBLAS_CSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const hipComplex *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrB, const int *ldb, 
                   const hipComplex *beta, const devptr_t *devPtrC, const int *ldc)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *B = (hipComplex *)(*devPtrB);
    hipComplex *C = (hipComplex *)(*devPtrC);
    hipblasCsymm (*side, *uplo, *m, *n, *alpha, A, *lda, B, *ldb, *beta, C,
                 *ldc);
}

void CUBLAS_CHEMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const hipComplex *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrB, const int *ldb, 
                   const hipComplex *beta, const devptr_t *devPtrC, const int *ldc)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *B = (hipComplex *)(*devPtrB);
    hipComplex *C = (hipComplex *)(*devPtrC);
    hipblasChemm (*side, *uplo, *m, *n, *alpha, A, *lda, B, *ldb, *beta, C,
                 *ldc);
}

void CUBLAS_CTRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const hipComplex *alpha, const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrB, const int *ldb)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *B = (hipComplex *)(*devPtrB);
    hipblasCtrmm (*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B,
                 *ldb);
}

void CUBLAS_CTRSM ( const char *side, const char *uplo, const char *transa, 
                    const char *diag, const int *m, const int *n,
                    const hipComplex *alpha, const devptr_t *devPtrA, const int *lda,
                    const devptr_t *devPtrB, const int *ldb)
{
    hipComplex *A = (hipComplex *)*devPtrA;
    hipComplex *B = (hipComplex *)*devPtrB;
    hipblasCtrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha,
                 A, *lda, B, *ldb);
}

void CUBLAS_CSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const hipComplex *alpha, const devptr_t *devPtrA, 
                   const int *lda, const hipComplex *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *C = (hipComplex *)(*devPtrC);
    hipblasCsyrk (*uplo, *trans, *n, *k, *alpha, A, *lda, *beta, C, *ldc);
}

void CUBLAS_CSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const hipComplex *alpha, const devptr_t *devPtrA,
                    const int *lda, const devptr_t *devPtrB, const int *ldb, 
                    const hipComplex *beta, const devptr_t *devPtrC,
                    const int *ldc)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *B = (hipComplex *)(*devPtrB);
    hipComplex *C = (hipComplex *)(*devPtrC);
    hipblasCsyr2k (*uplo, *trans, *n, *k, *alpha, A, *lda, B, *ldb, *beta, 
                  C, *ldc);
}

void CUBLAS_CHERK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const float *alpha, const devptr_t *devPtrA, 
                   const int *lda, const float *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *C = (hipComplex *)(*devPtrC);
    hipblasCherk (*uplo, *trans, *n, *k, *alpha, A, *lda, *beta, C, *ldc);
}

void CUBLAS_CHER2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const hipComplex *alpha, const devptr_t *devPtrA,
                    const int *lda, const devptr_t *devPtrB, const int *ldb, 
                    const float *beta, const devptr_t *devPtrC,
                    const int *ldc)
{
    hipComplex *A = (hipComplex *)(*devPtrA);
    hipComplex *B = (hipComplex *)(*devPtrB);
    hipComplex *C = (hipComplex *)(*devPtrC);
    hipblasCher2k (*uplo, *trans, *n, *k, *alpha, A, *lda, B, *ldb, *beta, 
                  C, *ldc);
}

void CUBLAS_DGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const double *alpha,
                   const devptr_t *devPtrA, const int *lda, 
                   const devptr_t *devPtrB, const int *ldb, const double *beta,
                   const devptr_t *devPtrC, const int *ldc)
{
    double *A = (double *)(*devPtrA);
    double *B = (double *)(*devPtrB);
    double *C = (double *)(*devPtrC);
    hipblasDgemm (transa[0], transb[0], *m, *n, *k, *alpha, A, *lda, 
                 B, *ldb, *beta, C, *ldc);
}

void CUBLAS_DSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const double *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrB, const int *ldb, 
                   const double *beta, const devptr_t *devPtrC, const int *ldc)
{
    double *A = (double *)(*devPtrA);
    double *B = (double *)(*devPtrB);
    double *C = (double *)(*devPtrC);
    hipblasDsymm (*side, *uplo, *m, *n, *alpha, A, *lda, B, *ldb, *beta, C,
                 *ldc);
}

void CUBLAS_DSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const double *alpha, const devptr_t *devPtrA,
                    const int *lda, const devptr_t *devPtrB, const int *ldb, 
                    const double *beta, const devptr_t *devPtrC,
                    const int *ldc)
{
    double *A = (double *)(*devPtrA);
    double *B = (double *)(*devPtrB);
    double *C = (double *)(*devPtrC);
    hipblasDsyr2k (*uplo, *trans, *n, *k, *alpha, A, *lda, B, *ldb, *beta, 
                  C, *ldc);
}

void CUBLAS_DSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const double *alpha, const devptr_t *devPtrA, 
                   const int *lda, const double *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    double *A = (double *)(*devPtrA);
    double *C = (double *)(*devPtrC);
    hipblasDsyrk (*uplo, *trans, *n, *k, *alpha, A, *lda, *beta, C, *ldc);
}

void CUBLAS_ZSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const hipDoubleComplex *alpha, const devptr_t *devPtrA, 
                   const int *lda, const hipDoubleComplex *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *C = (hipDoubleComplex *)(*devPtrC);
    hipblasZsyrk (*uplo, *trans, *n, *k, *alpha, A, *lda, *beta, C, *ldc);
}

void CUBLAS_ZSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const hipDoubleComplex *alpha, const devptr_t *devPtrA,
                    const int *lda, const devptr_t *devPtrB, const int *ldb, 
                    const hipDoubleComplex *beta, const devptr_t *devPtrC,
                    const int *ldc)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *B = (hipDoubleComplex *)(*devPtrB);
    hipDoubleComplex *C = (hipDoubleComplex *)(*devPtrC);
    hipblasZsyr2k (*uplo, *trans, *n, *k, *alpha, A, *lda, B, *ldb, *beta, 
                  C, *ldc);
}

void CUBLAS_DTRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const double *alpha, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrB, const int *ldb)
{
    double *A = (double *)(*devPtrA);
    double *B = (double *)(*devPtrB);
    hipblasDtrmm (*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B,
                 *ldb);
}

void CUBLAS_ZTRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const hipDoubleComplex *alpha, const devptr_t *devPtrA, 
                   const int *lda, const devptr_t *devPtrB, const int *ldb)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *B = (hipDoubleComplex *)(*devPtrB);
    hipblasZtrmm (*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B,
                 *ldb);
}


void CUBLAS_DTRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n, 
                   const double *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrB, const int *ldb)
{
    double *A = (double *)*devPtrA;
    double *B = (double *)*devPtrB;
    hipblasDtrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha,
                 A, *lda, B, *ldb);
}

void CUBLAS_ZTRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n, 
                   const hipDoubleComplex *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrB, const int *ldb)
{
    hipDoubleComplex *A = (hipDoubleComplex *)*devPtrA;
    hipDoubleComplex *B = (hipDoubleComplex *)*devPtrB;
    hipblasZtrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha,
                 A, *lda, B, *ldb);
}

void CUBLAS_ZGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const hipDoubleComplex *alpha,
                   const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrB, const int *ldb, 
                   const hipDoubleComplex *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    hipDoubleComplex *A = (hipDoubleComplex *)*devPtrA;
    hipDoubleComplex *B = (hipDoubleComplex *)*devPtrB;
    hipDoubleComplex *C = (hipDoubleComplex *)*devPtrC;    
    hipblasZgemm (transa[0], transb[0], *m, *n, *k, *alpha, A, *lda, B, *ldb, 
                 *beta, C, *ldc);
}


void CUBLAS_ZHERK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const double *alpha, const devptr_t *devPtrA, 
                   const int *lda, const double *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *C = (hipDoubleComplex *)(*devPtrC);
    hipblasZherk (*uplo, *trans, *n, *k, *alpha, A, *lda, *beta, C, *ldc);
}

void CUBLAS_ZHER2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const hipDoubleComplex *alpha, const devptr_t *devPtrA,
                    const int *lda, const devptr_t *devPtrB, const int *ldb, 
                    const double *beta, const devptr_t *devPtrC,
                    const int *ldc)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *B = (hipDoubleComplex *)(*devPtrB);
    hipDoubleComplex *C = (hipDoubleComplex *)(*devPtrC);
    hipblasZher2k (*uplo, *trans, *n, *k, *alpha, A, *lda, B, *ldb, *beta, 
                  C, *ldc);
}


void CUBLAS_ZSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const hipDoubleComplex *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrB, const int *ldb, 
                   const hipDoubleComplex *beta, const devptr_t *devPtrC, const int *ldc)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *B = (hipDoubleComplex *)(*devPtrB);
    hipDoubleComplex *C = (hipDoubleComplex *)(*devPtrC);
    hipblasZsymm (*side, *uplo, *m, *n, *alpha, A, *lda, B, *ldb, *beta, C,
                 *ldc);
}

void CUBLAS_ZHEMM (const char *side, const char *uplo, const int *m, 
                   const int *n, const hipDoubleComplex *alpha, const devptr_t *devPtrA,
                   const int *lda, const devptr_t *devPtrB, const int *ldb, 
                   const hipDoubleComplex *beta, const devptr_t *devPtrC, const int *ldc)
{
    hipDoubleComplex *A = (hipDoubleComplex *)(*devPtrA);
    hipDoubleComplex *B = (hipDoubleComplex *)(*devPtrB);
    hipDoubleComplex *C = (hipDoubleComplex *)(*devPtrC);
    hipblasZhemm (*side, *uplo, *m, *n, *alpha, A, *lda, B, *ldb, *beta, C,
                 *ldc);
}


