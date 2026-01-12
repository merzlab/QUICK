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
#if defined(__GNUC__) || defined (__PGIC__)
  #include <stdint.h>
  #include <assert.h>
#endif /* __GNUC__ */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "quick_cusolver.h"
#include "cusolver_fortran_common.h"


#define imax(a,b) (((a) < (b)) ? (b) : (a))


#define CUBLAS_WRAPPER_ERROR_NOERR (0)
#define CUBLAS_WRAPPER_ERROR_ALLOC (1)
#define CUBLAS_WRAPPER_ERROR_SET (2)
#define CUBLAS_WRAPPER_ERROR_GET (3)
#define CUBLAS_WRAPPER_ERROR_STUB (4)


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


void Fock_DIAG (double* o, const double* x, const double* E, const double* vec, const int* nbasis)
{
    cudaError_t err1, err2, err3;
    cublasStatus_t stat1, stat2, stat3;

    double* devPtr_o = NULL;
    double* devPtr_x = NULL;
    double* devPtr_hold = NULL;

    if (*nbasis == 0) return;

    int dim = *nbasis;

    err1 = cudaMalloc((void**)&devPtr_o, sizeof(double) * imax(1, dim * dim));
    err2 = cudaMalloc((void**)&devPtr_x, sizeof(double) * imax(1, dim * dim));
    err3 = cudaMalloc((void**)&devPtr_hold, sizeof(double) * dim * dim);

    if ((err1 != cudaSuccess)
            || (err2 != cudaSuccess)
            || (err3 != cudaSuccess)) {
        fprintf(stderr, "cudaMalloc failed in Fock_DIAG\n");
        cudaFree(devPtr_o);
        cudaFree(devPtr_x);
        cudaFree(devPtr_hold);
        return;
    }

    err1 = cudaMemcpy(devPtr_o, o, sizeof(double)*dim*dim, cudaMemcpyHostToDevice);
    err2 = cudaMemcpy(devPtr_x, x, sizeof(double)*dim*dim, cudaMemcpyHostToDevice);

    if ((err1 != cudaSuccess)
            || (err2 != cudaSuccess)) {
        fprintf(stderr, "cudaMemcpyHostToDevice cudaMemcpy failed in Fock_DIAG\n");
        cudaFree(devPtr_o);
        cudaFree(devPtr_x);
        cudaFree(devPtr_hold);
        return;
    }

    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cublas_status = cublasCreate_v2(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);


    const double h_one = 1;
    const double h_zero = 0;

    // hold = o * x
    cublasDgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &h_one, devPtr_o, dim,
            devPtr_x, dim, &h_zero, devPtr_hold, dim);

    // o = x * hold
    cublasDgemm_v2(cublasH, CUBLAS_OP_N,CUBLAS_OP_N, dim, dim, dim, &h_one, devPtr_x, dim,
            devPtr_hold, dim, &h_zero, devPtr_o, dim);

    err3 = cudaMemcpy(o, devPtr_o, sizeof(double)*dim*dim, cudaMemcpyDeviceToHost);

    if (err3 != cudaSuccess){
        fprintf(stderr, "cudaMemcpyDeviceToHost cudaMemcpy failed in Fock_DIAG\n");
    };

    CUDA_DIAG (o, E, vec, nbasis);

    if (devPtr_o) cudaFree(devPtr_o);
    if (devPtr_x) cudaFree(devPtr_x);
    if (devPtr_hold) cudaFree(devPtr_hold);
    if (cublasH) cublasDestroy_v2(cublasH);
}

void CUDA_DIAG (double* M, const double* E, const double* vec, const int* nbasis)
{
    if (nbasis == 0) return;

    int dim = *nbasis;

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    //Step 1: create cusolver handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    //Step 2: Copy arrays to device
    double* devPtr_M = NULL;
    cudaStat1 = cudaMalloc((void**)&devPtr_M, sizeof(double) * dim * dim);
    if (cudaStat1 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed in CUDA_DIAG\n");
        cudaFree(devPtr_M);
        return;
    }
    cudaStat1 = cudaMemcpy(devPtr_M, M, sizeof(double)*dim*dim, cudaMemcpyHostToDevice);
    if (cudaStat1 != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed in CUDA_DIAG\n");
        cudaFree(devPtr_M);
        return;
    }

    double* devPtr_E = NULL;
    cudaStat2 = cudaMalloc((void**)&devPtr_E, sizeof(double) * dim);
    assert(cudaSuccess == cudaStat2);

    int* devPtr_devInfo = NULL;
    cudaStat3 = cudaMalloc((void**)&devPtr_devInfo, sizeof(double));
    assert(cudaSuccess == cudaStat3);

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;

    // Query the workspace for work buffer size
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz,
            uplo, dim, devPtr_M, dim, devPtr_E, &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    double* devPtr_work = NULL;

    // Allocate work space
    cudaStat4 = cudaMalloc((void**)&devPtr_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat4);

    // Compute Spectrum
    cusolver_status = cusolverDnDsyevd(cusolverH, jobz,
            uplo, dim, devPtr_M, dim, devPtr_E, devPtr_work, lwork, devPtr_devInfo);
    cudaStat2 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat2);

    //kwk

    cudaStat3 = cudaMemcpy((void*) vec, devPtr_M, sizeof(double)*dim*dim, cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy((void*) E, devPtr_E, sizeof(double)*dim, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat3 && cudaSuccess == cudaStat4);

    if (devPtr_M) cudaFree(devPtr_M);
    if (devPtr_E) cudaFree(devPtr_E);
    if (devPtr_devInfo) cudaFree(devPtr_devInfo);
    if (devPtr_work) cudaFree(devPtr_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
}
