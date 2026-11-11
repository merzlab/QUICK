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

/*
 * CUDA_DIAG -- symmetric eigen-decomposition on the GPU via cuSOLVER.
 *
 * Fortran-callable (see cusolver_fortran_common.h for the name mangling); the
 * only caller is MAT_DIAG in src/subs/mat_diag.f90, which dispatches here for
 * CUDA/CUDA_MPIV builds.
 *
 * The trailing *info is an error flag, returned so that MAT_DIAG's CUDA branch
 * matches the magmaDIAG/rocDIAG/CPU_DIAG branches, all of which already report
 * one.  Values follow the LAPACK/cuSOLVER convention:
 *
 *    0   success
 *  > 0   cusolverDnDsyevd did not converge: that many off-diagonal elements of
 *        the intermediate tridiagonal form failed to converge to zero.  NOTE
 *        that cuSOLVER still returns CUSOLVER_STATUS_SUCCESS in this case, so
 *        this condition is ONLY visible through devInfo.  Until this was
 *        checked, a non-converged solve returned partially garbage eigenvalues
 *        silently -- which is exactly the failure mode that is expected to be
 *        reachable for the large per-subsystem matrices of a DnC run (the GPU
 *        regression suite's DIVCON tests are far too small to reach it).
 *   -1   cuSOLVER handle creation failed
 *   -2   device allocation failed
 *   -3   host<->device copy failed
 *   -4   cusolverDnDsyevd_bufferSize failed
 *   -5   cusolverDnDsyevd returned a non-success status
 *   -6   cudaDeviceSynchronize after the solve failed
 *   -7   devInfo < 0: illegal argument passed to cusolverDnDsyevd
 *
 * On any failure the caller's eval/evec are zero-filled rather than left at
 * whatever the freshly allocated Fortran arrays happened to contain, so that a
 * caller which ignores *info can never consume uninitialised memory.
 */

/* Report free/total device memory alongside an allocation failure. */
static void cuda_diag_report_mem (const char * what, int dim, size_t bytes,
        cudaError_t stat)
{
    size_t diag_free = 0, diag_total = 0;
    cudaError_t meminfo_stat = cudaMemGetInfo(&diag_free, &diag_total);

    fprintf(stderr, "CUDA_DIAG: %s failed: dim=%d bytes=%zu error=%s (code %d)\n",
            what, dim, bytes, cudaGetErrorString(stat), (int)stat);
    if (meminfo_stat == cudaSuccess) {
        fprintf(stderr, "CUDA_DIAG:   at failure: free=%zu MB total=%zu MB\n",
                diag_free / (1024*1024), diag_total / (1024*1024));
    }
    fflush(stderr);
}

void CUDA_DIAG (double * o, const int * dim1, const int * dim2, double * E,
        double * vec, int * info)
{
    *info = 0;

    if (*dim1 == 0 || *dim2 == 0) return;

    int dim = *dim1;
    size_t nmat = (size_t)dim * (size_t)dim;

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;

    double* devPtr_M = NULL;
    double* devPtr_E = NULL;
    double* devPtr_work = NULL;
    int* devPtr_devInfo = NULL;

    int hostInfo = 0;
    int lwork = 0;

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    //Step 1: create cusolver handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA_DIAG: cusolverDnCreate failed (status %d)\n",
                (int)cusolver_status);
        fflush(stderr);
        *info = -1;
        goto cleanup;
    }

    //Step 2: Copy arrays to device
    cudaStat = cudaMalloc((void**)&devPtr_M, sizeof(double) * nmat);
    if (cudaStat != cudaSuccess) {
        cuda_diag_report_mem("cudaMalloc(matrix)", dim, sizeof(double) * nmat, cudaStat);
        *info = -2;
        goto cleanup;
    }

    cudaStat = cudaMemcpy(devPtr_M, o, sizeof(double) * nmat, cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA_DIAG: cudaMemcpy(matrix, H2D) failed: %s\n",
                cudaGetErrorString(cudaStat));
        fflush(stderr);
        *info = -3;
        goto cleanup;
    }

    cudaStat = cudaMalloc((void**)&devPtr_E, sizeof(double) * dim);
    if (cudaStat != cudaSuccess) {
        cuda_diag_report_mem("cudaMalloc(eigenvalues)", dim, sizeof(double) * dim, cudaStat);
        *info = -2;
        goto cleanup;
    }

    cudaStat = cudaMalloc((void**)&devPtr_devInfo, sizeof(int));
    if (cudaStat != cudaSuccess) {
        cuda_diag_report_mem("cudaMalloc(devInfo)", dim, sizeof(int), cudaStat);
        *info = -2;
        goto cleanup;
    }

    // Query the workspace for work buffer size
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz,
            uplo, dim, devPtr_M, dim, devPtr_E, &lwork);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA_DIAG: cusolverDnDsyevd_bufferSize failed (status %d, dim=%d)\n",
                (int)cusolver_status, dim);
        fflush(stderr);
        *info = -4;
        goto cleanup;
    }

    // Allocate work space
    cudaStat = cudaMalloc((void**)&devPtr_work, sizeof(double) * (size_t)lwork);
    if (cudaStat != cudaSuccess) {
        cuda_diag_report_mem("cudaMalloc(workspace)", dim,
                sizeof(double) * (size_t)lwork, cudaStat);
        *info = -2;
        goto cleanup;
    }

    // Compute Spectrum
    cusolver_status = cusolverDnDsyevd(cusolverH, jobz,
            uplo, dim, devPtr_M, dim, devPtr_E, devPtr_work, lwork, devPtr_devInfo);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA_DIAG: cusolverDnDsyevd failed (status %d, dim=%d)\n",
                (int)cusolver_status, dim);
        fflush(stderr);
        *info = -5;
        goto cleanup;
    }

    cudaStat = cudaDeviceSynchronize();
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA_DIAG: cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(cudaStat));
        fflush(stderr);
        *info = -6;
        goto cleanup;
    }

    // cusolverDnDsyevd reports non-convergence ONLY through devInfo, while
    // still returning CUSOLVER_STATUS_SUCCESS.  Read it back and surface it.
    cudaStat = cudaMemcpy(&hostInfo, devPtr_devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA_DIAG: cudaMemcpy(devInfo, D2H) failed: %s\n",
                cudaGetErrorString(cudaStat));
        fflush(stderr);
        *info = -3;
        goto cleanup;
    }

    if (hostInfo != 0) {
        if (hostInfo < 0) {
            fprintf(stderr, "CUDA_DIAG: cusolverDnDsyevd reports illegal argument %d (dim=%d)\n",
                    -hostInfo, dim);
            *info = -7;
        } else {
            fprintf(stderr, "CUDA_DIAG: cusolverDnDsyevd did NOT converge: devInfo=%d of dim=%d "
                    "off-diagonal elements failed to converge; eigenvalues are unreliable\n",
                    hostInfo, dim);
            *info = hostInfo;
        }
        fflush(stderr);
        goto cleanup;
    }

    cudaStat = cudaMemcpy((void*) vec, devPtr_M, sizeof(double) * nmat, cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA_DIAG: cudaMemcpy(eigenvectors, D2H) failed: %s\n",
                cudaGetErrorString(cudaStat));
        fflush(stderr);
        *info = -3;
        goto cleanup;
    }

    cudaStat = cudaMemcpy((void*) E, devPtr_E, sizeof(double) * dim, cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "CUDA_DIAG: cudaMemcpy(eigenvalues, D2H) failed: %s\n",
                cudaGetErrorString(cudaStat));
        fflush(stderr);
        *info = -3;
        goto cleanup;
    }

cleanup:
    if (*info != 0) {
        // Never hand back uninitialised memory: a caller that ignores *info
        // would otherwise read whatever the Fortran allocate() left behind.
        size_t i;
        for (i = 0; i < (size_t)dim; i++) E[i] = 0.0;
        for (i = 0; i < nmat; i++) vec[i] = 0.0;
    }

    if (devPtr_M) cudaFree(devPtr_M);
    if (devPtr_E) cudaFree(devPtr_E);
    if (devPtr_devInfo) cudaFree(devPtr_devInfo);
    if (devPtr_work) cudaFree(devPtr_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
}
