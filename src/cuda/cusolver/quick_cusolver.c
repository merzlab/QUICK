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
#include <assert.h>
#endif /* __GNUC__ */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "quick_cusolver.h"
#include "cusolver_fortran_common.h"




#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)<(b))?(b):(a))


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

    stat1=cublasAlloc (imax(1, dim*dim), sizeof(devPtr_o), (void**)&devPtr_o);
    stat2=cublasAlloc (imax(1, dim*dim), sizeof(devPtr_x), (void**)&devPtr_x);
    stat3=cublasAlloc (dim*dim, sizeof(devPtr_hold), (void**)&devPtr_hold);

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

    const double h_one = 1;
    const double h_zero = 0;

      // hold = o * x

            cublasDgemm_v2 (cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &h_one, devPtr_o, dim,
		      devPtr_x, dim, &h_zero, devPtr_hold, dim);

    // o = x * hold
      cublasDgemm_v2 (cublasH, CUBLAS_OP_N,CUBLAS_OP_N, dim, dim, dim, &h_one, devPtr_x, dim,
		      devPtr_hold, dim, &h_zero, devPtr_o, dim);


      //retrieve output matrix:	stat1=cublasGetMatrix(dim, dim, sizeof(o[0]), devPtr_o, dim, o, dim);
      //
      //
      //      stat1=cublasGetMatrix(dim, dim, sizeof(hold[0]), devPtr_hold, dim, hold, dim);

	      cusolverDnHandle_t cusolverH = NULL;
	    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	    cudaError_t cudaStat1 = cudaSuccess;
	    cudaError_t cudaStat2 = cudaSuccess;
	    cudaError_t cudaStat3 = cudaSuccess;

	    //Step 1: create cusolver handle
	    cusolver_status = cusolverDnCreate(&cusolverH);

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

    double* devPtr_work = NULL;

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

    //kwk

	//printf("started diag copy\n");    
        cudaStat1 = cudaMemcpy((void*) vec, devPtr_o, sizeof(double)*dim*dim, cudaMemcpyDeviceToHost);
        cudaStat2 = cudaMemcpy((void*) E, devPtr_E, sizeof(double)*dim, cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat1 && cudaSuccess == cudaStat2);
	//printf("finished diag copy\n");	


    //stat1=cublasGetMatrix(dim, dim, sizeof(vec[0]), devPtr_o, dim, vec, dim);


			  

    /*
    stat1=cublasGetMatrix(dim, dim, sizeof(o[0]), devPtr_o, dim, o, dim);
    printf("o\n");
    for(int i=0; i<dim*dim; i++)
      {
	printf("%d %f\n", i, o[i]);
      }

    stat1=cublasGetMatrix(dim, dim, sizeof(o[0]), devPtr_E, dim, o, dim);
    printf("E\n");
    for(int i=0; i<dim*dim; i++)
      {
	printf("%d %f\n", i, o[i]);
      }

    stat1=cublasGetMatrix(dim, dim, sizeof(o[0]), devPtr_work, dim, o, dim);
    printf("work\n");
    for(int i=0; i<dim*dim; i++)
      {
	printf("%d %f\n", i, o[i]);
      }    
    
    */


    if (devPtr_o) cudaFree(devPtr_o);
    if (devPtr_x) cudaFree(devPtr_x);
    if (devPtr_hold) cudaFree(devPtr_hold);
    if (devPtr_E) cudaFree(devPtr_E);
    if (devPtr_devInfo) cudaFree(devPtr_devInfo);
    if (devPtr_work) cudaFree(devPtr_work);


    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (cublasH) cublasDestroy(cublasH);

    //    cudaDeviceReset();
        
}
