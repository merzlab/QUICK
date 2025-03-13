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
 *  Fortran callable BLAS functions that include GPU memory allocation and
 *  copy-up and copy-down code. These can be called from unmodified Fortran 
 *  code, but they are inefficient due to the data constantly bouncing back 
 *  and forth between CPU and GPU.
 */


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
/* BLAS1 */
#if CUBLAS_FORTRAN_COMPILER==CUBLAS_G77
double CUBLAS_SDOT (const int *n, const float *x, const int *incx, float *y, 
                    const int *incy); 
double CUBLAS_SASUM (const int *n, const float *x, const int *incx);
double CUBLAS_SNRM2 (const int *n, const float *x, const int *incx);
double CUBLAS_SCASUM (const int *n, const cuComplex *x, const int *incx);
double CUBLAS_SCNRM2 (const int *n, const cuComplex *x, const int *incx);
#else
float CUBLAS_SDOT (const int *n, const float *x, const int *incx, float *y, 
                   const int *incy);
float CUBLAS_SASUM (const int *n, const float *x, const int *incx);
float CUBLAS_SNRM2 (const int *n, const float *x, const int *incx);
float CUBLAS_SCASUM (const int *n, const cuComplex *x, const int *incx);
float CUBLAS_SCNRM2 (const int *n, const cuComplex *x, const int *incx);
#endif
double CUBLAS_DZNRM2 (const int *n, const cuDoubleComplex *x, const int *incx);
double CUBLAS_DZASUM (const int *n, const cuDoubleComplex *x, const int *incx);

int CUBLAS_ISAMAX (const int *n, const float *x, const int *incx);
int CUBLAS_ISAMIN (const int *n, const float *x, const int *incx);
void CUBLAS_SAXPY (const int *n, const float *alpha, const float *x, 
                   const int *incx, float *y, const int *incy);
void CUBLAS_SCOPY (const int *n, const float *x, const int *incx, float *y, 
                   const int *incy);
void CUBLAS_SROT (const int *n, float *x, const int *incx, float *y, 
                  const int *incy, const float *sc, const float *ss);
void CUBLAS_SROTG (float *sa, float *sb, float *sc, float *ss);
void CUBLAS_SROTM (const int *n, float *x, const int *incx, float *y,
                   const int *incy, const float* sparam);
void CUBLAS_SROTMG (float *sd1, float *sd2, float *sx1, const float *sy1, 
                    float* sparam);
void CUBLAS_SSCAL (const int *n, const float *alpha, float *x,
                   const int *incx);
void CUBLAS_SSWAP(const int *n, float *x, const int *incx, float *y,
                  const int *incy);

void CUBLAS_CAXPY (const int *n, const cuComplex *alpha, const cuComplex *x, 
                   const int *incx, cuComplex *y, const int *incy);
void CUBLAS_ZAXPY (const int *n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, 
                   const int *incx, cuDoubleComplex *y, const int *incy);
void CUBLAS_CCOPY (const int *n, const cuComplex *x, const int *incx, 
                   cuComplex *y, const int *incy);
void CUBLAS_ZCOPY (const int *n, const cuDoubleComplex *x, const int *incx, 
                   cuDoubleComplex *y, const int *incy);
void CUBLAS_CROT (const int *n, cuComplex *x, const int *incx, cuComplex *y, 
                  const int *incy, const float *sc, const cuComplex *cs);
void CUBLAS_ZROT (const int *n, cuDoubleComplex *x, const int *incx, cuDoubleComplex *y, 
                  const int *incy, const double *sc, const cuDoubleComplex *cs);                  
void CUBLAS_CROTG (cuComplex *ca, const cuComplex *cb, float *sc, 
                   cuComplex *cs);
void CUBLAS_ZROTG (cuDoubleComplex *ca, const cuDoubleComplex *cb, double *sc, 
                   cuDoubleComplex *cs);                   
void CUBLAS_CSCAL (const int *n, const cuComplex *alpha, cuComplex *x, 
                   const int *incx);
void CUBLAS_CSROT (const int *n, cuComplex *x, const int *incx, cuComplex *y, 
                   const int *incy, const float *sc, const float *ss);                   
void CUBLAS_ZDROT (const int *n, cuDoubleComplex *x, const int *incx, cuDoubleComplex *y, 
                   const int *incy, const double *sc, const double *ss);                   
void CUBLAS_CSSCAL (const int *n, const float *alpha, cuComplex *x,
                    const int *incx);                   
void CUBLAS_CSWAP (const int *n, cuComplex *x, const int *incx, cuComplex *y,
                   const int *incy);
void CUBLAS_ZSWAP (const int *n, cuDoubleComplex *x, const int *incx, cuDoubleComplex *y,
                   const int *incy);
void CUBLAS_CTRMV (const char *uplo, const char *trans, const char *diag, const int *n, 
                   const cuComplex *A, const int *lda, cuComplex *x, const int *incx);
void CUBLAS_ZTRMV (const char *uplo, const char *trans, const char *diag, const int *n, 
                   const cuDoubleComplex *A, const int *lda, cuDoubleComplex *x, const int *incx);                
#ifdef RETURN_COMPLEX                                     
cuComplex CUBLAS_CDOTU ( const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy);
cuComplex CUBLAS_CDOTC ( const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy);
#else
void CUBLAS_CDOTU (cuComplex *retVal, const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy);
void CUBLAS_CDOTC (cuComplex *retVal,const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy);
#endif                   
int CUBLAS_ICAMAX (const int *n, const cuComplex *x, const int *incx);
int CUBLAS_ICAMIN (const int *n, const cuComplex *x, const int *incx);
int CUBLAS_IZAMAX (const int *n, const cuDoubleComplex *x, const int *incx);
int CUBLAS_IZAMIN (const int *n, const cuDoubleComplex *x, const int *incx);

/* BLAS2 */
void CUBLAS_SGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const float *alpha, 
                   const float *A, const int *lda, const float *x,
                   const int *incx, const float *beta, float *y, 
                   const int *incy);
void CUBLAS_DGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const double *alpha, 
                   const double *A, const int *lda, const double *x,
                   const int *incx, const double *beta, double *y, 
                   const int *incy);
void CUBLAS_CGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const cuComplex *alpha, 
                   const cuComplex *A, const int *lda, const cuComplex *x,
                   const int *incx, const cuComplex *beta, cuComplex *y, 
                   const int *incy);
void CUBLAS_ZGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, const cuDoubleComplex *alpha, 
                   const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *x,
                   const int *incx, const cuDoubleComplex *beta, cuDoubleComplex *y, 
                   const int *incy);
                                                                            
void CUBLAS_SGEMV (const char *trans, const int *m, const int *n,
                   const float *alpha, const float *A, const int *lda,
                   const float *x, const int *incx, const float *beta, 
                   float *y, const int *incy);
void CUBLAS_SGER (const int *m, const int *n, const float *alpha,
                  const float *x, const int *incx, const float *y,
                  const int *incy, float *A, const int *lda);
void CUBLAS_SSBMV (const char *uplo, const int *n, const int *k, 
                   const float *alpha, const float *A, const int *lda,
                   const float *x, const int *incx, const float *beta,
                   float *y, const int *incy);
void CUBLAS_DSBMV (const char *uplo, const int *n, const int *k, 
                   const double *alpha, const double *A, const int *lda,
                   const double *x, const int *incx, const double *beta,
                   double *y, const int *incy);
void CUBLAS_CHBMV (const char *uplo, const int *n, const int *k, 
                   const cuComplex *alpha, const cuComplex *A, const int *lda,
                   const cuComplex *x, const int *incx, const cuComplex *beta,
                   cuComplex *y, const int *incy);
void CUBLAS_ZHBMV (const char *uplo, const int *n, const int *k, 
                   const cuDoubleComplex *alpha, const cuDoubleComplex *A, const int *lda,
                   const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *beta,
                   cuDoubleComplex *y, const int *incy);                                                         
void CUBLAS_SSPMV (const char *uplo, const int *n, const float *alpha, 
                   const float *AP, const float *x, const int *incx, 
                   const float *beta, float *y, const int *incy);
void CUBLAS_DSPMV (const char *uplo, const int *n, const double *alpha, 
                   const double *AP, const double *x, const int *incx, 
                   const double *beta, double *y, const int *incy);
void CUBLAS_CHPMV (const char *uplo, const int *n, const cuComplex *alpha, 
                   const cuComplex *AP, const cuComplex *x, const int *incx, 
                   const cuComplex *beta, cuComplex *y, const int *incy);
void CUBLAS_ZHPMV (const char *uplo, const int *n, const cuDoubleComplex *alpha, 
                   const cuDoubleComplex *AP, const cuDoubleComplex *x, const int *incx, 
                   const cuDoubleComplex *beta, cuDoubleComplex *y, const int *incy);                                                         
void CUBLAS_SSPR (const char *uplo, const int *n, const float *alpha,
                  const float *x, const int *incx, float *AP);
void CUBLAS_DSPR (const char *uplo, const int *n, const double *alpha,
                  const double *x, const int *incx, double *AP);
void CUBLAS_CHPR (const char *uplo, const int *n, const float *alpha,
                  const cuComplex *x, const int *incx, cuComplex *AP);
void CUBLAS_ZHPR (const char *uplo, const int *n, const double *alpha,
                  const cuDoubleComplex *x, const int *incx, cuDoubleComplex *AP);                                                      
void CUBLAS_SSPR2 (const char *uplo, const int *n, const float *alpha,
                   const float *x, const int *incx, const float *y,
                   const int *incy, float *AP);
void CUBLAS_DSPR2 (const char *uplo, const int *n, const double *alpha,
                   const double *x, const int *incx, const double *y,
                   const int *incy, double *AP);
void CUBLAS_CHPR2 (const char *uplo, const int *n, const cuComplex *alpha,
                   const cuComplex *x, const int *incx, const cuComplex *y,
                   const int *incy, cuComplex *AP);
void CUBLAS_ZHPR2 (const char *uplo, const int *n, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *y,
                   const int *incy, cuDoubleComplex *AP);                                                         
void CUBLAS_SSYMV (const char *uplo, const int *n, const float *alpha,
                   const float *A, const int *lda, const float *x,
                   const int *incx, const float *beta, float *y, 
                   const int *incy);
void CUBLAS_DSYMV (const char *uplo, const int *n, const double *alpha,
                   const double *A, const int *lda, const double *x,
                   const int *incx, const double *beta, double *y, 
                   const int *incy);
void CUBLAS_CHEMV (const char *uplo, const int *n, const cuComplex *alpha,
                   const cuComplex *A, const int *lda, const cuComplex *x,
                   const int *incx, const cuComplex *beta, cuComplex *y, 
                   const int *incy);
void CUBLAS_ZHEMV (const char *uplo, const int *n, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *x,
                   const int *incx, const cuDoubleComplex *beta, cuDoubleComplex *y, 
                   const int *incy);                                                         
void CUBLAS_SSYR (const char *uplo, const int *n, const float *alpha,
                  const float *x, const int *incx, float *A, const int *lda);
void CUBLAS_SSYR2 (const char *uplo, const int *n, const float *alpha,
                   const float *x, const int *incx, const float *y,
                   const int *incy, float *A, const int *lda);
void CUBLAS_DSYR2 (const char *uplo, const int *n, const double *alpha,
                   const double *x, const int *incx, const double *y,
                   const int *incy, double *A, const int *lda);  
void CUBLAS_CHER2 (const char *uplo, const int *n, const cuComplex *alpha,
                   const cuComplex *x, const int *incx, const cuComplex *y,
                   const int *incy, cuComplex *A, const int *lda); 
void CUBLAS_ZHER2 (const char *uplo, const int *n, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *y,
                   const int *incy, cuDoubleComplex *A, const int *lda);                                                        
void CUBLAS_STBMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const int *k, const float *A, const int *lda, 
                   float *x, const int *incx);
void CUBLAS_DTBMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const int *k, const double *A, const int *lda, 
                   double *x, const int *incx);
void CUBLAS_CTBMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const int *k, const cuComplex *A, const int *lda, 
                   cuComplex *x, const int *incx);
void CUBLAS_ZTBMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const int *k, const cuDoubleComplex *A, const int *lda, 
                   cuDoubleComplex *x, const int *incx);                                                         
void CUBLAS_STBSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const int *k, const float *A, const int *lda, 
                   float *x, const int *incx);
void CUBLAS_DTBSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const int *k, const double *A, const int *lda, 
                   double *x, const int *incx);
void CUBLAS_CTBSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const int *k, const cuComplex *A, const int *lda, 
                   cuComplex *x, const int *incx);
void CUBLAS_ZTBSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const int *k, const cuDoubleComplex *A, const int *lda, 
                   cuDoubleComplex *x, const int *incx);                                                         
void CUBLAS_STPMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const float *AP, float *x, const int *incx);
void CUBLAS_DTPMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const double *AP, double *x, const int *incx);
void CUBLAS_CTPMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const cuComplex *AP, cuComplex *x, const int *incx);
void CUBLAS_ZTPMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const cuDoubleComplex *AP, cuDoubleComplex *x, const int *incx);                                                         
void CUBLAS_STPSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const float *AP, float *x, const int *incx);
void CUBLAS_DTPSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const double *AP, double *x, const int *incx);
void CUBLAS_CTPSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const cuComplex *AP, cuComplex *x, const int *incx);
void CUBLAS_ZTPSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const cuDoubleComplex *AP, cuDoubleComplex *x, const int *incx);                                                         
void CUBLAS_STRMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const float *A, const int *lda, float *x, 
                   const int *incx);
void CUBLAS_STRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const float *A, const int *lda, float *x, 
                   const int *incx);
void CUBLAS_CTRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const cuComplex *A, const int *lda, cuComplex *x, 
                   const int *incx);                   

/* BLAS3 */
void CUBLAS_SGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const float *alpha, 
                   const float *A, const int *lda, const float *B, 
                   const int *ldb, const float *beta, float *C, 
                   const int *ldc);
void CUBLAS_SSYMM (const char *side, const char *uplo, const int *m,
                   const int *n, const float *alpha, const float *A,
                   const int *lda, const float *B, const int *ldb, 
                   const float *beta, float *C, const int *ldc);
void CUBLAS_SSYR2K (const char *uplo, const char *trans, const int *n, 
                    const int *k, const float *alpha, const float *A,
                    const int *lda,const float *B, const int *ldb, 
                    const float *beta, float *C, const int *ldc);
void CUBLAS_SSYRK (const char *uplo, const char *trans, const int *n,
                   const int *k, const float *alpha, const float *A,
                   const int *lda, const float *beta, float *C,
                   const int *ldc);
void CUBLAS_STRMM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n,
                   const float *alpha, const float *A, const int *lda,
                   float *B, const int *ldb);
void CUBLAS_CTRMM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n,
                   const cuComplex *alpha, const cuComplex *A, const int *lda,
                   cuComplex *B, const int *ldb);                   
void CUBLAS_STRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n,
                   const float *alpha, const float *A, const int *lda,
                   float *B, const int *ldb);

void CUBLAS_CGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const cuComplex *alpha,
                   const cuComplex *A, const int *lda, const cuComplex *B,
                   const int *ldb, const cuComplex *beta, cuComplex *C, 
                   const int *ldc);

/* DP BLAS1 */
double CUBLAS_DDOT (const int *n, const double *x, const int *incx, double *y, 
                   const int *incy);
double CUBLAS_DASUM (const int *n, const double *x, const int *incx);
double CUBLAS_DNRM2 (const int *n, const double *x, const int *incx);
int CUBLAS_IDAMAX (const int *n, const double *x, const int *incx);
int CUBLAS_IDAMIN (const int *n, const double *x, const int *incx);
void CUBLAS_DAXPY (const int *n, const double *alpha, const double *x, 
                   const int *incx, double *y, const int *incy);
void CUBLAS_DCOPY (const int *n, const double *x, const int *incx, double *y, 
                   const int *incy);
void CUBLAS_DROT (const int *n, double *x, const int *incx, double *y, 
                  const int *incy, const double *sc, const double *ss);
void CUBLAS_DROTG (double *sa, double *sb, double *sc, double *ss);
void CUBLAS_DROTM (const int *n, double *x, const int *incx, double *y,
                   const int *incy, const double* sparam);
void CUBLAS_DROTMG (double *sd1, double *sd2, double *sx1, const double *sy1, 
                    double* sparam);
void CUBLAS_DSCAL (const int *n, const double *alpha, double *x,
                   const int *incx);
void CUBLAS_DSWAP(const int *n, double *x, const int *incx, double *y,
                  const int *incy);

/* DP Complex BLAS1 */
#ifdef RETURN_COMPLEX
cuDoubleComplex CUBLAS_ZDOTU ( const int *n, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *y, const int *incy);
cuDoubleComplex CUBLAS_ZDOTC ( const int *n, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *y, const int *incy);                   
#else
void CUBLAS_ZDOTU (cuDoubleComplex *retVal, const int *n, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *y, const int *incy);
void CUBLAS_ZDOTC (cuDoubleComplex *retVal, const int *n, const cuDoubleComplex *x, 
                   const int *incx, const cuDoubleComplex *y, const int *incy);                   

#endif                   
void CUBLAS_ZSCAL (const int *n, const cuDoubleComplex *alpha, cuDoubleComplex *x, 
                   const int *incx);
void CUBLAS_ZDSCAL (const int *n, const double *alpha, cuDoubleComplex *x, 
                   const int *incx);                   

/* DP BLAS2 */
void CUBLAS_DGEMV (const char *trans, const int *m, const int *n,
                   const double *alpha, const double *A, const int *lda,
                   const double *x, const int *incx, const double *beta, 
                   double *y, const int *incy);
void CUBLAS_DGER (const int *m, const int *n, const double *alpha,
                  const double *x, const int *incx, const double *y,
                  const int *incy, double *A, const int *lda);
void CUBLAS_DSYR (const char *uplo, const int *n, const double *alpha,
                  const double *x, const int *incx, double *A, const int *lda);
void CUBLAS_DTRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const double *A, const int *lda, double *x, 
                   const int *incx);
void CUBLAS_DTRMV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const double *A, const int *lda, double *x, 
                   const int *incx);

/* DP Complex BLAS2 */
void CUBLAS_CGEMV (const char *trans, const int *m, const int *n,
                   const cuComplex *alpha, const cuComplex *A,
                   const int *lda, const cuComplex *x, const int *incx,
                   const cuComplex *beta, cuComplex *y,
                   const int *incy);
void CUBLAS_ZGEMV (const char *trans, const int *m, const int *n,
                   const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                   const int *lda, const cuDoubleComplex *x, const int *incx,
                   const cuDoubleComplex *beta, cuDoubleComplex *y,
                   const int *incy);
void CUBLAS_CSYMM (const char *side, const char *uplo, const int *m,
                   const int *n, const cuComplex *alpha, const cuComplex *A,
                   const int *lda, const cuComplex *B, const int *ldb, 
                   const cuComplex *beta, cuComplex *C, const int *ldc);
                   
void CUBLAS_CHEMM (const char *side, const char *uplo, const int *m,
                   const int *n, const cuComplex *alpha, const cuComplex *A,
                   const int *lda, const cuComplex *B, const int *ldb, 
                   const cuComplex *beta, cuComplex *C, const int *ldc);                   
                   
void CUBLAS_CGERU (const int *m, const int *n, const cuComplex *alpha,
                  const cuComplex *x, const int *incx, const cuComplex *y,
                  const int *incy, cuComplex *A, const int *lda);  
                  
void CUBLAS_CGERC (const int *m, const int *n, const cuComplex *alpha,
                  const cuComplex *x, const int *incx, const cuComplex *y,
                  const int *incy, cuComplex *A, const int *lda);  
void CUBLAS_ZGERU (const int *m, const int *n, const cuDoubleComplex *alpha,
                  const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *y,
                  const int *incy, cuDoubleComplex *A, const int *lda);  
                  
void CUBLAS_ZGERC (const int *m, const int *n, const cuDoubleComplex *alpha,
                  const cuDoubleComplex *x, const int *incx, const cuDoubleComplex *y,
                  const int *incy, cuDoubleComplex *A, const int *lda);                    
                                                     
void CUBLAS_ZTRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const cuDoubleComplex *A, const int *lda, 
                   cuDoubleComplex *x, const int *incx); 

void CUBLAS_CHER (const char *uplo, const int *n, const float *alpha,
                  const cuComplex *x, const int *incx, cuComplex *A, const int *lda);   
                  
void CUBLAS_ZHER (const char *uplo, const int *n, const double *alpha,
                  const cuDoubleComplex *x, const int *incx, cuDoubleComplex *A, const int *lda);                                     
                   
/* DP BLAS3 */
void CUBLAS_DGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const double *alpha, 
                   const double *A, const int *lda, const double *B, 
                   const int *ldb, const double *beta, double *C, 
                   const int *ldc);
void CUBLAS_DSYMM (const char *side, const char *uplo, const int *m,
                   const int *n, const double *alpha, const double *A,
                   const int *lda, const double *B, const int *ldb, 
                   const double *beta, double *C, const int *ldc);
void CUBLAS_ZSYMM (const char *side, const char *uplo, const int *m,
                   const int *n, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                   const int *lda, const cuDoubleComplex *B, const int *ldb, 
                   const cuDoubleComplex *beta, cuDoubleComplex *C, const int *ldc);
                   
void CUBLAS_ZHEMM (const char *side, const char *uplo, const int *m,
                   const int *n, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                   const int *lda, const cuDoubleComplex *B, const int *ldb, 
                   const cuDoubleComplex *beta, cuDoubleComplex *C, const int *ldc);                   
                                      
void CUBLAS_DSYR2K (const char *uplo, const char *trans, const int *n, 
                    const int *k, const double *alpha, const double *A,
                    const int *lda,const double *B, const int *ldb, 
                    const double *beta, double *C, const int *ldc);
void CUBLAS_DSYRK (const char *uplo, const char *trans, const int *n,
                   const int *k, const double *alpha, const double *A,
                   const int *lda, const double *beta, double *C,
                   const int *ldc);
void CUBLAS_DTRMM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n,
                   const double *alpha, const double *A, const int *lda,
                   double *B, const int *ldb);
void CUBLAS_ZTRMM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n,
                   const cuDoubleComplex *alpha, const cuDoubleComplex *A, const int *lda,
                   cuDoubleComplex *B, const int *ldb);                   
void CUBLAS_DTRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n,
                   const double *alpha, const double *A, const int *lda,
                   double *B, const int *ldb);
void CUBLAS_CTRSM ( const char *side, const char *uplo, const char *transa,
                    const char *diag, const int *m, const int *n,
                    const cuComplex *alpha, const cuComplex *A, const int *lda,
                    cuComplex *B, const int *ldb);                   
void CUBLAS_ZTRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n,
                   const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda,
                   cuDoubleComplex *B, const int *ldb);
void CUBLAS_CSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const cuComplex *alpha,
                   const cuComplex *A, const int *lda,
                   const cuComplex *beta, cuComplex *C, 
                   const int *ldc);   
void CUBLAS_CSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const cuComplex *alpha, const cuComplex *A, 
                    const int *lda, const cuComplex *B, const int *ldb, 
                    const cuComplex *beta, cuComplex *C, const int *ldc);                                   
void CUBLAS_ZSYRK (const char *uplo, const char *trans, const int *n,
                   const int *k, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda,
                   const cuDoubleComplex *beta, cuDoubleComplex *C,
                   const int *ldc);
void CUBLAS_ZSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, 
                    const int *lda, const cuDoubleComplex *B, const int *ldb, 
                    const cuDoubleComplex *beta, cuDoubleComplex *C, const int *ldc);                   
void CUBLAS_CHERK (const char *uplo, const char *trans, const int *n, 
                   const int *k, const float *alpha,
                   const cuComplex *A, const int *lda,
                   const float *beta, cuComplex *C, 
                   const int *ldc); 
void CUBLAS_CHER2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const cuComplex *alpha, const cuComplex *A, 
                    const int *lda, const cuComplex *B, const int *ldb, 
                    const float *beta, cuComplex *C, const int *ldc);                     
void CUBLAS_ZHERK (const char *uplo, const char *trans, const int *n,
                   const int *k, const double *alpha,
                   const cuDoubleComplex *A, const int *lda,
                   const double *beta, cuDoubleComplex *C,
                   const int *ldc);  
void CUBLAS_ZHER2K (const char *uplo, const char *trans, const int *n,
                    const int *k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, 
                    const int *lda, const cuDoubleComplex *B, const int *ldb, 
                    const double *beta, cuDoubleComplex *C, const int *ldc);                                                        

void CUBLAS_ZGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda, 
                   const cuDoubleComplex *B, const int *ldb, 
                   const cuDoubleComplex *beta, cuDoubleComplex *C, 
                   const int *ldc);

#if defined(__cplusplus)
}
#endif /* __cplusplus */
