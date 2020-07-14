/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  
 *
 * This software and the information contained herein is being provided 
 * under the terms and conditions of a Source Code License Agreement.     
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
 */
 
#define CUBLAS_G77              1
#define CUBLAS_INTEL_FORTRAN    2
#define CUBLAS_G95              3

/* Default to g77 on Linux, and Intel Fortran on Win32 */
#if defined(_WIN32)
#define CUBLAS_FORTRAN_COMPILER CUBLAS_INTEL_FORTRAN
#elif defined(__linux)
#define CUBLAS_FORTRAN_COMPILER CUBLAS_G95
#elif defined(__APPLE__)
#define CUBLAS_FORTRAN_COMPILER CUBLAS_G95
#define RETURN_COMPLEX   1
#else
#error unsupported platform
#endif

#if (CUBLAS_FORTRAN_COMPILER==CUBLAS_G77) || (CUBLAS_FORTRAN_COMPILER==CUBLAS_G95)
/* NOTE: Must use -fno-second-underscore when building Fortran source with g77
 *       g77 invocation may not use -fno-f2c, which forces different return 
 *       type conventions than the one used below
 */
#define CUBLAS_INIT             cublas_init_
#define CUBLAS_SHUTDOWN         cublas_shutdown_
#define CUBLAS_ALLOC            cublas_alloc_
#define CUBLAS_FREE             cublas_free_
#define CUBLAS_SET_VECTOR       cublas_set_vector_
#define CUBLAS_GET_VECTOR       cublas_get_vector_
#define CUBLAS_SET_MATRIX       cublas_set_matrix_
#define CUBLAS_GET_MATRIX       cublas_get_matrix_
#define CUBLAS_GET_ERROR        cublas_get_error_
#define CUBLAS_XERBLA           cublas_xerbla_
#define CUBLAS_ISAMAX           cublas_isamax_
#define CUBLAS_ISAMIN           cublas_isamin_
#define CUBLAS_SASUM            cublas_sasum_
#define CUBLAS_SAXPY            cublas_saxpy_
#define CUBLAS_SCOPY            cublas_scopy_
#define CUBLAS_SDOT             cublas_sdot_
#define CUBLAS_SNRM2            cublas_snrm2_
#define CUBLAS_SROT             cublas_srot_
#define CUBLAS_SROTG            cublas_srotg_
#define CUBLAS_SROTM            cublas_srotm_
#define CUBLAS_SROTMG           cublas_srotmg_
#define CUBLAS_SSCAL            cublas_sscal_
#define CUBLAS_SSWAP            cublas_sswap_
#define CUBLAS_CAXPY            cublas_caxpy_
#define CUBLAS_CCOPY            cublas_ccopy_
#define CUBLAS_CROT             cublas_crot_
#define CUBLAS_CROTG            cublas_crotg_
#define CUBLAS_CSCAL            cublas_cscal_
#define CUBLAS_CSROT            cublas_csrot_
#define CUBLAS_CSSCAL           cublas_csscal_
#define CUBLAS_CSWAP            cublas_cswap_
#define CUBLAS_CTRMV            cublas_ctrmv_
#define CUBLAS_CDOTU            cublas_cdotu_
#define CUBLAS_CDOTC            cublas_cdotc_
#define CUBLAS_ICAMAX           cublas_icamax_
#define CUBLAS_SCASUM           cublas_scasum_
#define CUBLAS_SCNRM2           cublas_scnrm2_
#define CUBLAS_SGBMV            cublas_sgbmv_
#define CUBLAS_SGEMV            cublas_sgemv_
#define CUBLAS_SGER             cublas_sger_
#define CUBLAS_SSBMV            cublas_ssbmv_
#define CUBLAS_SSPMV            cublas_sspmv_
#define CUBLAS_SSPR             cublas_sspr_
#define CUBLAS_SSPR2            cublas_sspr2_
#define CUBLAS_SSYMV            cublas_ssymv_
#define CUBLAS_SSYR             cublas_ssyr_
#define CUBLAS_SSYR2            cublas_ssyr2_
#define CUBLAS_STBMV            cublas_stbmv_
#define CUBLAS_STBSV            cublas_stbsv_
#define CUBLAS_STPMV            cublas_stpmv_
#define CUBLAS_STPSV            cublas_stpsv_
#define CUBLAS_STRMV            cublas_strmv_
#define CUBLAS_STRSV            cublas_strsv_
#define CUBLAS_SGEMM            cublas_sgemm_
#define CUBLAS_SSYMM            cublas_ssymm_
#define CUBLAS_SSYR2K           cublas_ssyr2k_
#define CUBLAS_SSYRK            cublas_ssyrk_
#define CUBLAS_STRMM            cublas_strmm_
#define CUBLAS_STRSM            cublas_strsm_
#define CUBLAS_CGEMM            cublas_cgemm_
#define CUBLAS_CHEMM            cublas_chemm_
#define CUBLAS_CSYMM            cublas_csymm_
#define CUBLAS_CTRMM            cublas_ctrmm_
#define CUBLAS_CTRSM            cublas_ctrsm_
#define CUBLAS_CHERK            cublas_cherk_
#define CUBLAS_CSYRK            cublas_csyrk_
#define CUBLAS_CHER2K           cublas_cher2k_
#define CUBLAS_CSYR2K           cublas_csyr2k_
#define CUBLAS_IDAMAX           cublas_idamax_
#define CUBLAS_IDAMIN           cublas_idamin_
#define CUBLAS_DASUM            cublas_dasum_
#define CUBLAS_DAXPY            cublas_daxpy_
#define CUBLAS_DCOPY            cublas_dcopy_
#define CUBLAS_DDOT             cublas_ddot_
#define CUBLAS_DNRM2            cublas_dnrm2_
#define CUBLAS_DROT             cublas_drot_
#define CUBLAS_DROTG            cublas_drotg_
#define CUBLAS_DROTM            cublas_drotm_
#define CUBLAS_DROTMG           cublas_drotmg_
#define CUBLAS_DSCAL            cublas_dscal_
#define CUBLAS_DSWAP            cublas_dswap_
#define CUBLAS_ZAXPY            cublas_zaxpy_
#define CUBLAS_ZCOPY            cublas_zcopy_
#define CUBLAS_ZROT             cublas_zrot_
#define CUBLAS_ZROTG            cublas_zrotg_
#define CUBLAS_ZSCAL            cublas_zscal_
#define CUBLAS_ZDROT            cublas_zdrot_
#define CUBLAS_ZDSCAL           cublas_zdscal_
#define CUBLAS_ZSWAP            cublas_zswap_
#define CUBLAS_ZDOTU            cublas_zdotu_
#define CUBLAS_ZDOTC            cublas_zdotc_
#define CUBLAS_IZAMAX           cublas_izamax_
#define CUBLAS_DZASUM           cublas_dzasum_
#define CUBLAS_DZNRM2           cublas_dznrm2_
#define CUBLAS_DGBMV            cublas_dgbmv_
#define CUBLAS_DGEMV            cublas_dgemv_
#define CUBLAS_ZGEMV            cublas_zgemv_
#define CUBLAS_DGER             cublas_dger_
#define CUBLAS_DSBMV            cublas_dsbmv_
#define CUBLAS_DSPMV            cublas_dspmv_
#define CUBLAS_DSPR             cublas_dspr_
#define CUBLAS_DSPR2            cublas_dspr2_
#define CUBLAS_DSYMV            cublas_dsymv_
#define CUBLAS_DSYR             cublas_dsyr_
#define CUBLAS_DSYR2            cublas_dsyr2_
#define CUBLAS_DTBMV            cublas_dtbmv_
#define CUBLAS_DTBSV            cublas_dtbsv_
#define CUBLAS_DTPMV            cublas_dtpmv_
#define CUBLAS_DTPSV            cublas_dtpsv_
#define CUBLAS_DTRMV            cublas_dtrmv_
#define CUBLAS_DTRSV            cublas_dtrsv_
#define CUBLAS_DGEMM            cublas_dgemm_
#define CUBLAS_DSYMM            cublas_dsymm_
#define CUBLAS_DSYR2K           cublas_dsyr2k_
#define CUBLAS_DSYRK            cublas_dsyrk_
#define CUBLAS_ZSYRK            cublas_zsyrk_
#define CUBLAS_DTRMM            cublas_dtrmm_
#define CUBLAS_DTRSM            cublas_dtrsm_
#define CUBLAS_ZGEMM            cublas_zgemm_
#define CUBLAS_ZHEMM            cublas_zhemm_
#define CUBLAS_ZSYMM            cublas_zsymm_
#define CUBLAS_ZTRMM            cublas_ztrmm_
#define CUBLAS_ZTRSM            cublas_ztrsm_
#define CUBLAS_ZHERK            cublas_zherk_
#define CUBLAS_ZSYRK            cublas_zsyrk_
#define CUBLAS_ZHER2K           cublas_zher2k_
#define CUBLAS_ZSYR2K           cublas_zsyr2k_

#define  CUBLAS_CGEMV           cublas_cgemv_
#define  CUBLAS_CGBMV           cublas_cgbmv_
#define  CUBLAS_CHEMV           cublas_chemv_
#define  CUBLAS_CHBMV           cublas_chbmv_
#define  CUBLAS_CHPMV           cublas_chpmv_
#define  CUBLAS_CTBMV           cublas_ctbmv_
#define  CUBLAS_CTPMV           cublas_ctpmv_
#define  CUBLAS_CTRSV           cublas_ctrsv_
#define  CUBLAS_CTBSV           cublas_ctbsv_
#define  CUBLAS_CTPSV           cublas_ctpsv_
#define  CUBLAS_CGERC           cublas_cgerc_
#define  CUBLAS_CGERU           cublas_cgeru_
#define  CUBLAS_CHPR            cublas_chpr_
#define  CUBLAS_CHPR2           cublas_chpr2_
#define  CUBLAS_CHER            cublas_cher_
#define  CUBLAS_CHER2           cublas_cher2_

// stubs for zblat2
#define CUBLAS_ZGBMV           cublas_zgbmv_
#define CUBLAS_ZHEMV           cublas_zhemv_
#define CUBLAS_ZHBMV           cublas_zhbmv_
#define CUBLAS_ZHPMV           cublas_zhpmv_
#define CUBLAS_ZTRMV           cublas_ztrmv_
#define CUBLAS_ZTBMV           cublas_ztbmv_
#define CUBLAS_ZTPMV           cublas_ztpmv_
#define CUBLAS_ZTRSV           cublas_ztrsv_
#define CUBLAS_ZTBSV           cublas_ztbsv_
#define CUBLAS_ZTPSV           cublas_ztpsv_
#define CUBLAS_ZGERC           cublas_zgerc_
#define CUBLAS_ZGERU           cublas_zgeru_
#define CUBLAS_ZHER            cublas_zher_
#define CUBLAS_ZHPR            cublas_zhpr_
#define CUBLAS_ZHER2           cublas_zher2_
#define CUBLAS_ZHPR2           cublas_zhpr2_

#elif CUBLAS_FORTRAN_COMPILER==CUBLAS_INTEL_FORTRAN

#define CUBLAS_INIT             CUBLAS_INIT 
#define CUBLAS_SHUTDOWN         CUBLAS_SHUTDOWN
#define CUBLAS_ALLOC            CUBLAS_ALLOC
#define CUBLAS_FREE             CUBLAS_FREE
#define CUBLAS_SET_VECTOR       CUBLAS_SET_VECTOR
#define CUBLAS_GET_VECTOR       CUBLAS_GET_VECTOR
#define CUBLAS_SET_MATRIX       CUBLAS_SET_MATRIX
#define CUBLAS_GET_MATRIX       CUBLAS_GET_MATRIX
#define CUBLAS_GET_ERROR        CUBLAS_GET_ERROR
#define CUBLAS_XERBLA           CUBLAS_XERBLA
#define CUBLAS_ISAMAX           CUBLAS_ISAMAX
#define CUBLAS_ISAMIN           CUBLAS_ISAMIN
#define CUBLAS_SASUM            CUBLAS_SASUM
#define CUBLAS_SAXPY            CUBLAS_SAXPY
#define CUBLAS_SCOPY            CUBLAS_SCOPY
#define CUBLAS_SDOT             CUBLAS_SDOT
#define CUBLAS_SNRM2            CUBLAS_SNRM2
#define CUBLAS_SROT             CUBLAS_SROT
#define CUBLAS_SROTG            CUBLAS_SROTG
#define CUBLAS_SROTM            CUBLAS_SROTM
#define CUBLAS_SROTMG           CUBLAS_SROTMG
#define CUBLAS_SSCAL            CUBLAS_SSCAL
#define CUBLAS_SSWAP            CUBLAS_SSWAP
#define CUBLAS_CAXPY            CUBLAS_CAXPY
#define CUBLAS_CCOPY            CUBLAS_CCOPY
#define CUBLAS_ZCOPY            CUBLAS_ZCOPY
#define CUBLAS_CROT             CUBLAS_CROT
#define CUBLAS_CROTG            CUBLAS_CROTG
#define CUBLAS_CSCAL            CUBLAS_CSCAL
#define CUBLAS_CSROT            CUBLAS_CSROT
#define CUBLAS_CSSCAL           CUBLAS_CSSCAL
#define CUBLAS_CSWAP            CUBLAS_CSWAP 
#define CUBLAS_ZSWAP            CUBLAS_ZSWAP 
#define CUBLAS_CTRMV            CUBLAS_CTRMV 
#define CUBLAS_CDOTU            CUBLAS_CDOTU
#define CUBLAS_CDOTC            CUBLAS_CDOTC
#define CUBLAS_ICAMAX           CUBLAS_ICAMAX
#define CUBLAS_SCASUM           CUBLAS_SCASUM
#define CUBLAS_SCNRM2           CUBLAS_SCNRM2
#define CUBLAS_SGBMV            CUBLAS_SGBMV
#define CUBLAS_SGEMV            CUBLAS_SGEMV
#define CUBLAS_SGER             CUBLAS_SGER
#define CUBLAS_SSBMV            CUBLAS_SSBMV
#define CUBLAS_SSPMV            CUBLAS_SSPMV
#define CUBLAS_SSPR             CUBLAS_SSPR
#define CUBLAS_SSPR2            CUBLAS_SSPR2
#define CUBLAS_SSYMV            CUBLAS_SSYMV
#define CUBLAS_SSYR             CUBLAS_SSYR
#define CUBLAS_SSYR2            CUBLAS_SSYR2
#define CUBLAS_STBMV            CUBLAS_STBMV
#define CUBLAS_STBSV            CUBLAS_STBSV
#define CUBLAS_STPMV            CUBLAS_STPMV
#define CUBLAS_STPSV            CUBLAS_STPSV
#define CUBLAS_STRMV            CUBLAS_STRMV
#define CUBLAS_STRSV            CUBLAS_STRSV
#define CUBLAS_SGEMM            CUBLAS_SGEMM
#define CUBLAS_SSYMM            CUBLAS_SSYMM
#define CUBLAS_SSYR2K           CUBLAS_SSYR2K
#define CUBLAS_SSYRK            CUBLAS_SSYRK
#define CUBLAS_STRMM            CUBLAS_STRMM
#define CUBLAS_STRSM            CUBLAS_STRSM
#define CUBLAS_CGEMM            CUBLAS_CGEMM
#define CUBLAS_CHEMM            CUBLAS_CHEMM
#define CUBLAS_CSYMM            CUBLAS_CSYMM
#define CUBLAS_CTRMM            CUBLAS_CTRMM
#define CUBLAS_CTRSM            CUBLAS_CTRSM
#define CUBLAS_CHERK            CUBLAS_CHERK
#define CUBLAS_CSYRK            CUBLAS_CSYRK
#define CUBLAS_CHER2K           CUBLAS_CHER2K
#define CUBLAS_CSYR2K           CUBLAS_CSYR2K
#define CUBLAS_IDAMAX           CUBLAS_IDAMAX
#define CUBLAS_IDAMIN           CUBLAS_IDAMIN
#define CUBLAS_DASUM            CUBLAS_DASUM
#define CUBLAS_DAXPY            CUBLAS_DAXPY
#define CUBLAS_DCOPY            CUBLAS_DCOPY
#define CUBLAS_DDOT             CUBLAS_DDOT
#define CUBLAS_DNRM2            CUBLAS_DNRM2
#define CUBLAS_DROT             CUBLAS_DROT
#define CUBLAS_DROTG            CUBLAS_DROTG
#define CUBLAS_DROTM            CUBLAS_DROTM
#define CUBLAS_DROTMG           CUBLAS_DROTMG
#define CUBLAS_DSCAL            CUBLAS_DSCAL
#define CUBLAS_DSWAP            CUBLAS_DSWAP
#define CUBLAS_ZAXPY            CUBLAS_ZAXPY
#define CUBLAS_ZCOPY            CUBLAS_ZCOPY
#define CUBLAS_ZROT             CUBLAS_ZROT
#define CUBLAS_ZROTG            CUBLAS_ZROTG
#define CUBLAS_ZSCAL            CUBLAS_ZSCAL
#define CUBLAS_ZDROT            CUBLAS_ZDROT
#define CUBLAS_ZDSCAL           CUBLAS_ZDSCAL
#define CUBLAS_ZSWAP            CUBLAS_ZSWAP 
#define CUBLAS_ZDOTU            CUBLAS_ZDOTU
#define CUBLAS_ZDOTC            CUBLAS_ZDOTC
#define CUBLAS_IZAMAX           CUBLAS_IZAMAX
#define CUBLAS_DZASUM           CUBLAS_DZASUM
#define CUBLAS_DZNRM2           CUBLAS_DZNRM2
#define CUBLAS_DGBMV            CUBLAS_DGBMV
#define CUBLAS_DGEMV            CUBLAS_DGEMV
#define CUBLAS_ZGEMV            CUBLAS_ZGEMV
#define CUBLAS_DGER             CUBLAS_DGER
#define CUBLAS_DSBMV            CUBLAS_DSBMV
#define CUBLAS_DSPMV            CUBLAS_DSPMV
#define CUBLAS_DSPR             CUBLAS_DSPR
#define CUBLAS_DSPR2            CUBLAS_DSPR2
#define CUBLAS_DSYMV            CUBLAS_DSYMV
#define CUBLAS_DSYR             CUBLAS_DSYR
#define CUBLAS_DSYR2            CUBLAS_DSYR2
#define CUBLAS_DTBMV            CUBLAS_DTBMV
#define CUBLAS_DTBSV            CUBLAS_DTBSV
#define CUBLAS_DTPMV            CUBLAS_DTPMV
#define CUBLAS_DTPSV            CUBLAS_DTPSV
#define CUBLAS_DTRMV            CUBLAS_DTRMV
#define CUBLAS_DTRSV            CUBLAS_DTRSV
#define CUBLAS_DGEMM            CUBLAS_DGEMM
#define CUBLAS_DSYMM            CUBLAS_DSYMM
#define CUBLAS_DSYR2K           CUBLAS_DSYR2K
#define CUBLAS_ZSYRK            CUBLAS_ZSYRK
#define CUBLAS_DTRMM            CUBLAS_DTRMM
#define CUBLAS_DTRSM            CUBLAS_DTRSM
#define CUBLAS_ZGEMM            CUBLAS_ZGEMM
#define CUBLAS_ZHEMM            CUBLAS_ZHEMM
#define CUBLAS_ZSYMM            CUBLAS_ZSYMM
#define CUBLAS_ZTRMM            CUBLAS_ZTRMM
#define CUBLAS_ZTRSM            CUBLAS_ZTRSM
#define CUBLAS_ZHERK            CUBLAS_ZHERK
#define CUBLAS_ZSYRK            CUBLAS_ZSYRK
#define CUBLAS_ZHER2K           CUBLAS_ZHER2K
#define CUBLAS_ZSYR2K           CUBLAS_ZSYR2K

#define  CUBLAS_CGEMV           CUBLAS_CGEMV
#define  CUBLAS_CGBMV           CUBLAS_CGBMV
#define  CUBLAS_CHEMV           CUBLAS_CHEMV
#define  CUBLAS_CHBMV           CUBLAS_CHBMV
#define  CUBLAS_CHPMV           CUBLAS_CHPMV
#define  CUBLAS_CTBMV           CUBLAS_CTBMV
#define  CUBLAS_CTPMV           CUBLAS_CTPMV
#define  CUBLAS_CTRSV           CUBLAS_CTRSV
#define  CUBLAS_CTBSV           CUBLAS_CTBSV
#define  CUBLAS_CTPSV           CUBLAS_CTPSV
#define  CUBLAS_CGERC           CUBLAS_CGERC
#define  CUBLAS_CGERU           CUBLAS_CGERU
#define  CUBLAS_CHPR            CUBLAS_CHPR


// stubs for zblat2
#define CUBLAS_ZGBMV           CUBLAS_ZGBMV
#define CUBLAS_ZHEMV           CUBLAS_ZHEMV
#define CUBLAS_ZHBMV           CUBLAS_ZHBMV
#define CUBLAS_ZHPMV           CUBLAS_ZHPMV
#define CUBLAS_ZTRMV           CUBLAS_ZTRMV
#define CUBLAS_ZTBMV           CUBLAS_ZTBMV
#define CUBLAS_ZTPMV           CUBLAS_ZTPMV
#define CUBLAS_ZTRSV           CUBLAS_ZTRSV
#define CUBLAS_ZTBSV           CUBLAS_ZTBSV
#define CUBLAS_ZTPSV           CUBLAS_ZTPSV
#define CUBLAS_ZGERC           CUBLAS_ZGERC
#define CUBLAS_ZGERU           CUBLAS_ZGERU
#define CUBLAS_ZHER            CUBLAS_ZHER
#define CUBLAS_ZHPR            CUBLAS_ZHPR
#define CUBLAS_ZHER2           CUBLAS_ZHER2
#define CUBLAS_ZHPR2           CUBLAS_ZHPR2

#else
#error unsupported Fortran compiler
#endif
