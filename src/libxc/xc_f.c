/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "config.h"

#ifdef HAVE_FORTRAN

#include "xc.h"
#include "string_f.h"

//#ifndef CUDA
/* version */
void FC_FUNC(xc_f90_version, XC_F90_VERSION)
     (int *major, int *minor, int *micro)
{
  xc_version(major, minor, micro);
}

void FC_FUNC(xc_f90_version_string, XC_F90_VERSIN_STRING)
     (STR_F_TYPE version_string STR_ARG1)
{
  const char *version;

  version = xc_version_string();
  TO_F_STR1(version, version_string);
}

/* info */
CC_FORTRAN_INT FC_FUNC(xc_f90_info_number, XC_F90_INFO_NUMBER)
     (void **info)
{
  return (CC_FORTRAN_INT) ((xc_func_info_type *)(*info))->number;
}


CC_FORTRAN_INT FC_FUNC(xc_f90_info_kind, XC_F90_INFO_KIND)
     (void **info)
{
  return (CC_FORTRAN_INT) ((xc_func_info_type *)(*info))->kind;
}


void FC_FUNC(xc_f90_info_name, XC_F90_INFO_NAME)
     (void **info, STR_F_TYPE s STR_ARG1)
{
  TO_F_STR1(((xc_func_info_type *)(*info))->name, s);
}


CC_FORTRAN_INT  FC_FUNC(xc_f90_info_family, XC_F90_INFO_FAMILY)
     (void **info)
{
  return (CC_FORTRAN_INT) ((xc_func_info_type *)(*info))->family;
}


CC_FORTRAN_INT  FC_FUNC(xc_f90_info_flags, XC_F90_INFO_FLAGS)
     (void **info)
{
  return (CC_FORTRAN_INT) ((xc_func_info_type *)(*info))->flags;
}


void FC_FUNC(xc_f90_info_refs, XC_F90_INFO_REFS)
     (void **info, CC_FORTRAN_INT *number, STR_F_TYPE ref_f STR_ARG1)
{
  xc_func_info_type *func_p = (xc_func_info_type *)(*info);

  assert(*number >=0 && *number < 5);

  if(func_p->refs[*number] == NULL){
    *number = -1;
    return;
  }

  TO_F_STR1(func_p->refs[*number]->ref, ref_f);

  (*number)++;
  fflush(stdout);
}


void FC_FUNC(xc_f90_functional_get_name, XC_F90_FUNCTIONAL_GET_NAME)
     (CC_FORTRAN_INT *func_number, STR_F_TYPE func_string STR_ARG1)
{
  char *name;

  name = xc_functional_get_name(*func_number);
  if ( name == NULL ) name = strndup("unknown", 256);

  TO_F_STR1(name, func_string);
  free(name);
}


CC_FORTRAN_INT  FC_FUNC(xc_f90_functional_get_number, XC_F90_FUNCTIONAL_GET_NUMBER)
     (STR_F_TYPE func_string STR_ARG1)
{
  char *name;
  int ret;

  TO_C_STR1(func_string, name);
  
  ret = xc_functional_get_number(name);
  free(name);

  return (CC_FORTRAN_INT) ret;
}


/* functionals */
CC_FORTRAN_INT  FC_FUNC(xc_f90_family_from_id, XC_F90_FAMILY_FROM_ID)
  (CC_FORTRAN_INT  *functional)
{
  return (CC_FORTRAN_INT) xc_family_from_id((int) (*functional), NULL, NULL);
}

CC_FORTRAN_INT FC_FUNC(xc_f90_number_of_functionals, XC_F90_NUMBER_OF_FUNCTIONALS)
  ()
{
  return (CC_FORTRAN_INT) xc_number_of_functionals();
}

CC_FORTRAN_INT FC_FUNC(xc_f90_maximum_name_length, XC_F90_MAXIMUM_LENGTH_NAME)
  ()
{
  return (CC_FORTRAN_INT) xc_maximum_name_length();
}

void FC_FUNC(xc_f90_available_functional_numbers, XC_F90_AVAILABLE_FUNCTIONAL_NUMBERS)
  (CC_FORTRAN_INT *list)
{
  xc_available_functional_numbers(list);
}


/* Standard initialization */
void FC_FUNC(xc_f90_func_init, XC_F90_FUNC_INIT)
     (void **p, void **info, CC_FORTRAN_INT *functional, CC_FORTRAN_INT *nspin)
{
  xc_func_type *func_p;
  
  func_p = (xc_func_type *)malloc(sizeof(xc_func_type));
  xc_func_init(func_p, (int) (*functional), (int) (*nspin));

  *p    = (void *) func_p;
  *info = (void *)(func_p->info);
}

void FC_FUNC(xc_f90_func_end, XC_F90_FUNC_END)
     (void **p)
{
  xc_func_end((xc_func_type *)(*p));
  free(*p);
  *p = NULL;
}

void FC_FUNC(xc_f90_func_set_dens_threshold, XC_F90_FUNC_SET_DENS_THRESHOLD)
     (void **p, double *dens_threshold)
{
  xc_func_set_dens_threshold((xc_func_type *)(*p), *dens_threshold);
}

void FC_FUNC(xc_f90_func_set_ext_params, XC_F90_FUNC_SET_EXT_PARAMS)
     (void **p, double *ext_params)
{
  xc_func_set_ext_params((xc_func_type *)(*p), ext_params);
}


/* LDAs */
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
void FC_FUNC(xc_f90_lda, XC_F90_LDA)
     (void **p, CC_FORTRAN_INT *np, double *rho, 
      double *zk, double *vrho, double *v2rho2, double *v3rho3, void *gpu_work_params)
{
  xc_lda((xc_func_type *)(*p), *np, rho, zk, vrho, v2rho2, v3rho3, gpu_work_params);
}

void FC_FUNC(xc_f90_lda_exc, XC_F90_LDA_EXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *zk, void *gpu_work_params)
{
  xc_lda((xc_func_type *)(*p), *np, rho, zk, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_lda_exc_vxc, XC_F90_LDA_EXC_VXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *zk, double *vrho, void *gpu_work_params)
{
  xc_lda((xc_func_type *)(*p), *np, rho, zk, vrho, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_lda_vxc, XC_F90_LDA_VXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *vrho, void *gpu_work_params)
{
  xc_lda((xc_func_type *)(*p), *np, rho, NULL, vrho, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_lda_vxc_fxc, XC_F90_LDA_VXC_FXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *vrho, double *v2rho2, void *gpu_work_params)
{
  xc_lda((xc_func_type *)(*p), *np, rho, NULL, vrho, v2rho2, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_lda_fxc, XC_F90_LDA_FXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *v2rho2, void *gpu_work_params)
{
  xc_lda((xc_func_type *)(*p), *np, rho, NULL, NULL, v2rho2, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_lda_kxc, XC_F90_LDA_KXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *v3rho3, void *gpu_work_params)
{
  xc_lda((xc_func_type *)(*p), *np, rho, NULL, NULL, NULL, v3rho3, gpu_work_params);
}

/* GGAs */

void FC_FUNC(xc_f90_gga, XC_F90_GGA)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma,
      double *zk, double *vrho, double *vsigma,
      double *v2rho2, double *v2rhosigma, double *v2sigma2,
      double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3, void *gpu_work_params)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, zk, vrho, vsigma,
          v2rho2, v2rhosigma, v2sigma2, v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3, gpu_work_params);
}

void FC_FUNC(xc_f90_gga_exc, XC_F90_GGA_EXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma,
      double *zk, void *gpu_work_params)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, zk, NULL, NULL,
          NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_gga_exc_vxc, XC_F90_GGA_EXC_VXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma,
      double *zk, double *vrho, double *vsigma, void *gpu_work_params)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, zk, vrho, vsigma,
          NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_gga_vxc, XC_F90_GGA_VXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma,
      double *vrho, double *vsigma, void *gpu_work_params)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, NULL, vrho, vsigma,
          NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_gga_vxc_fxc, XC_F90_GGA_VXC_FXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma,
      double *vrho, double *vsigma,
      double *v2rho2, double *v2rhosigma, double *v2sigma2, void *gpu_work_params)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, NULL, vrho, vsigma,
          v2rho2, v2rhosigma, v2sigma2, NULL, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_gga_fxc, XC_F90_GGA_FXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma,
      double *v2rho2, double *v2rhosigma, double *v2sigma2, void *gpu_work_params)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, NULL, NULL, NULL,
          v2rho2, v2rhosigma, v2sigma2, NULL, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_gga_kxc, XC_F90_GGA_KXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma,
      double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3, void *gpu_work_params)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, NULL, NULL, NULL,
          NULL, NULL, NULL, v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3, gpu_work_params);
}

void FC_FUNC(xc_f90_gga_lb_modified, XC_F90_GGA_LB_MODIFIED)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *r, double *vrho, void *gpu_work_params)
{
  xc_gga_lb_modified((xc_func_type *)(*p), *np, rho, sigma, *r, vrho, gpu_work_params);
}

#else
void FC_FUNC(xc_f90_lda, XC_F90_LDA)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *zk, double *vrho, double *v2rho2, double *v3rho3)
{
  xc_lda((xc_func_type *)(*p), *np, rho, zk, vrho, v2rho2, v3rho3);
}
void FC_FUNC(xc_f90_lda_exc, XC_F90_LDA_EXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *zk)
{
  xc_lda((xc_func_type *)(*p), *np, rho, zk, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_lda_exc_vxc, XC_F90_LDA_EXC_VXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, 
      double *zk, double *vrho)
{
  xc_lda((xc_func_type *)(*p), *np, rho, zk, vrho, NULL, NULL);
}

void FC_FUNC(xc_f90_lda_vxc, XC_F90_LDA_VXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, 
      double *vrho)
{
  xc_lda((xc_func_type *)(*p), *np, rho, NULL, vrho, NULL, NULL);
}

void FC_FUNC(xc_f90_lda_vxc_fxc, XC_F90_LDA_VXC_FXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *vrho, double *v2rho2)
{
  xc_lda((xc_func_type *)(*p), *np, rho, NULL, vrho, v2rho2, NULL);
}

void FC_FUNC(xc_f90_lda_fxc, XC_F90_LDA_FXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *v2rho2)
{
  xc_lda((xc_func_type *)(*p), *np, rho, NULL, NULL, v2rho2, NULL);
}

void FC_FUNC(xc_f90_lda_kxc, XC_F90_LDA_KXC)
     (void **p, CC_FORTRAN_INT *np, double *rho,
      double *v3rho3)
{
  xc_lda((xc_func_type *)(*p), *np, rho, NULL, NULL, NULL, v3rho3);
}

/* GGAs */

void FC_FUNC(xc_f90_gga, XC_F90_GGA)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, 
      double *zk, double *vrho, double *vsigma,
      double *v2rho2, double *v2rhosigma, double *v2sigma2,
      double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, zk, vrho, vsigma, 
	  v2rho2, v2rhosigma, v2sigma2, v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
}

void FC_FUNC(xc_f90_gga_exc, XC_F90_GGA_EXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, 
      double *zk)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, zk, NULL, NULL, 
	  NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_gga_exc_vxc, XC_F90_GGA_EXC_VXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, 
      double *zk, double *vrho, double *vsigma)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, zk, vrho, vsigma, 
	  NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_gga_vxc, XC_F90_GGA_VXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, 
      double *vrho, double *vsigma)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, NULL, vrho, vsigma, 
	  NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_gga_vxc_fxc, XC_F90_GGA_VXC_FXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, 
      double *vrho, double *vsigma,
      double *v2rho2, double *v2rhosigma, double *v2sigma2)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, NULL, vrho, vsigma, 
	  v2rho2, v2rhosigma, v2sigma2, NULL, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_gga_fxc, XC_F90_GGA_FXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, 
      double *v2rho2, double *v2rhosigma, double *v2sigma2)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, NULL, NULL, NULL, 
	  v2rho2, v2rhosigma, v2sigma2, NULL, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_gga_kxc, XC_F90_GGA_KXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, 
      double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
{
  xc_gga((xc_func_type *)(*p), *np, rho, sigma, NULL, NULL, NULL, 
	  NULL, NULL, NULL, v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
}

void FC_FUNC(xc_f90_gga_lb_modified, XC_F90_GGA_LB_MODIFIED)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *r, double *vrho)
{
  xc_gga_lb_modified((xc_func_type *)(*p), *np, rho, sigma, *r, vrho);
}
#endif

void FC_FUNC(f90_gga_x_b86_set_par, F90_GGA_X_B86_SET_PAR)
  (void **p, double *beta, double *gamma, double *omega)
{
  xc_gga_x_b86_set_params((xc_func_type *)(*p), *beta, *gamma, *omega);
}

void FC_FUNC(f90_gga_x_b88_set_par, F90_GGA_X_B88_SET_PAR)
  (void **p, double *beta, double *gamma)
{
  xc_gga_x_b88_set_params((xc_func_type *)(*p), *beta, *gamma);
}

void FC_FUNC(f90_gga_x_pbe_set_par, F90_GGA_X_PBE_SET_PAR)
  (void **p, double *kappa, double *mu)
{
  xc_gga_x_pbe_set_params((xc_func_type *)(*p), *kappa, *mu);
}

void FC_FUNC(f90_gga_x_pbeint_set_par, F90_GGA_X_PBEINT_SET_PAR)
 (void **p, double *kappa, double *alpha, double *muPBE, double *muGE)
{
  xc_gga_x_pbeint_set_params((xc_func_type *)(*p), *kappa, *alpha, *muPBE, *muGE);
}

void FC_FUNC(f90_gga_c_pbe_set_par, F90_GGA_C_PBE_SET_PAR)
  (void **p, double *beta)
{
  xc_gga_c_pbe_set_params((xc_func_type *)(*p), *beta);
}

void FC_FUNC(f90_gga_x_pw91_set_par, F90_GGA_X_PW91_SET_PAR)
  (void **p, double *a, double *b, double *c, double *d, double *f, double *alpha, double *expo)
{
  xc_gga_x_pw91_set_params((xc_func_type *)(*p), *a, *b, *c, *d, *f, *alpha, *expo);
}

void FC_FUNC(f90_gga_x_pw91_set_par2, F90_GGA_X_PW91_SET_PAR2)
  (void **p, double *bt, double *alpha, double *expo)
{
  xc_gga_x_pw91_set_params2((xc_func_type *)(*p), *bt, *alpha, *expo);
}

void FC_FUNC(f90_gga_x_rpbe_set_par, F90_GGA_X_RPBE_SET_PAR)
  (void **p, double *kappa, double *mu)
{
  xc_gga_x_rpbe_set_params((xc_func_type *)(*p), *kappa, *mu);
}

void FC_FUNC(f90_gga_x_optx_set_par, F90_GGA_X_OPTX_SET_PAR)
  (void **p, double *a, double *b, double *gamma)
{
  xc_gga_x_optx_set_params((xc_func_type *)(*p), *a, *b, *gamma);
}

void FC_FUNC(f90_gga_c_lyp_set_par, F90_GGA_C_LYP_SET_PAR)
  (void **p, double *A, double *B, double *c, double *d)
{
  xc_gga_c_lyp_set_params((xc_func_type *)(*p), *A, *B, *c, *d);
}

void FC_FUNC(f90_gga_x_wpbeh_set_par, F90_GGA_X_WPBEH_SET_PAR)
  (void **p, double *omega)
{
  xc_gga_x_wpbeh_set_params((xc_func_type *)(*p), *omega);
}

void FC_FUNC(f90_gga_x_hjs_set_par, F90_GGA_X_HJS_SET_PAR)
  (void **p, double *omega)
{
  xc_gga_x_hjs_set_params((xc_func_type *)(*p), *omega);
}

void FC_FUNC(f90_gga_x_hjs_b88_v2_set_par, F90_GGA_X_HJS_SET_PAR)
  (void **p, double *omega)
{
  xc_gga_x_hjs_b88_v2_set_params((xc_func_type *)(*p), *omega);
}

void FC_FUNC(f90_gga_x_ityh_set_par, F90_GGA_X_ITYH_SET_PAR)
  (void **p, CC_FORTRAN_INT *func_id, double *omega)
{
  xc_gga_x_ityh_set_params((xc_func_type *)(*p), *func_id, *omega);
}

void FC_FUNC(f90_gga_x_sfat_set_par, F90_GGA_X_SFAT_SET_PAR)
  (void **p, CC_FORTRAN_INT *func_id, double *omega)
{
  xc_gga_x_sfat_set_params((xc_func_type *)(*p), *func_id, *omega);
}

void FC_FUNC(f90_gga_x_ssb_sw_set_par, F90_GGA_X_SSB_SW_SET_PAR)
  (void **p, double *A, double *B, double *C, double *D, double *E)
{
  xc_gga_x_ssb_sw_set_params((xc_func_type *)(*p), *A, *B, *C, *D, *E);
}

void FC_FUNC(f90_gga_x_kt_set_par, F90_GGA_X_KT_SET_PAR)
  (void **p, double *gamma, double *delta)
{
  xc_gga_x_kt_set_params((xc_func_type *)(*p), *gamma, *delta);
}

void FC_FUNC(xc_f90_gga_ak13_get_asymptotic, XC_F90_GGA_AK13_GET_ASYMPTOTIC)
  (double *homo, double *asymp)
{
  *asymp = xc_gga_ak13_get_asymptotic(*homo);
}

void FC_FUNC(xc_f90_hyb_exx_coef, XC_F90_HYB_EXX_COEF)
   (void **p, double *coef)
{
  *coef = xc_hyb_exx_coef((xc_func_type *)(*p));
}

void FC_FUNC(xc_f90_hyb_cam_coef, XC_F90_HYB_CAM_COEF)
  (void **p, double *omega, double *alpha, double *beta)
{
  xc_hyb_cam_coef((xc_func_type *)(*p), omega, alpha, beta);
}

void FC_FUNC(xc_f90_nlc_coef, XC_F90_NLC_COEF)
  (void **p, double *nlc_b, double *nlc_c)
{
  xc_nlc_coef((xc_func_type *)(*p), nlc_b, nlc_c);
}

/* meta-GGAs */
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
void FC_FUNC(xc_f90_mgga, XC_F90_MGGA)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
      double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
      double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
      double *v2rhosigma, double *v2rholapl, double *v2rhotau,
      double *v2sigmalapl, double *v2sigmatau, double *v2lapltau, void *gpu_work_params)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau,
           zk, vrho, vsigma, vlapl, vtau,
           v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
           v2sigmalapl, v2sigmatau, v2lapltau, gpu_work_params);

}

/*void FC_FUNC(xc_f90_mgga, XC_F90_MGGA)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
      double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
      double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
      double *v2rhosigma, double *v2rholapl, double *v2rhotau,
      double *v2sigmalapl, double *v2sigmatau, double *v2lapltau, void *gpu_work_params)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau,
           zk, vrho, vsigma, vlapl, vtau,
           v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
           v2sigmalapl, v2sigmatau, v2lapltau, gpu_work_params);

}*/

void FC_FUNC(xc_f90_mgga_exc, XC_F90_MGGA_EXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
      double *zk, void *gpu_work_params)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau,
           zk, NULL, NULL, NULL, NULL,
           NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_mgga_exc_vxc, XC_F90_MGGA_EXC_VXC)
  (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
   double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau, void *gpu_work_params)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau,
           zk, vrho, vsigma, vlapl, vtau,
           NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_mgga_vxc, XC_F90_MGGA_VXC)
  (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
   double *vrho, double *vsigma, double *vlapl, double *vtau, void *gpu_work_params)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau,
           NULL, vrho, vsigma, vlapl, vtau,
           NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void FC_FUNC(xc_f90_mgga_vxc_fxc, XC_F90_MGGA_VXC_FXC)
  (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
   double *vrho, double *vsigma, double *vlapl, double *vtau,
   double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
   double *v2rhosigma, double *v2rholapl, double *v2rhotau,
   double *v2sigmalapl, double *v2sigmatau, double *v2lapltau, void *gpu_work_params)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau,
           NULL, vrho, vsigma, vlapl, vtau,
           v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
           v2sigmalapl, v2sigmatau, v2lapltau, gpu_work_params);
}

void FC_FUNC(xc_f90_mgga_fxc, XC_F90_MGGA_FXC)
  (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
      double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
      double *v2rhosigma, double *v2rholapl, double *v2rhotau,
      double *v2sigmalapl, double *v2sigmatau, double *v2lapltau, void *gpu_work_params)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau,
           NULL, NULL, NULL, NULL, NULL,
           v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
           v2sigmalapl, v2sigmatau, v2lapltau, gpu_work_params);
}
#else
void FC_FUNC(xc_f90_mgga, XC_F90_MGGA)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
      double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
      double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
      double *v2rhosigma, double *v2rholapl, double *v2rhotau, 
      double *v2sigmalapl, double *v2sigmatau, double *v2lapltau)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau, 
	   zk, vrho, vsigma, vlapl, vtau,
	   v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau, 
	   v2sigmalapl, v2sigmatau, v2lapltau);

}

void FC_FUNC(xc_f90_mgga_exc, XC_F90_MGGA_EXC)
     (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau, 
      double *zk)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau, 
	   zk, NULL, NULL, NULL, NULL, 
	   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_mgga_exc_vxc, XC_F90_MGGA_EXC_VXC)
  (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
   double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau, 
	   zk, vrho, vsigma, vlapl, vtau, 
	   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_mgga_vxc, XC_F90_MGGA_VXC)
  (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
   double *vrho, double *vsigma, double *vlapl, double *vtau)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau, 
	   NULL, vrho, vsigma, vlapl, vtau, 
	   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void FC_FUNC(xc_f90_mgga_vxc_fxc, XC_F90_MGGA_VXC_FXC)
  (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
   double *vrho, double *vsigma, double *vlapl, double *vtau,
   double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
   double *v2rhosigma, double *v2rholapl, double *v2rhotau, 
   double *v2sigmalapl, double *v2sigmatau, double *v2lapltau)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau, 
	   NULL, vrho, vsigma, vlapl, vtau,
	   v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau, 
	   v2sigmalapl, v2sigmatau, v2lapltau);
}

void FC_FUNC(xc_f90_mgga_fxc, XC_F90_MGGA_FXC)
  (void **p, CC_FORTRAN_INT *np, double *rho, double *sigma, double *lapl, double *tau,
      double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
      double *v2rhosigma, double *v2rholapl, double *v2rhotau, 
      double *v2sigmalapl, double *v2sigmatau, double *v2lapltau)
{
  xc_mgga((xc_func_type *)(*p), *np, rho, sigma, lapl, tau, 
	   NULL, NULL, NULL, NULL, NULL, 
	   v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau, 
	   v2sigmalapl, v2sigmatau, v2lapltau);
}
#endif
void FC_FUNC(f90_mgga_x_tpss_set_par, F90_MGGA_X_TPSS_SET_PAR)
     (void **p, double *b, double *c, double *e, double *kappa, double *mu, double *BLOC_a, double *BLOC_b)
{
    xc_mgga_x_tpss_set_params((xc_func_type *)(*p), *b, *c, *e, *kappa, *mu, *BLOC_a, *BLOC_b);
}

void FC_FUNC(f90_mgga_c_tpss_set_par, F90_MGGA_C_TPSS_SET_PAR)
  (void **p, double *beta, double *d, double *C0_0, double *C0_1, double *C0_2, double *C0_3)
{
  xc_mgga_c_tpss_set_params((xc_func_type *)(*p), *beta, *d, *C0_0, *C0_1, *C0_2, *C0_3);
}

void FC_FUNC(f90_mgga_c_bc95_set_par, F90_MGGA_C_BC95_SET_PAR)
  (void **p, double *css, double *copp)
{
  xc_mgga_c_bc95_set_params((xc_func_type *)(*p), *css, *copp);
}
//#endif
#endif
