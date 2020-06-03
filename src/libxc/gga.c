/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"
#include "funcs_gga.c"
#include "funcs_hyb_gga.c"

/* Some useful formulas:

   sigma_st          = grad rho_s . grad rho_t
   zk                = energy density per unit particle

   vrho_s            = d zk / d rho_s
   vsigma_st         = d n*zk / d sigma_st
   
   v2rho2_st         = d^2 n*zk / d rho_s d rho_t
   v2rhosigma_svx    = d^2 n*zk / d rho_s d sigma_tv
   v2sigma2_stvx     = d^2 n*zk / d sigma_st d sigma_vx

   v3rho3_stv        = d^3 n*zk / d rho_s d rho_t d rho_v
   v3rho2sigma_stvx  = d^3 n*zk / d rho_s d rho_t d sigma_vx
   v3rhosigma2_svxyz = d^3 n*zk / d rho_s d sigma_vx d sigma_yz
   v3sigma3_stvxyz   = d^3 n*zk / d sigma_st d sigma_vx d sigma_yz

if nspin == 2
   rho(2)          = (u, d)
   sigma(3)        = (uu, ud, dd)

   vrho(2)         = (u, d)
   vsigma(3)       = (uu, ud, dd)

   v2rho2(3)       = (u_u, u_d, d_d)
   v2rhosigma(6)   = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
   v2sigma2(6)     = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)

   v3rho3(4)       = (u_u_u, u_u_d, u_d_d, d_d_d)
   v3rho2sigma(9)  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
   v3rhosigma2(12) = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
   v3sigma(10)     = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)
   
*/
#if defined CUDA || defined CUDA_MPIV
void xc_gga(const xc_func_type *func, int np, const double *rho, const double *sigma,
             double *zk, double *vrho, double *vsigma,
             double *v2rho2, double *v2rhosigma, double *v2sigma2,
             double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3, void *gpu_work_params)
{
#else
void xc_gga(const xc_func_type *func, int np, const double *rho, const double *sigma,
	     double *zk, double *vrho, double *vsigma,
	     double *v2rho2, double *v2rhosigma, double *v2sigma2,
	     double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
{
#endif
  assert(func != NULL);
  
  /* sanity check */
  if(zk != NULL && !(func->info->flags & XC_FLAGS_HAVE_EXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of Exc\n",
	    func->info->name);
    exit(1);
  }

  if(vrho != NULL && !(func->info->flags & XC_FLAGS_HAVE_VXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of vxc\n",
	    func->info->name);
    exit(1);
  }

  if(v2rho2 != NULL && !(func->info->flags & XC_FLAGS_HAVE_FXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of fxc\n",
	    func->info->name);
    exit(1);
  }

  if(v3rho3 != NULL && !(func->info->flags & XC_FLAGS_HAVE_KXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of kxc\n",
	    func->info->name);
    exit(1);
  }

  /* initialize output to zero */
  if(zk != NULL)
    memset(zk, 0, func->n_zk*np*sizeof(double));

  if(vrho != NULL){
    assert(vsigma != NULL);
    
    memset(vrho,   0, func->n_vrho  *np*sizeof(double));
    memset(vsigma, 0, func->n_vsigma*np*sizeof(double));
  }

  if(v2rho2 != NULL){
    assert(v2rhosigma!=NULL && v2sigma2!=NULL);

    memset(v2rho2,     0, func->n_v2rho2    *np*sizeof(double));
    memset(v2rhosigma, 0, func->n_v2rhosigma*np*sizeof(double));
    memset(v2sigma2,   0, func->n_v2sigma2  *np*sizeof(double));
  }

  if(v3rho3 != NULL){
    assert(v3rho2sigma!=NULL && v3rhosigma2!=NULL && v3sigma3!=NULL);

    memset(v3rho3,      0, func->n_v3rho3     *np*sizeof(double));
    memset(v3rho2sigma, 0, func->n_v3rho2sigma*np*sizeof(double));
    memset(v3rhosigma2, 0, func->n_v3rhosigma2*np*sizeof(double));
    memset(v3sigma3,    0, func->n_v3sigma3   *np*sizeof(double));    
  }

  /* call functional */
  if(func->info->gga != NULL){
#if defined CUDA || defined CUDA_MPIV
    func->info->gga(func, np, rho, sigma, zk, vrho, vsigma,
                    v2rho2, v2rhosigma, v2sigma2,
                    v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3, gpu_work_params);
#else
    func->info->gga(func, np, rho, sigma, zk, vrho, vsigma, 
		    v2rho2, v2rhosigma, v2sigma2,
		    v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
#endif
  }
  if(func->mix_coef != NULL){
#if defined CUDA || defined CUDA_MPIV
    xc_mix_func(func, np, rho, sigma, NULL, NULL, zk, vrho, vsigma, NULL, NULL,
                 v2rho2, v2sigma2, NULL, NULL, v2rhosigma, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
#else
    xc_mix_func(func, np, rho, sigma, NULL, NULL, zk, vrho, vsigma, NULL, NULL,
		 v2rho2, v2sigma2, NULL, NULL, v2rhosigma, NULL, NULL, NULL, NULL, NULL);
#endif
  }
}

/* specializations */
/* returns only energy */
#if defined CUDA || defined CUDA_MPIV
void
xc_gga_exc(const xc_func_type *p, int np, const double *rho, const double *sigma,
            double *zk, void *gpu_work_params)
{
  xc_gga(p, np, rho, sigma, zk, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

/* returns only potential */
void
xc_gga_vxc(const xc_func_type *p, int np, const double *rho, const double *sigma,
            double *vrho, double *vsigma, void *gpu_work_params)
{
  xc_gga(p, np, rho, sigma, NULL, vrho, vsigma, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

/* returns both energy and potential (the most common call usually) */
void
xc_gga_exc_vxc(const xc_func_type *p, int np, const double *rho, const double *sigma,
                double *zk, double *vrho, double *vsigma, void *gpu_work_params)
{
  xc_gga(p, np, rho, sigma, zk, vrho, vsigma, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

/* returns second derivatives */
void
xc_gga_fxc(const xc_func_type *p, int np, const double *rho, const double *sigma,
            double *v2rho2, double *v2rhosigma, double *v2sigma2, void *gpu_work_params)
{
  xc_gga(p, np, rho, sigma, NULL, NULL, NULL, v2rho2, v2rhosigma, v2sigma2, NULL, NULL, NULL, NULL, gpu_work_params);
}

/* returns third derivatives */
void
xc_gga_kxc(const xc_func_type *p, int np, const double *rho, const double *sigma,
            double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3, void *gpu_work_params)
{
  xc_gga(p, np, rho, sigma, NULL, NULL, NULL, NULL, NULL, NULL, v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3, gpu_work_params);
}
#else
void
xc_gga_exc(const xc_func_type *p, int np, const double *rho, const double *sigma, 
	    double *zk)
{
  xc_gga(p, np, rho, sigma, zk, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

/* returns only potential */
void
xc_gga_vxc(const xc_func_type *p, int np, const double *rho, const double *sigma,
	    double *vrho, double *vsigma)
{
  xc_gga(p, np, rho, sigma, NULL, vrho, vsigma, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

/* returns both energy and potential (the most common call usually) */
void
xc_gga_exc_vxc(const xc_func_type *p, int np, const double *rho, const double *sigma,
		double *zk, double *vrho, double *vsigma)
{
  xc_gga(p, np, rho, sigma, zk, vrho, vsigma, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

/* returns second derivatives */
void
xc_gga_fxc(const xc_func_type *p, int np, const double *rho, const double *sigma,
	    double *v2rho2, double *v2rhosigma, double *v2sigma2)
{
  xc_gga(p, np, rho, sigma, NULL, NULL, NULL, v2rho2, v2rhosigma, v2sigma2, NULL, NULL, NULL, NULL);
}

/* returns third derivatives */
void
xc_gga_kxc(const xc_func_type *p, int np, const double *rho, const double *sigma,
	    double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
{
  xc_gga(p, np, rho, sigma, NULL, NULL, NULL, NULL, NULL, NULL, v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
}
#endif
