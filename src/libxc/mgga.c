/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"
#include "funcs_mgga.c"
#include "funcs_hyb_mgga.c"

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
void
xc_mgga(const xc_func_type *func, int np,
         const double *rho, const double *sigma, const double *lapl, const double *tau,
         double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
         double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
         double *v2rhosigma, double *v2rholapl, double *v2rhotau,
         double *v2sigmalapl, double *v2sigmatau, double *v2lapltau, void *gpu_work_params)
#else
void 
xc_mgga(const xc_func_type *func, int np,
	 const double *rho, const double *sigma, const double *lapl, const double *tau,
	 double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
	 double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
	 double *v2rhosigma, double *v2rholapl, double *v2rhotau, 
	 double *v2sigmalapl, double *v2sigmatau, double *v2lapltau)
#endif
{
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

  /* initialize output to zero */
  if(zk != NULL)
    memset(zk, 0, func->n_zk*np*sizeof(double));

  if(vrho != NULL){
    assert(vsigma != NULL);

    memset(vrho,   0, func->n_vrho  *np*sizeof(double));
    memset(vsigma, 0, func->n_vsigma*np*sizeof(double));
    memset(vtau,   0, func->n_vtau  *np*sizeof(double));
    if(func->info->flags & XC_FLAGS_NEEDS_LAPLACIAN)
      memset(vlapl,  0, func->n_vlapl *np*sizeof(double));
  }

  if(v2rho2 != NULL){
    /* warning : lapl terms missing here */
    assert(v2sigma2   != NULL && v2tau2      != NULL && v2lapl2   != NULL &&
	   v2rhosigma != NULL && v2rhotau    != NULL && v2rholapl != NULL &&
	   v2sigmatau != NULL && v2sigmalapl != NULL && v2lapltau != NULL);

    memset(v2rho2,      0, func->n_v2rho2     *np*sizeof(double));
    memset(v2sigma2,    0, func->n_v2sigma2   *np*sizeof(double));
    memset(v2tau2,      0, func->n_v2tau2     *np*sizeof(double));
    memset(v2rhosigma,  0, func->n_v2rhosigma *np*sizeof(double));
    memset(v2rhotau,    0, func->n_v2rhotau   *np*sizeof(double));
    memset(v2sigmatau,  0, func->n_v2sigmatau *np*sizeof(double));

    if(func->info->flags & XC_FLAGS_NEEDS_LAPLACIAN){
      memset(v2lapl2,     0, func->n_v2lapl2    *np*sizeof(double));
      memset(v2rholapl,   0, func->n_v2rholapl  *np*sizeof(double));
      memset(v2sigmalapl, 0, func->n_v2sigmalapl*np*sizeof(double));
      memset(v2lapltau,   0, func->n_v2lapltau  *np*sizeof(double));
    }
  }

  /* call functional */
  if(func->info->mgga != NULL)
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
    func->info->mgga(func, np, rho, sigma, lapl, tau, zk, vrho, vsigma, vlapl, vtau, 
		     v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
		     v2sigmalapl, v2sigmatau, v2lapltau, gpu_work_params);
#else
    func->info->mgga(func, np, rho, sigma, lapl, tau, zk, vrho, vsigma, vlapl, vtau,
                     v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
                     v2sigmalapl, v2sigmatau, v2lapltau);
#endif

  if(func->mix_coef != NULL)
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
    xc_mix_func(func, np, rho, sigma, lapl, tau, zk, vrho, vsigma, vlapl, vtau, 
		 v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
		 v2sigmalapl, v2sigmatau, v2lapltau, gpu_work_params);
#else
    xc_mix_func(func, np, rho, sigma, lapl, tau, zk, vrho, vsigma, vlapl, vtau,
                 v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
                 v2sigmalapl, v2sigmatau, v2lapltau);
#endif
}

/* specializations */
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
void
xc_mgga_exc(const xc_func_type *p, int np,
             const double *rho, const double *sigma, const double *lapl, const double *tau,
             double *zk, void *gpu_work_params)
{
  xc_mgga(p, np, rho, sigma, lapl, tau, zk, NULL, NULL, NULL, NULL,
           NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void
xc_mgga_exc_vxc(const xc_func_type *p, int np,
                 const double *rho, const double *sigma, const double *lapl, const double *tau,
                 double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau, void *gpu_work_params)
{
  xc_mgga(p, np, rho, sigma, lapl, tau, zk, vrho, vsigma, vlapl, vtau,
           NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void
xc_mgga_vxc(const xc_func_type *p, int np,
             const double *rho, const double *sigma, const double *lapl, const double *tau,
             double *vrho, double *vsigma, double *vlapl, double *vtau, void *gpu_work_params)
{
  xc_mgga(p, np, rho, sigma, lapl, tau, NULL, vrho, vsigma, vlapl, vtau,
           NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, gpu_work_params);
}

void
xc_mgga_fxc(const xc_func_type *p, int np,
             const double *rho, const double *sigma, const double *lapl, const double *tau,
             double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
             double *v2rhosigma, double *v2rholapl, double *v2rhotau,
             double *v2sigmalapl, double *v2sigmatau, double *v2lapltau, void *gpu_work_params)
{
  xc_mgga(p, np, rho, sigma, lapl, tau, NULL, NULL, NULL, NULL, NULL,
           v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
           v2sigmalapl, v2sigmatau, v2lapltau, gpu_work_params);
}

#else
void
xc_mgga_exc(const xc_func_type *p, int np, 
	     const double *rho, const double *sigma, const double *lapl, const double *tau,
	     double *zk)
{
  xc_mgga(p, np, rho, sigma, lapl, tau, zk, NULL, NULL, NULL, NULL, 
	   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void
xc_mgga_exc_vxc(const xc_func_type *p, int np,
		 const double *rho, const double *sigma, const double *lapl, const double *tau,
		 double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau)
{
  xc_mgga(p, np, rho, sigma, lapl, tau, zk, vrho, vsigma, vlapl, vtau, 
	   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void
xc_mgga_vxc(const xc_func_type *p, int np,
	     const double *rho, const double *sigma, const double *lapl, const double *tau,
	     double *vrho, double *vsigma, double *vlapl, double *vtau)
{
  xc_mgga(p, np, rho, sigma, lapl, tau, NULL, vrho, vsigma, vlapl, vtau, 
	   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

void
xc_mgga_fxc(const xc_func_type *p, int np,
	     const double *rho, const double *sigma, const double *lapl, const double *tau,
	     double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
	     double *v2rhosigma, double *v2rholapl, double *v2rhotau, 
	     double *v2sigmalapl, double *v2sigmatau, double *v2lapltau)
{
  xc_mgga(p, np, rho, sigma, lapl, tau, NULL, NULL, NULL, NULL, NULL, 
	   v2rho2, v2sigma2, v2lapl2, v2tau2, v2rhosigma, v2rholapl, v2rhotau,
	   v2sigmalapl, v2sigmatau, v2lapltau);
}
#endif
