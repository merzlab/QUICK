/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

/**
 * @file work_lda.c
 * @brief This file is to be included in LDA functionals. As often these
 *        functionals are written as a function of rs and zeta, this
 *        routine performs the necessary conversions between this and a functional
 *        of rho.
 */


#ifndef XC_DIMENSIONS
#define XC_DIMENSIONS 3
#endif


/**
 * @param[in,out] func_type: pointer to pspdata structure to be initialized
 */
static void 
#if defined CUDA || defined CUDA_MPIV
work_lda(const xc_func_type *p, int np, const double *rho,
         double *zk, double *vrho, double *v2rho2, double *v3rho3, void *gpu_work_params)
#else
work_lda(const xc_func_type *p, int np, const double *rho, 
	 double *zk, double *vrho, double *v2rho2, double *v3rho3)
#endif
{
  xc_lda_work_t r;
  int is, ip;
  double dens, drs, d2rs, d3rs;

  /* Wigner radius */
# if   XC_DIMENSIONS == 1
  const double cnst_rs = 0.5;
# elif XC_DIMENSIONS == 2
  const double cnst_rs = 1.0/M_SQRTPI;
# else /* three dimensions */
  const double cnst_rs = RS_FACTOR;
# endif

  /* Initialize memory */
  memset(&r, 0, sizeof(r));

  r.order = -1;
  if(zk     != NULL) r.order = 0;
  if(vrho   != NULL) r.order = 1;
  if(v2rho2 != NULL) r.order = 2;
  if(v3rho3 != NULL) r.order = 3;
  if(r.order < 0) return;

#if defined CUDA || defined CUDA_MPIV
#include "gpu_extern.h"

#define GET_XSTR(fval) GET_STR(fval)
#define GET_STR(fval) #fval

        //char fname[50] = GET_XSTR(func);
	//char fname[50] = "test_func";

#ifdef DEBUG         
                printf("FILE: %s, LINE: %d, FUNCTION: %s, XC_DIMENSIONS: %d, work_lda at work.. \n",
                __FILE__, __LINE__, __func__, XC_DIMENSIONS);
#endif
	int xc_dim = XC_DIMENSIONS;	
        set_gpu_lda_work_params(p->dens_threshold, cnst_rs, xc_dim, kernel_id, (gpu_lda_work_params*)gpu_work_params);

#else
  for(ip = 0; ip < np; ip++){
    xc_rho2dzeta(p->nspin, rho, &dens, &r.z);

    if(dens < p->dens_threshold) goto end_ip_loop;

    r.rs = cnst_rs*pow(dens, -1.0/XC_DIMENSIONS);

    func(p, &r);

    if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
      *zk = r.f;

    if(r.order < 1) goto end_ip_loop;

    drs = -r.rs/(XC_DIMENSIONS*dens);
    
    if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC)){
      vrho[0] = r.f + dens*r.dfdrs*drs;

      if(p->nspin == XC_POLARIZED){
	vrho[1] = vrho[0] - (r.z + 1.0)*r.dfdz;
	vrho[0] = vrho[0] - (r.z - 1.0)*r.dfdz;
      }
    }
  
    if(r.order < 2) goto end_ip_loop;
    
    d2rs = -drs*(1.0 + XC_DIMENSIONS)/(XC_DIMENSIONS*dens);
    
    if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC)){
      v2rho2[0] = r.dfdrs*(2.0*drs + dens*d2rs) + dens*r.d2fdrs2*drs*drs;
      
      if(p->nspin == XC_POLARIZED){
	double sign[3][2] = {{-1.0, -1.0}, {-1.0, +1.0}, {+1.0, +1.0}};
	
	for(is=2; is>=0; is--){
	  v2rho2[is] = v2rho2[0] - r.d2fdrsz*(2.0*r.z + sign[is][0] + sign[is][1])*drs
	    + (r.z + sign[is][0])*(r.z + sign[is][1])*r.d2fdz2/dens;
	}
      }
    }
    
    if(r.order < 3) goto end_ip_loop;

    d3rs = -d2rs*(1.0 + 2.0*XC_DIMENSIONS)/(XC_DIMENSIONS*dens);
    
    if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC)){
      v3rho3[0] = r.dfdrs*(3.0*d2rs + dens*d3rs) + 
	3.0*r.d2fdrs2*drs*(drs + dens*d2rs) + r.d3fdrs3*dens*drs*drs*drs;
      
      if(p->nspin == XC_POLARIZED){
	double sign[4][3] = {{-1.0, -1.0, -1.0}, {-1.0, -1.0, +1.0}, {-1.0, +1.0, +1.0}, {+1.0, +1.0, +1.0}};
	
	for(is=3; is>=0; is--){
	  double ff;
	  
	  v3rho3[is]  = v3rho3[0] - (2.0*r.z  + sign[is][0] + sign[is][1])*(d2rs*r.d2fdrsz + drs*drs*r.d3fdrs2z);
	  v3rho3[is] += (r.z + sign[is][0])*(r.z + sign[is][1])*(-r.d2fdz2/dens + r.d3fdrsz2*drs)/dens;
	  
	  ff  = r.d2fdrsz*(2.0*drs + dens*d2rs) + dens*r.d3fdrs2z*drs*drs;
	  ff += -2.0*r.d2fdrsz*drs - r.d3fdrsz2*(2.0*r.z + sign[is][0] + sign[is][1])*drs;
	  ff += (r.z + sign[is][0])*(r.z + sign[is][1])*r.d3fdz3/dens;
	  ff += (2.0*r.z  + sign[is][0] + sign[is][1])*r.d2fdz2/dens;
	  
	  v3rho3[is] += -ff*(r.z + sign[is][2])/dens;
	}
      }
    }

  end_ip_loop:
    rho += p->n_rho;

    if(zk != NULL)
      zk += p->n_zk;
    
    if(vrho != NULL)
      vrho += p->n_vrho;

    if(v2rho2 != NULL)
      v2rho2 += p->n_v2rho2;

    if(v3rho3 != NULL)
      v3rho3 += p->n_v3rho3;

  } /* for(ip) */
#endif
}
