/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

/************************************************************************
  This file is to be included in GGA exchange functionals. As often these
  functionals are written as a function of x = |grad n|/n^(4/3), this
  routine performs the necessary conversions between a functional of x
  and of rho.
************************************************************************/

#ifndef XC_DIMENSIONS
#  define XC_DIMENSIONS 3
#endif

static void
#ifdef XC_KINETIC_FUNCTIONAL
work_gga_k
#else
work_gga_x
#endif
#if defined CUDA || defined CUDA_MPIV
(const xc_func_type *p, int np, const double *rho, const double *sigma,
 double *zk, double *vrho, double *vsigma,
 double *v2rho2, double *v2rhosigma, double *v2sigma2,
 double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3, void *gpu_work_params)
#else
(const xc_func_type *p, int np, const double *rho, const double *sigma,
 double *zk, double *vrho, double *vsigma,
 double *v2rho2, double *v2rhosigma, double *v2sigma2,
 double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
#endif
{
  xc_gga_work_x_t r;

  double sfact, sfact2, x_factor_c, alpha, beta, dens;
  int is, is2, ip;

  /* constants for the evaluation of the different terms */
  double c_zk[1];
  double c_vrho[3], c_vsigma[2];
  double c_v2rho2[3], c_v2rhosigma[4], c_v2sigma2[2];
  double c_v3rho3[4], c_v3rho2sigma[3], c_v3rhosigma2[3], c_v3sigma3[3];

  /* variables used inside the is loop */
  double gdm, ds, rhoLDA;

  /* alpha is the power of rho in the corresponding LDA
     beta  is the power of rho in the expression for x */

  beta = 1.0 + 1.0/XC_DIMENSIONS; /* exponent of the density in expression for x */

#ifndef XC_KINETIC_FUNCTIONAL
  alpha = beta;

#  if XC_DIMENSIONS == 2
  x_factor_c = -X_FACTOR_2D_C;
#  else /* three dimensions */
  x_factor_c = -X_FACTOR_C;
#  endif

#else

#  if XC_DIMENSIONS == 2
#  else /* three dimensions */
  alpha = 5.0/3.0;
  x_factor_c = K_FACTOR_C;
#  endif

#endif

  /* Initialize memory */
  memset(&r, 0, sizeof(r));

  sfact = (p->nspin == XC_POLARIZED) ? 1.0 : 2.0;
  sfact2 = sfact*sfact;

  /* Initialize several constants */
  r.order = -1;
  if(zk     != NULL){
    r.order = 0;
    c_zk[0] = sfact*x_factor_c;
  }
  if(vrho   != NULL){
    r.order = 1;
    c_vrho[0]   =  x_factor_c*alpha;
    c_vrho[1]   = -x_factor_c*beta;
    c_vrho[2]   =  x_factor_c;
    c_vsigma[0] =  sfact*x_factor_c;
    c_vsigma[1] =  sfact*x_factor_c;
  }
  if(v2rho2 != NULL){
    r.order = 2;
    c_v2rho2[0] = (x_factor_c/sfact) * (alpha - 1.0)*alpha;
    c_v2rho2[1] = (x_factor_c/sfact) * beta*(beta - 2.0*alpha + 1.0);
    c_v2rho2[2] = (x_factor_c/sfact) * beta*beta;
    c_v2rhosigma[0] =  x_factor_c * (alpha - beta)/2.0;
    c_v2rhosigma[1] = -x_factor_c * beta/2.0;
    c_v2rhosigma[2] =  x_factor_c * alpha;
    c_v2rhosigma[3] = -x_factor_c * beta;
    c_v2sigma2[0] = x_factor_c*sfact / 4.0; 
    c_v2sigma2[1] = x_factor_c*sfact;
  }
  if(v3rho3 != NULL){
    r.order = 3;
    c_v3rho3[0] =  (x_factor_c/sfact2) * (alpha - 2.0)*(alpha - 1.0)*alpha;
    c_v3rho3[1] = -(x_factor_c/sfact2) * (3.0*alpha*alpha - 3.0*alpha*(2.0 + beta) + (1.0 + beta)*(2.0 + beta))*beta;
    c_v3rho3[2] = -(x_factor_c/sfact2) * 3.0*(1.0 - alpha + beta)*beta*beta;
    c_v3rho3[3] = -(x_factor_c/sfact2) * beta*beta*beta;
    c_v3rho2sigma[0] = (x_factor_c/sfact) * (alpha - beta - 1.0)*(alpha - beta)/2.0;
    c_v3rho2sigma[1] = (x_factor_c/sfact) * (1.0 - 2.0*alpha + 3.0*beta)*beta/2.0;
    c_v3rho2sigma[2] = (x_factor_c/sfact) * beta*beta/2.0;
    c_v3rhosigma2[0] = -x_factor_c * (alpha - beta)/4.0;
    c_v3rhosigma2[1] =  x_factor_c * (alpha - beta)/4.0;
    c_v3rhosigma2[2] = -x_factor_c * beta/4.0;
    c_v3sigma3[0] =  x_factor_c*sfact * 3.0/8.0;
    c_v3sigma3[1] = -x_factor_c*sfact * 3.0/8.0;
    c_v3sigma3[2] =  x_factor_c*sfact /8.0;
  }
  if(r.order < 0) return;

#if defined CUDA || defined CUDA_MPIV

#define GET_XSTR(fval) GET_STR(fval)
#define GET_STR(fval) #fval

	char fname[50] = GET_XSTR(func); 
	//gpu_ggax_work_params* w;
	//w = (gpu_ggax_work_params*)gpu_work_params;

	//w->pkernel = (point_kernel) func;

	//memcpy (w->fname, fname,50);

	//printf("Value of fname variable: %s \n", w->fname);
		

	set_gpu_ggax_work_params(sfact, p->dens_threshold, alpha,
        beta, c_zk[0], c_vrho[0], c_vrho[1], c_vrho[2], c_vsigma[0], c_vsigma[1], kernel_id, (gpu_ggax_work_params*)gpu_work_params);

#else
  /* the loop over the points starts */
  for(ip = 0; ip < np; ip++){
    dens = (p->nspin == XC_UNPOLARIZED) ? rho[0] : rho[0] + rho[1];
    if(dens < p->dens_threshold) goto end_ip_loop;

    for(is=0; is<p->nspin; is++){
      is2 = 2*is;

      if(rho[is] < p->dens_threshold) continue;

      gdm    = max(sqrt(sigma[is2])/sfact, p->dens_threshold);
      ds     = rho[is]/sfact;
      rhoLDA = pow(ds, alpha);
      r.x    = gdm/pow(ds, beta);

      func(p, &r);

      if(r.order > 0) r.dfdx   *= r.x;
      if(r.order > 1) r.d2fdx2 *= r.x*r.x;
      if(r.order > 2) r.d3fdx3 *= r.x*r.x*r.x;

      if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
	*zk += rhoLDA*
	  c_zk[0]*r.f;
      
      if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC)){
	vrho[is] += (rhoLDA/ds)*
	  (c_vrho[0]*r.f + c_vrho[1]*r.dfdx);
	
	if(gdm > p->dens_threshold)
	  vsigma[is2] = rhoLDA*
	    (c_vsigma[0]*r.dfdx/(2.0*sigma[is2]));
      }
      
      if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC)){
	v2rho2[is2] = rhoLDA/(ds*ds) * (c_v2rho2[0]*r.f + c_v2rho2[1]*r.dfdx + c_v2rho2[2]*r.d2fdx2);
	
	if(gdm > p->dens_threshold){
	  v2rhosigma[is*5] = (rhoLDA/ds) *
	    ((c_v2rhosigma[0]*r.dfdx + c_v2rhosigma[1]*r.d2fdx2)/sigma[is2]);
	  v2sigma2  [is*5] = rhoLDA*
	    (c_v2sigma2[0]*(r.d2fdx2 - r.dfdx)/(sigma[is2]*sigma[is2]));
	}
      }

      if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC)){
	v3rho3[is*3] = rhoLDA/(ds*ds*ds) *
	  (c_v3rho3[0]*r.f + c_v3rho3[1]*r.dfdx + c_v3rho3[2]*r.d2fdx2 + c_v3rho3[3]*r.d3fdx3);

	if(gdm > p->dens_threshold){
	  v3rho2sigma[is*8] = rhoLDA/(ds*ds) *
	    (c_v3rho2sigma[0]*r.dfdx + c_v3rho2sigma[1]*r.d2fdx2 + c_v3rho2sigma[2]*r.d3fdx3)/sigma[is2];

	  v3rhosigma2[is*11] = (rhoLDA/ds) *
	    (c_v3rhosigma2[0]*r.dfdx + c_v3rhosigma2[1]*r.d2fdx2 + c_v3rhosigma2[2]*r.d3fdx3)/(sigma[is2]*sigma[is2]);

	  v3sigma3[is*9] = rhoLDA*
	    (c_v3sigma3[0]*r.dfdx + c_v3sigma3[1]*r.d2fdx2 + c_v3sigma3[2]*r.d3fdx3)/(sigma[is2]*sigma[is2]*sigma[is2]);
	}
      }
    }

    if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
      *zk /= dens; /* we want energy per particle */
    
  end_ip_loop:
    /* increment pointers */
    rho   += p->n_rho;
    sigma += p->n_sigma;
    
    if(zk != NULL)
      zk += p->n_zk;
    
    if(vrho != NULL){
      vrho   += p->n_vrho;
      vsigma += p->n_vsigma;
    }

    if(v2rho2 != NULL){
      v2rho2     += p->n_v2rho2;
      v2rhosigma += p->n_v2rhosigma;
      v2sigma2   += p->n_v2sigma2;
    }

    if(v3rho3 != NULL){
      v3rho3      += p->n_v3rho3;
      v3rho2sigma += p->n_v3rho2sigma;
      v3rhosigma2 += p->n_v3rhosigma2;
      v3sigma3    += p->n_v3sigma3;
    }
  }

#endif

}
