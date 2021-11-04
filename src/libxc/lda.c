/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"
#include "funcs_lda.c"

/* get the lda functional */
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
void xc_lda(const xc_func_type *func, int np, const double *rho, double *zk, double *vrho, double *v2rho2, double *v3rho3, void *gpu_work_params)
#else
void 
xc_lda(const xc_func_type *func, int np, const double *rho, double *zk, double *vrho, double *v2rho2, double *v3rho3)
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

  if(v3rho3 != NULL && !(func->info->flags & XC_FLAGS_HAVE_KXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of kxc\n",
	    func->info->name);
    exit(1);
  }

  /* initialize output */
  if(zk != NULL)
    memset(zk,     0, np*sizeof(double)*func->n_zk);

  if(vrho != NULL)
    memset(vrho,   0, np*sizeof(double)*func->n_vrho);

  if(v2rho2 != NULL)
    memset(v2rho2, 0, np*sizeof(double)*func->n_v2rho2);

  if(v3rho3 != NULL)
    memset(v3rho3, 0, np*sizeof(double)*func->n_v3rho3);


  assert(func->info!=NULL && func->info->lda!=NULL);

  /* call the LDA routines */
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  func->info->lda(func, np, rho, zk, vrho, v2rho2, v3rho3, gpu_work_params);
#else
  func->info->lda(func, np, rho, zk, vrho, v2rho2, v3rho3);
#endif
}


/* specializations */
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
void
xc_lda_exc(const xc_func_type *p, int np, const double *rho, double *zk, void *gpu_work_params)
{
  xc_lda(p, np, rho, zk, NULL, NULL, NULL, gpu_work_params);
}

void
xc_lda_exc_vxc(const xc_func_type *p, int np, const double *rho, double *zk, double *vrho, void *gpu_work_params)
{
  xc_lda(p, np, rho, zk, vrho, NULL, NULL, gpu_work_params);
}

void
xc_lda_vxc(const xc_func_type *p, int np, const double *rho, double *vrho, void *gpu_work_params)
{
  xc_lda(p, np, rho, NULL, vrho, NULL, NULL, gpu_work_params);
}

void
xc_lda_fxc(const xc_func_type *p, int np, const double *rho, double *v2rho2, void *gpu_work_params)
{
  xc_lda(p, np, rho, NULL, NULL, v2rho2, NULL, gpu_work_params);
}

void
xc_lda_kxc(const xc_func_type *p, int np, const double *rho, double *v3rho3, void *gpu_work_params)
{
  xc_lda(p, np, rho, NULL, NULL, NULL, v3rho3, gpu_work_params);
}
#else
void
xc_lda_exc(const xc_func_type *p, int np, const double *rho, double *zk)
{
  xc_lda(p, np, rho, zk, NULL, NULL, NULL);
}

void
xc_lda_exc_vxc(const xc_func_type *p, int np, const double *rho, double *zk, double *vrho)
{
  xc_lda(p, np, rho, zk, vrho, NULL, NULL);
}

void
xc_lda_vxc(const xc_func_type *p, int np, const double *rho, double *vrho)
{
  xc_lda(p, np, rho, NULL, vrho, NULL, NULL);
}

void
xc_lda_fxc(const xc_func_type *p, int np, const double *rho, double *v2rho2)
{
  xc_lda(p, np, rho, NULL, NULL, v2rho2, NULL);
}

void
xc_lda_kxc(const xc_func_type *p, int np, const double *rho, double *v3rho3)
{
  xc_lda(p, np, rho, NULL, NULL, NULL, v3rho3);
}
#endif

#define DELTA_RHO 1e-6

/* get the xc kernel through finite differences */
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
void
xc_lda_fxc_fd(const xc_func_type *func, int np, const double *rho, double *v2rho2, void *gpu_work_params)
#else
void 
xc_lda_fxc_fd(const xc_func_type *func, int np, const double *rho, double *v2rho2)
#endif
{
  int i, ip;

  assert(func != NULL);

  for(ip=0; ip<np; ip++){
    for(i=0; i<func->nspin; i++){
      double rho2[2], vc1[2], vc2[2];
      int j, js;
      
      j  = (i+1) % 2;
      js = (i==0) ? 0 : 2;
      
      rho2[i] = rho[i] + DELTA_RHO;
      rho2[j] = (func->nspin == XC_POLARIZED) ? rho[j] : 0.0;
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
      xc_lda_vxc(func, 1, rho2, vc1, gpu_work_params);
#else
     xc_lda_vxc(func, 1, rho2, vc1);
#endif 
      if(rho[i]<2.0*DELTA_RHO){ /* we have to use a forward difference */
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
	xc_lda_vxc(func, 1, rho, vc2, gpu_work_params);
#else
	xc_lda_vxc(func, 1, rho, vc2);
#endif	
	v2rho2[js] = (vc1[i] - vc2[i])/(DELTA_RHO);
	if(func->nspin == XC_POLARIZED && i==0)
	  v2rho2[1] = (vc1[j] - vc2[j])/(DELTA_RHO);
	
      }else{                    /* centered difference (more precise)  */
	rho2[i] = rho[i] - DELTA_RHO;
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
	xc_lda_vxc(func, 1, rho2, vc2, gpu_work_params);
#else
	xc_lda_vxc(func, 1, rho2, vc2);
#endif     
	v2rho2[js] = (vc1[i] - vc2[i])/(2.0*DELTA_RHO);
	if(func->nspin == XC_POLARIZED && i==0)
	  v2rho2[1] = (vc1[j] - vc2[j])/(2.0*DELTA_RHO);
      }
    }

    rho    += func->n_rho;
    v2rho2 += func->n_v2rho2;
  } /* for(ip) */
}

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
void
xc_lda_kxc_fd(const xc_func_type *func, int np, const double *rho, double *v3rho3, void *gpu_work_params)
#else
void
xc_lda_kxc_fd(const xc_func_type *func, int np, const double *rho, double *v3rho3)
#endif
{
  /* Kxc, this is a third order tensor with respect to the densities */
  int ip, i, j, n;

  assert(func != NULL);

  for(ip=0; ip<np; ip++){
    for(i=0; i<func->nspin; i++){
      double rho2[2], vc1[2], vc2[2], vc3[2];

      for(n=0; n<func->nspin; n++) rho2[n] = rho[n];
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
      xc_lda_vxc(func, 1, rho, vc2, gpu_work_params);
#else
	xc_lda_vxc(func, 1, rho, vc2);
#endif
      rho2[i] += DELTA_RHO;
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
      xc_lda_vxc(func, 1, rho2, vc1, gpu_work_params);
#else
	xc_lda_vxc(func, 1, rho2, vc1);
#endif	
      rho2[i] -= 2.0*DELTA_RHO;
#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
      xc_lda_vxc(func, 1, rho2, vc3, gpu_work_params);    
#else
	xc_lda_vxc(func, 1, rho2, vc3);
#endif    
      for(j=0; j<func->nspin; j++)
	v3rho3[i*func->nspin + j] = (vc1[j] - 2.0*vc2[j] + vc3[j])/(DELTA_RHO*DELTA_RHO);
    }
    
    rho    += func->n_rho;
    v3rho3 += func->n_v3rho3;
  } /* for(ip) */
}
