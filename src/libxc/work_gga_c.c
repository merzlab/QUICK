/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"

static void 
#ifdef CUDA 
work_gga_c(const xc_func_type *p, int np, const double *rho, const double *sigma,
           double *zk, double *vrho, double *vsigma,
           double *v2rho2, double *v2rhosigma, double *v2sigma2,
           double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3, void *gpu_work_params)
#else
work_gga_c(const xc_func_type *p, int np, const double *rho, const double *sigma,
	   double *zk, double *vrho, double *vsigma,
	   double *v2rho2, double *v2rhosigma, double *v2sigma2,
	   double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
#endif
{
  xc_gga_work_c_t r;
  double min_grad2 = p->dens_threshold * p->dens_threshold;
  int ip;

  double drs, dxtdn, dxtds, ndzdn[2], dxsdn[2], dxsds[2];;
  double d2rs, d2xtdn2, d2xtds2, d2xtdns, d2xsdn2[2], d2xsds2[2], d2xsdns[2];
  double d3rs, d3xtdn3, d3xtdn2s, d3xtdns2, d3xtds3, d3xsdn3[2], d3xsdn2s[2], d3xsdns2[2], d3xsds3[2];

  /* set all elements of r to zero */
  memset(&r, 0, sizeof(r));

  r.order = -1;
  if(zk     != NULL) r.order = 0;
  if(vrho   != NULL) r.order = 1;
  if(v2rho2 != NULL) r.order = 2;
  if(v3rho3 != NULL) r.order = 3;

  if(r.order < 0) return;

  for(ip = 0; ip < np; ip++){
    xc_rho2dzeta(p->nspin, rho, &(r.dens), &(r.z));

    if(r.dens < p->dens_threshold) goto end_ip_loop;
    
    r.rs = RS(r.dens);
    if(p->nspin == XC_UNPOLARIZED){
      r.ds[0]  = r.dens/2.0;
      r.ds[1]  = r.ds[0];

      r.sigmat = max(min_grad2, sigma[0]);
      r.xt     = sqrt(r.sigmat)/ pow(r.dens, 4.0/3.0);

      r.sigmas[0] = r.sigmat/4.0;
      r.sigmas[1] = r.sigmas[0];
      r.sigmas[2] = r.sigmas[0];

      r.xs[0]  = CBRT(2.0)*r.xt;
      r.xs[1]  = r.xs[0];
    }else{
      /* there are lots of derivatives that involve inverse
         powers of (1 +- z). For these not to give NaN, we
         must have abs(1 +- z) > DBL_EPSILON                 */
      if(1.0 + r.z < DBL_EPSILON) r.z = -1.0 + DBL_EPSILON;
      if(1.0 - r.z < DBL_EPSILON) r.z =  1.0 - DBL_EPSILON;

      r.ds[0]  = max(p->dens_threshold, rho[0]);
      r.ds[1]  = max(p->dens_threshold, rho[1]);
      
      r.sigmat = max(min_grad2, sigma[0] + 2.0*sigma[1] + sigma[2]);
      r.xt     = sqrt(r.sigmat)/ pow(r.dens, 4.0/3.0);
      
      r.sigmas[0] = max(min_grad2, sigma[0]);
      r.sigmas[1] = max(min_grad2, sigma[1]);
      r.sigmas[2] = max(min_grad2, sigma[2]);

      r.xs[0] = sqrt(r.sigmas[0])/pow(r.ds[0], 4.0/3.0);
      r.xs[1] = sqrt(r.sigmas[2])/pow(r.ds[1], 4.0/3.0);
    }
  
    func(p, &r);

    if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
      *zk = r.f;

    if(r.order < 1) goto end_ip_loop;
    
    /* setup auxiliary variables */
    drs   =     -r.rs/(3.0*r.dens);
    dxtdn = -4.0*r.xt/(3.0*r.dens);
    dxtds = r.xt/(2.0*r.sigmat);
    if(p->nspin == XC_POLARIZED){
      ndzdn[1] = -(r.z + 1.0);
      ndzdn[0] = -(r.z - 1.0);

      dxsdn[1] = -4.0/3.0*r.xs[1]/r.ds[1];
      dxsdn[0] = -4.0/3.0*r.xs[0]/r.ds[0];

      dxsds[1] = r.xs[1]/(2.0*r.sigmas[2]);
      dxsds[0] = r.xs[0]/(2.0*r.sigmas[0]);
    }else{
      dxsdn[0] = M_CBRT2*dxtdn;
      dxsds[0] = M_CBRT2*dxtds;
    }

    if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC)){
      vrho[0]   = r.f + r.dens*(r.dfdrs*drs + r.dfdxt*dxtdn);
      vsigma[0] = r.dens*r.dfdxt*dxtds;

      if(p->nspin == XC_POLARIZED){
	vrho[1] = vrho[0] + r.dfdz*ndzdn[1] + r.dens*r.dfdxs[1]*dxsdn[1];
	vrho[0] = vrho[0] + r.dfdz*ndzdn[0] + r.dens*r.dfdxs[0]*dxsdn[0];;

	vsigma[2] = vsigma[0] + r.dens*r.dfdxs[1]*dxsds[1];
	vsigma[1] = 2.0*vsigma[0];
	vsigma[0] = vsigma[0] + r.dens*r.dfdxs[0]*dxsds[0];
	
      }else{
	vrho[0]   += 2.0*r.dens*r.dfdxs[0]*dxsdn[0]; /* factor of 2 comes from sum over sigma */
	vsigma[0] += 2.0*r.dens*r.dfdxs[0]*dxsds[0];
      }
    }

    if(r.order < 2) goto end_ip_loop;
  
    /* setup auxiliary variables */
    d2rs    = -4.0*drs/(3.0*r.dens);
    d2xtdn2 = -7.0*dxtdn/(3.0*r.dens);
    d2xtds2 = -dxtds/(2.0*r.sigmat);
    d2xtdns =  dxtdn/(2.0*r.sigmat);
    if(p->nspin == XC_POLARIZED){
      d2xsdn2[0] = -7.0*dxsdn[0]/(3.0*r.ds[0]);
      d2xsdn2[1] = -7.0*dxsdn[1]/(3.0*r.ds[1]);

      d2xsdns[0] = -4.0/3.0*dxsds[0]/r.ds[0];
      d2xsdns[1] = -4.0/3.0*dxsds[1]/r.ds[1];

      d2xsds2[0] = -dxsds[0]/(2.0*r.sigmas[0]);
      d2xsds2[1] = -dxsds[1]/(2.0*r.sigmas[2]);
    }else{
      d2xsdn2[0] = M_CBRT2*d2xtdn2;
      d2xsdns[0] = M_CBRT2*d2xtdns;
      d2xsds2[0] = M_CBRT2*d2xtds2;
    }

    if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC)){
      v2rho2[0]     = 2.0*r.dfdrs*drs + 2.0*r.dfdxt*dxtdn +
	r.dens*(r.d2fdrs2*drs*drs + r.d2fdxt2*dxtdn*dxtdn + r.dfdrs*d2rs + r.dfdxt*d2xtdn2 + 2.0*r.d2fdrsxt*drs*dxtdn);

      v2sigma2[0]   = r.dens*(r.d2fdxt2*dxtds*dxtds + r.dfdxt*d2xtds2);

      v2rhosigma[0] = r.dfdxt*dxtds + r.dens*(r.d2fdrsxt*drs*dxtds + r.d2fdxt2*dxtdn*dxtds + r.dfdxt*d2xtdns);

      if(p->nspin == XC_POLARIZED){
	int is;

        for(is=2; is>=0; is--){
	  int s1 = (is > 1) ?  1 :  0;         /* {0, 0, 1}[is] */
	  int s2 = (is > 0) ?  1 :  0;         /* {0, 1, 1}[is] */

	  v2rho2[is]  = v2rho2[0];

	  v2rho2[is] += r.dfdxs[s1]*dxsdn[s1] +
	    ndzdn[s1]*(r.d2fdrsz*drs + r.d2fdzxt*dxtdn + r.d2fdzxs[s2]*dxsdn[s2]) +
	    r.dens*(r.d2fdrsxs[s1]*drs*dxsdn[s1] + r.d2fdxtxs[s1]*dxtdn*dxsdn[s1]);

	  v2rho2[is] += r.dfdxs[s2]*dxsdn[s2] +
	    ndzdn[s2]*(r.d2fdrsz*drs + r.d2fdzxt*dxtdn + r.d2fdzxs[s1]*dxsdn[s1]) +
	    r.dens*(r.d2fdrsxs[s2]*drs*dxsdn[s2] + r.d2fdxtxs[s2]*dxtdn*dxsdn[s2]);

	  v2rho2[is] += r.d2fdz2*ndzdn[s1]*ndzdn[s2]/r.dens + r.dens*r.d2fdxs2[is]*dxsdn[s1]*dxsdn[s2];

	  if(is != 1)
	    v2rho2[is] += r.dens*r.dfdxs[s1]*d2xsdn2[s1];
	}

	/* v2sigma */
	v2sigma2[5] =     v2sigma2[0] + r.dens*
	  (2.0*r.d2fdxtxs[1]*dxtds*dxsds[1] + r.d2fdxs2[2]*dxsds[1]*dxsds[1] + r.dfdxs[1]*d2xsds2[1]);
	v2sigma2[4] = 2.0*v2sigma2[0] + r.dens*
	  (2.0*r.d2fdxtxs[1]*dxtds*dxsds[1]);
	v2sigma2[3] = 4.0*v2sigma2[0];
	v2sigma2[2] =     v2sigma2[0] + r.dens*
	  (    dxtds*(r.d2fdxtxs[0]*dxsds[0] + r.d2fdxtxs[1]*dxsds[1]) + r.d2fdxs2[1]*dxsds[0]*dxsds[1]);
	v2sigma2[1] =  2.0*v2sigma2[0] + r.dens*
	  (2.0*r.d2fdxtxs[0]*dxtds*dxsds[0]);
	v2sigma2[0] =     v2sigma2[0] + r.dens*
	  (2.0*r.d2fdxtxs[0]*dxtds*dxsds[0] + r.d2fdxs2[0]*dxsds[0]*dxsds[0] + r.dfdxs[0]*d2xsds2[0]);

	/* v2rhosigma */
	v2rhosigma[5] =     v2rhosigma[0] + r.dfdxs[1]*dxsds[1] + ndzdn[1]*(r.d2fdzxt*dxtds + r.d2fdzxs[1]*dxsds[1]) + 
	  r.dens*(r.d2fdrsxs[1]*drs*dxsds[1] + r.d2fdxtxs[1]*(dxsdn[1]*dxtds + dxtdn*dxsds[1]) + r.d2fdxs2[2]*dxsdn[1]*dxsds[1] +
		r.dfdxs[1]*d2xsdns[1]);

	v2rhosigma[4] = 2.0*v2rhosigma[0] + 2.0*ndzdn[1]*r.d2fdzxt*dxtds + 
		  2.0*r.dens*r.d2fdxtxs[1]*dxsdn[1]*dxtds;

	v2rhosigma[3] =     v2rhosigma[0] + r.dfdxs[0]*dxsds[0] + ndzdn[1]*(r.d2fdzxt*dxtds + r.d2fdzxs[0]*dxsds[0]) + 
	  r.dens*(r.d2fdrsxs[0]*drs*dxsds[0] + r.d2fdxtxs[1]*(dxsdn[1]*dxtds + dxtdn*dxsds[0]) + r.d2fdxs2[1]*dxsdn[1]*dxsds[0]);

	v2rhosigma[2] =     v2rhosigma[0] + r.dfdxs[1]*dxsds[1] + ndzdn[0]*(r.d2fdzxt*dxtds + r.d2fdzxs[1]*dxsds[1]) + 
	  r.dens*(r.d2fdrsxs[1]*drs*dxsds[1] + r.d2fdxtxs[0]*(dxsdn[0]*dxtds + dxtdn*dxsds[1]) + r.d2fdxs2[1]*dxsdn[0]*dxsds[1]);

	v2rhosigma[1] = 2.0*v2rhosigma[0] + 2.0*ndzdn[0]*r.d2fdzxt*dxtds + 
	  2.0*r.dens*r.d2fdxtxs[0]*dxsdn[0]*dxtds;
	
	v2rhosigma[0] =     v2rhosigma[0] + r.dfdxs[0]*dxsds[0] + ndzdn[0]*(r.d2fdzxt*dxtds + r.d2fdzxs[0]*dxsds[0]) + 
	  r.dens*(r.d2fdrsxs[0]*drs*dxsds[0] + r.d2fdxtxs[0]*(dxsdn[0]*dxtds + dxtdn*dxsds[0]) + r.d2fdxs2[0]*dxsdn[0]*dxsds[0] + 
		r.dfdxs[0]*d2xsdns[0]);

      }else{
	v2rho2[0]     += 2.0*dxsdn[0]*
	  (2.0*r.dfdxs[0] + r.dens*(2.0*r.d2fdrsxs[0]*drs + 2.0*r.d2fdxtxs[0]*dxtdn + (r.d2fdxs2[0] + r.d2fdxs2[1])*dxsdn[0]))
	  + 2.0*r.dens*r.dfdxs[0]*d2xsdn2[0];

	v2sigma2[0]   += 2.0*r.dens*((r.d2fdxs2[0] + r.d2fdxs2[1])*dxsds[0]*dxsds[0] + r.dfdxs[0]*d2xsds2[0] + 2.0*r.d2fdxtxs[0]*dxtds*dxsds[0]);

	v2rhosigma[0] += 2.0*r.dens*r.d2fdxtxs[0]*(dxsdn[0]*dxtds + dxtdn*dxsds[0]) +
	  2.0*(r.dfdxs[0] + r.dens*(r.d2fdrsxs[0]*drs + (r.d2fdxs2[0] + r.d2fdxs2[1])*dxsdn[0]))*dxsds[0]
	  + 2.0*r.dens*r.dfdxs[0]*d2xsdns[0];
      }
    }

    if(r.order < 3) goto end_ip_loop;
  
    /* setup auxiliary variables */
    d3rs     =  -7.0*d2rs/(3.0*r.dens);

    d3xtdn3  = -10.0*d2xtdn2/(3.0*r.dens);
    d3xtdn2s =       d2xtdn2/(2.0*r.sigmat);
    d3xtdns2 =      -d2xtdns/(2.0*r.sigmat);
    d3xtds3  =  -3.0*d2xtds2/(2.0*r.sigmat);

    if(p->nspin == XC_POLARIZED){
      /* not done */
      d3xsdn3[0] = -7.0*dxsdn[0]/(3.0*r.ds[0]);
      d3xsdn3[1] = -7.0*dxsdn[1]/(3.0*r.ds[1]);

      d3xsdn2s[0] = -4.0/3.0*dxsds[0]/r.ds[0];
      d3xsdn2s[1] = -4.0/3.0*dxsds[1]/r.ds[1];

      d3xsds3[0] = -dxsds[0]/(2.0*r.sigmas[0]);
      d3xsds3[1] = -dxsds[1]/(2.0*r.sigmas[2]);
    }else{
      d3xsdn3[0]  = M_CBRT2*d3xtdn3;
      d3xsdn2s[0] = M_CBRT2*d3xtdn2s;
      d3xsdns2[0] = M_CBRT2*d3xtdns2;
      d3xsds3[0]  = M_CBRT2*d3xtds3;
    }

    if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC)){
      v3rho3[0]     = 
	r.dfdrs     * (3.0*d2rs    + r.dens*d3rs)   +
	r.dfdxt     * (3.0*d2xtdn2 + r.dens*d3xtdn3) +
	r.d2fdrs2   * 3.0*drs*(drs + r.dens*d2rs) +
	r.d2fdrsxt  * (6.0*drs*dxtdn + 3.0*r.dens*d2rs*dxtdn + 3.0*r.dens*drs*d2xtdn2) +
	r.d2fdxt2   * 3.0*dxtdn*(dxtdn + r.dens*d2xtdn2) +
	r.d3fdrs3   * r.dens*drs*drs*drs +
	r.d3fdrs2xt * 3.0*r.dens*drs*drs*dxtdn +
	r.d3fdrsxt2 * 3.0*r.dens*drs*dxtdn*dxtdn +
	r.d3fdxt3   * r.dens*dxtdn*dxtdn*dxtdn;

      v3rhosigma2[0] = r.dfdxt*d2xtds2 + r.d2fdxt2*dxtds*dxtds +
	r.dens*(r.dfdxt*d3xtdns2 + r.d2fdxt2*(d2xtds2*dxtdn + 2.0*dxtds*d2xtdns) + r.d2fdrsxt*d2xtds2*drs +
		dxtds*dxtds*(r.d3fdxt3*dxtdn + r.d3fdrsxt2*drs));

      v3rho2sigma[0] = 2.0*r.dfdxt*d2xtdns + 2.0*r.d2fdxt2*dxtds*dxtdn + 2.0*r.d2fdrsxt*drs*dxtds +
	r.dens*(r.dfdxt*d3xtdn2s + r.d2fdxt2*(d2xtdn2*dxtds + 2.0*dxtdn*d2xtdns) + r.d2fdrsxt*(2.0*drs*d2xtdns + d2rs*dxtds) +
		r.d3fdrs2xt*drs*drs*dxtds + 2.0*r.d3fdrsxt2*drs*dxtdn*dxtds + r.d3fdxt3*dxtdn*dxtdn*dxtds);

      v3sigma3[0]    = r.dens*(r.d3fdxt3*dxtds*dxtds*dxtds + 3.0*r.d2fdxt2*d2xtds2*dxtds + r.dfdxt*d3xtds3);

      if(p->nspin == XC_POLARIZED){
	int is;

        for(is=3; is>=0; is--){
	  int s1 = (is > 2) ?  1 :  0;         /* {0, 0, 0, 1}[is] */
	  int s2 = (is > 1) ?  1 :  0;         /* {0, 0, 1, 1}[is] */
	  int s3 = (is > 0) ?  1 :  0;         /* {0, 1, 1, 1}[is] */

	  int s12  = s1 + s2;                   /* 0 + 0 = 0         */
	  int s13  = s1 + s3;                   /* 0 + 1 = 1 + 0 = 1 */
	  int s23  = s2 + s3;                   /* 1 + 1 = 2         */
	  int s123 = s1 + s2 + s3;

	  v3rho3[is]  = v3rho3[0];

	  if(s1 == s2){
	    v3rho3[is] += 
	      r.dfdxs[s1]    * d2xsdn2[s1] +
	      r.d2fdxs2[s23] * r.dens*d2xsdn2[s2]*dxsdn[s3];
	  }
	  
	  if(s2 == s3){
	    v3rho3[is] += 
	      r.dfdxs[s2]    * d2xsdn2[s2] +
	      r.d2fdxs2[s13] * r.dens*d2xsdn2[s3]*dxsdn[s1];
	  }

	  if(s1 == s3){
	    v3rho3[is] += 
	      r.dfdxs[s3]    * d2xsdn2[s3] +
	      r.d2fdxs2[s12] * r.dens*d2xsdn2[s1]*dxsdn[s2];
	  }

	  if(s1 == s2 && s2 == s3)
	    v3rho3[is] += r.dens*r.dfdxs[s1] * d3xsdn3[s1];

	  v3rho3[0]    +=
	    r.d2fdz2        * (-ndzdn[s1]*ndzdn[s2] - ndzdn[s1]*ndzdn[s3] - ndzdn[s2]*ndzdn[s3])/(r.dens*r.dens) +
	    r.d3fdz3        * ndzdn[s1]*ndzdn[s2]*ndzdn[s3]/(r.dens*r.dens) +
	    r.d3fdxs3[s123] * r.dens*dxsdn[s1]*dxsdn[s2]*dxsdn[s3] +
	    r.d3fdrs2z      * drs*drs*(ndzdn[s1] + ndzdn[s2] + ndzdn[s3]) +
	    r.dens*drs*drs*(r.d3fdrs2xs[s1]*dxsdn[s1] + r.d3fdrs2xs[s2]*dxsdn[s2] + r.d3fdrs2xs[s3]*dxsdn[s3]) +
	    r.d3fdzxt2      * dxtdn*dxtdn*(ndzdn[s1] + ndzdn[s2] + ndzdn[s3]) +
	    r.dens*dxtdn*dxtdn*(r.d3fdxt2xs[s1]*dxsdn[s1] + r.d3fdxt2xs[s2]*dxsdn[s2] + r.d3fdxt2xs[s3]*dxsdn[s3]) +
	    r.d3fdrsz2      * drs  *(ndzdn[s1]*ndzdn[s2] + ndzdn[s1]*ndzdn[s3] + ndzdn[s2]*ndzdn[s3])/r.dens +
	    r.d3fdz2xt      * dxtdn*(ndzdn[s1]*ndzdn[s2] + ndzdn[s1]*ndzdn[s3] + ndzdn[s2]*ndzdn[s3])/r.dens +
	    r.d3fdz2xs[s1]  * ndzdn[s2]*ndzdn[s3]*dxsdn[s1]/r.dens +
	    r.d3fdz2xs[s2]  * ndzdn[s1]*ndzdn[s3]*dxsdn[s2]/r.dens +
	    r.d3fdz2xs[s3]  * ndzdn[s1]*ndzdn[s2]*dxsdn[s3]/r.dens +
	    r.dens*drs  *(r.d3fdrsxs2[s12]*dxsdn[s1]*dxsdn[s2] + r.d3fdrsxs2[s13]*dxsdn[s1]*dxsdn[s3] + r.d3fdrsxs2[s23]*dxsdn[s2]*dxsdn[s3]) +
	    r.dens*dxtdn*(r.d3fdxtxs2[s12]*dxsdn[s1]*dxsdn[s2] + r.d3fdxtxs2[s13]*dxsdn[s1]*dxsdn[s3] + r.d3fdxtxs2[s23]*dxsdn[s2]*dxsdn[s3])
	    ;
	    
	    }
	  
      }else{
      }
    }

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
}
